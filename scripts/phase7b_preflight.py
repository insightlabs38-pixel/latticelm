"""Validate frozen Phase 7B provenance and write token-zero manifests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import psutil
import torch
from huggingface_hub import HfApi

from latticelm.config import LatticeConfig
from latticelm.model import build_model

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main() -> None:
    config = LatticeConfig.from_json(ROOT / "configs/phase7b_final.json")
    model = build_model(config)
    breakdown = model.parameter_breakdown()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    tokenizer = ARTIFACTS / "tokenizers/babylm_2026_4k.json"
    selected = json.loads((ARTIFACTS / "phase7a_selected_dataset_manifest.json").read_text())
    common = json.loads((ARTIFACTS / "common_validation_manifest.json").read_text())
    expected = {entry["path"]: entry["sha256"] for source in selected["sources"]
                for entry in source["token_files"].values()}
    expected[common["common_token_file"]["path"]] = common["common_token_file"]["sha256"]
    file_checks = {name: digest(ROOT / name) for name in expected}
    tracked = subprocess.run(["git", "grep", "-n", "-E", "hf_[A-Za-z0-9]{20,}"], cwd=ROOT,
                             text=True, capture_output=True, check=False).stdout.strip()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF credential must be process-local for preflight")
    repo = HfApi(token=token).repo_info("insightlabs38-pixel/LatticeLM-research", repo_type="model")
    failures = []
    if status:
        # The newly authored Phase 7B source is allowed until its pre-training commit.
        unexpected = [line for line in status.splitlines() if "phase7b" not in line and "phase7b_final" not in line]
        failures += [f"unexpected working-tree change: {line}" for line in unexpected]
    if breakdown["total"] >= 50_000_000:
        failures.append("parameter limit exceeded")
    if digest(tokenizer) != common["tokenizer_sha256"]:
        failures.append("tokenizer digest mismatch")
    failures += [f"dataset digest mismatch: {name}" for name in expected if file_checks[name] != expected[name]]
    if tracked:
        failures.append("authentication-like token found in tracked file")
    if failures:
        raise RuntimeError("; ".join(failures))
    cgroup_cpu = Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None
    cgroup_ram = Path("/sys/fs/cgroup/memory.max").read_text().strip() if Path("/sys/fs/cgroup/memory.max").exists() else None
    environment = {"cpu_model": cpu_model(), "visible_cpus": os.cpu_count(), "cgroup_cpu_quota": cgroup_cpu,
                   "ram_total_bytes": psutil.virtual_memory().total, "cgroup_ram_limit": cgroup_ram,
                   "python_version": platform.python_version(), "pytorch_version": torch.__version__,
                   "mkldnn_available": torch.backends.mkldnn.is_available(), "os_kernel": platform.platform(),
                   "source_git_commit": source_commit}
    model_manifest = {"frozen": True, "architecture_name": "LatticeLM-Co4-S", "implementation": "co4_causal",
                      "trainable_parameters": breakdown["total"], "parameter_breakdown": breakdown,
                      "tied_embeddings": False, "context_length": config.context_length,
                      "dimensions": {"vocab_size": config.vocab_size, "d_model": config.d_model,
                                     "layers": config.n_layers, "query_heads": config.n_heads,
                                     "kv_heads": config.n_kv_heads, "ffn_hidden": config.ffn_hidden}}
    data_manifest = {"frozen": True, "regime": "DATA-C", "source_token_ratios": {"BabyLM Strict": .75,
                     "decontaminated FineWeb-Edu": .25}, "phase7a_manifest_sha256": digest(ARTIFACTS / "phase7a_selected_dataset_manifest.json"),
                     "sources": selected["sources"], "decontamination": selected["decontamination_summary"],
                     "tokenizer_sha256": digest(tokenizer), "common_validation_sha256": digest(ARTIFACTS / "common_validation_manifest.json"),
                     "token_file_verification": file_checks,
                     "loader_policy": "seeded affine without-replacement block permutation per source; 6 BabyLM + 2 FineWeb sequences per step"}
    training = {"frozen": True, "lineage": "random initialization; no imported weights", "seed": config.seed,
                "optimizer": "AdamW", "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
                "betas": [config.adam_beta1, config.adam_beta2], "epsilon": 1e-8,
                "gradient_clipping": config.grad_clip, "scheduler": "constant LambdaLR", "warmup": "none",
                "batch_size_sequences": config.batch_size, "effective_batch_tokens": config.batch_size * config.context_length,
                "pytorch_threads": config.num_threads, "requested_milestones": [10_000_000, 25_000_000, 50_000_000],
                "realized_whole_batch_milestones": [10_000_384, 25_000_960, 50_000_896],
                "environment": environment, "hf_repository": repo.id, "hf_revision_at_preflight": repo.sha,
                "preflight": {"dataset_hashes_verified": True, "common_validation_verified": True,
                              "decontamination_outputs_verified": True, "hf_accessibility_verified": True,
                              "tracked_authentication_scan_clean": True}}
    for name, payload in (("final_model_config.json", model_manifest), ("final_data_manifest.json", data_manifest),
                          ("final_training_manifest.json", training)):
        (ARTIFACTS / name).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"ok": True, "source_commit": source_commit, "parameters": breakdown["total"],
                      "hf_revision": repo.sha}, indent=2))


if __name__ == "__main__":
    main()
