"""Safe, integrity-checked Hugging Face model-repository persistence."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model, save_model

from .config import LatticeConfig
from .model import build_model


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualifies_as_remote_best(repo_id: str, token: str, validation_loss: float) -> bool:
    """Return true only when a candidate improves the repository-global best."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id, "best/metrics.json", repo_type="model", token=token)
    except Exception:
        return True
    current = json.loads(Path(path).read_text())
    return validation_loss < float(current["val_loss"])


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _compare_models(source: torch.nn.Module, restored: torch.nn.Module, vocab_size: int) -> None:
    source_state, restored_state = source.state_dict(), restored.state_dict()
    if source_state.keys() != restored_state.keys():
        raise RuntimeError("safetensors round-trip changed state-dict keys")
    for name in source_state:
        if not torch.equal(source_state[name], restored_state[name]):
            raise RuntimeError(f"safetensors round-trip mismatch: {name}")
    source.eval(); restored.eval()
    tokens = torch.arange(16, dtype=torch.long).remainder(vocab_size).view(1, -1)
    with torch.no_grad():
        source_logits = source(tokens)[0]
        restored_logits = restored(tokens)[0]
    if not torch.equal(source_logits, restored_logits):
        raise RuntimeError("safetensors round-trip changed deterministic logits")


def export_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    checkpoint_role: str,
    tokenizer_path: str | Path,
    tokenizer_report_path: str | Path,
    dataset_manifest_path: str | Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Export a training checkpoint and abort unless a fresh-model round-trip is exact."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = LatticeConfig(**checkpoint["config"])
    source = build_model(config)
    source.load_state_dict(checkpoint["model"], strict=True)
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    weights = output / "model.safetensors"
    save_model(source, weights, metadata={"format": "pt", "architecture": config.architecture})
    restored = build_model(config)
    missing, unexpected = load_model(restored, weights, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict safetensors load failed: missing={missing}, unexpected={unexpected}")
    _compare_models(source, restored, config.vocab_size)

    portable_config = config.to_dict()
    if portable_config.get("memory_token_map_path"):
        portable_config["memory_token_map_path"] = "tokenizer.report.json"
    (output / "config.json").write_text(json.dumps(portable_config, indent=2) + "\n")
    shutil.copy2(tokenizer_path, output / "tokenizer.json")
    shutil.copy2(tokenizer_report_path, output / "tokenizer.report.json")
    dataset_manifest = json.loads(Path(dataset_manifest_path).read_text())
    tokenizer_hash = sha256(tokenizer_path)
    breakdown = source.parameter_breakdown()
    manifest = {
        "architecture_name": config.architecture,
        "architecture_version": "latticelm-0.2",
        "source_git_commit": _git_commit(),
        "experiment_id": experiment_id,
        "random_seed": config.seed,
        "total_trainable_parameters": breakdown["total"],
        "neural_parameters": breakdown["total"] - breakdown["conditional_memory"],
        "memory_parameters": breakdown["conditional_memory"],
        "model_config": portable_config,
        "tokenizer": {"file": "tokenizer.json", "sha256": tokenizer_hash,
                      "vocab_size": config.vocab_size, "memory_metadata": "tokenizer.report.json"},
        "training_dataset": dataset_manifest["dataset"],
        "dataset_revision": dataset_manifest["revision"],
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "dataset_sources": dataset_manifest.get("sources", []),
        "training_tokens_seen": int(metrics["tokens_trained"]),
        "optimizer": "AdamW",
        "optimizer_settings": {"learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
                               "memory_lr_multiplier": config.memory_lr_multiplier},
        "wall_clock_seconds": float(metrics["wall_seconds"]),
        "cpu": _cpu_model(),
        "effective_cgroup_cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_loss": float(metrics["val_loss"]),
        "validation_perplexity": float(metrics["val_ppl"]),
        "checkpoint_role": checkpoint_role,
        "safetensors_roundtrip_verified": True,
        "tokenizer_and_memory_metadata": {
            "special_token_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
            "normalization_metadata_file": "tokenizer.report.json",
            "ngram_orders": list(config.memory_orders) if config.architecture == "mini_engram" else [],
            "hash_seed_formula": "10007 * (1 + order_index * memory_heads + head_index)",
            "memory_heads": config.memory_heads if config.architecture == "mini_engram" else 0,
            "slots_per_table": config.memory_slots if config.architecture == "mini_engram" else 0,
            "embedding_dimension_per_head": config.memory_dim if config.architecture == "mini_engram" else 0,
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # A resume artifact is private structured state, deliberately separate from
    # inference weights. Legacy checkpoints may not contain every RNG stream.
    resume = {key: value for key, value in checkpoint.items() if key != "model"}
    resume["model_safetensors_sha256"] = sha256(weights)
    resume["exact_rng_state_available"] = all(key in checkpoint for key in ("torch_rng_state", "data_generator_state"))
    torch.save(resume, output / "resume_state.pt")
    sums = []
    for path in sorted(output.iterdir()):
        if path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return manifest


def upload_checkpoint(local_dir: str | Path, repo_id: str, path_in_repo: str, token: str) -> str:
    """Upload one semantic checkpoint slot and verify every remote file size."""
    # Some managed proxies reject HTTPX's chunked LFS PUTs. Force the official
    # Hub LFS path and supply Content-Length; this remains an HfApi upload, not a
    # manually constructed Hub request.
    import huggingface_hub.constants as hub_constants
    import huggingface_hub.lfs as hub_lfs
    from huggingface_hub import HfApi
    hub_constants.HF_HUB_DISABLE_XET = True

    def content_length_lfs_upload(operation, upload_url: str) -> None:
        with operation.as_file(with_tqdm=True) as fileobj:
            # curl emits a fixed Content-Length through this environment's
            # proxy, whereas HTTPX file/bytes bodies are rewritten as chunked
            # and rejected by the Hub's signed S3 endpoint.
            completed = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--http1.1",
                                        "--header", f"Content-Length: {operation.upload_info.size}",
                                        "--request", "PUT", "--data-binary", f"@{fileobj.name}", upload_url],
                                       capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"signed LFS upload failed: {completed.stderr.strip()}")

    hub_lfs._upload_single_part = content_length_lfs_upload
    api = HfApi(token=token)
    expected_remote: dict[str, int]
    if os.environ.get("LATTICELM_HF_SKIP_LFS") == "1":
        revision, expected_remote = _upload_checkpoint_via_git(
            local_dir, repo_id, path_in_repo, token, RuntimeError("LFS disabled by environment"))
    else:
        try:
            commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_dir),
                                       path_in_repo=path_in_repo, commit_message=f"Update {path_in_repo} checkpoint")
            revision = commit.oid
            expected_remote = {f"{path_in_repo.rstrip('/')}/{path.name}": path.stat().st_size
                               for path in Path(local_dir).iterdir()}
        except Exception as api_error:
            # This cloud's egress proxy currently rewrites signed S3 PUTs to chunked
            # transfer, which the Hub rejects. Fall back to authenticated Hub Git
            # storage for selected checkpoint files. The API remains the primary
            # path, and the fallback never embeds credentials in a URL or file.
            revision, expected_remote = _upload_checkpoint_via_git(local_dir, repo_id, path_in_repo, token, api_error)
    files = {item.rfilename: item for item in api.repo_info(repo_id, repo_type="model", files_metadata=True).siblings}
    for remote_name, expected_size in expected_remote.items():
        if remote_name not in files or files[remote_name].size != expected_size:
            raise RuntimeError(f"remote verification failed for {remote_name}")
    return revision


def _upload_checkpoint_via_git(local_dir: str | Path, repo_id: str, path_in_repo: str,
                               token: str, api_error: Exception) -> tuple[str, dict[str, int]]:
    with tempfile.TemporaryDirectory(prefix="latticelm-hf-") as temporary:
        root = Path(temporary)
        askpass = root / "askpass.sh"
        askpass.write_text("""#!/bin/sh
case "$1" in
  *Username*) printf '%s\\n' hf_user ;;
  *) printf '%s\\n' "$HF_TOKEN" ;;
esac
""")
        askpass.chmod(0o700)
        env = os.environ.copy()
        env.update({"HF_TOKEN": token, "GIT_ASKPASS": str(askpass), "GIT_TERMINAL_PROMPT": "0"})
        repo = root / "repo"
        commands = [
            ["git", "clone", "--depth", "1", f"https://huggingface.co/{repo_id}", str(repo)],
        ]
        for command in commands:
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        destination = repo / path_in_repo
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        expected_remote: dict[str, int] = {}
        large_files = {}
        chunk_size = 6 * 1024 * 1024
        for local in Path(local_dir).iterdir():
            if local.stat().st_size <= 9 * 1024 * 1024:
                shutil.copy2(local, destination / local.name)
                expected_remote[f"{path_in_repo}/{local.name}"] = local.stat().st_size
                continue
            part_dir = destination / ".parts"
            part_dir.mkdir(exist_ok=True)
            parts = []
            with local.open("rb") as source:
                index = 0
                while chunk := source.read(chunk_size):
                    part_name = f"{local.name}.part{index:04d}.b64"
                    (part_dir / part_name).write_bytes(base64.b64encode(chunk) + b"\n")
                    remote_part = f"{path_in_repo}/.parts/{part_name}"
                    parts.append(f".parts/{part_name}")
                    expected_remote[remote_part] = (part_dir / part_name).stat().st_size
                    index += 1
            large_files[local.name] = {"size": local.stat().st_size, "sha256": sha256(local), "parts": parts}
        if large_files:
            large_manifest = destination / "large_files.json"
            large_manifest.write_text(json.dumps(large_files, indent=2) + "\n")
            expected_remote[f"{path_in_repo}/large_files.json"] = large_manifest.stat().st_size
        attributes = repo / ".gitattributes"
        with attributes.open("a") as handle:
            for local in destination.rglob("*"):
                if not local.is_file():
                    continue
                # Explicit path exceptions avoid the broken LFS transport;
                # large artifacts have already been split below the 10MiB Git limit.
                handle.write(f"\n/{local.relative_to(repo)} -filter -diff -merge -text")
            handle.write("\n")
        subprocess.run(["git", "add", ".gitattributes", path_in_repo], cwd=repo, env=env,
                       check=True, capture_output=True, text=True)
        subprocess.run(["git", "-c", "user.name=LatticeLM automation",
                        "-c", "user.email=latticelm@invalid.local", "commit", "-m",
                        f"Update {path_in_repo} checkpoint (API transport fallback)"],
                       cwd=repo, env=env, check=True, capture_output=True, text=True)
        pushed = subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=repo, env=env,
                                capture_output=True, text=True)
        if pushed.returncode:
            raise RuntimeError(f"Hub Git fallback push failed: {pushed.stderr.strip()}")
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, env=env, check=True,
                                  capture_output=True, text=True).stdout.strip()
        return revision, expected_remote
