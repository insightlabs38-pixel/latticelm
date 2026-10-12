"""Evaluate P1 weight interpolation before considering recovery training."""
from __future__ import annotations

import json
import sys

import torch

from latticelm.config import LatticeConfig
from latticelm.posthack_variants import build_posthack_model
from posthack_archive import upload
from posthack_campaign import RESEARCH, ROOT, atomic_json, sha256, utc
from posthack_phase1 import invoke
from posthack_single_run import TASKS
from posthack_train import evaluate
from train_final_production import load_arrays

PARENT_SELECTION = RESEARCH / "posttraining/parent_selection.json"
P1 = RESEARCH / "runs/P1-16M-ALIGNED-REPLAY-KL/latest.pt"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"


def main() -> None:
    parent_record = json.loads(PARENT_SELECTION.read_text())
    parent_path = ROOT / parent_record["checkpoint_path"]
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    child = torch.load(P1, map_location="cpu", weights_only=False)
    if child["parent_checkpoint_sha256"] != sha256(parent_path):
        raise RuntimeError("P1 parent differs from selected interpolation parent")
    if parent["config"] != child["config"] or parent.get("variant", "baseline") != child.get("variant", "baseline"):
        raise RuntimeError("interpolation requires matching architectures")
    cfg = LatticeConfig(**parent["config"])
    variant = parent.get("variant", "baseline")
    torch.set_num_threads(4)
    _, _, valid = load_arrays(MANIFEST)
    out = RESEARCH / "posttraining/interpolation"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    parent_metrics = json.loads((RESEARCH / "runs" / parent_record["experiment_id"] / "wikitext103.json").read_text())
    child_metrics = json.loads((RESEARCH / "runs/P1-16M-ALIGNED-REPLAY-KL/wikitext103.json").read_text())
    parent_validation = [json.loads(line) for line in (RESEARCH / "runs" / parent_record["experiment_id"] / "evaluations.jsonl").open()][-1]
    child_validation = [json.loads(line) for line in (RESEARCH / "runs/P1-16M-ALIGNED-REPLAY-KL/evaluations.jsonl").open()][-1]
    rows.append({"alpha": 0.0, "bits_per_byte": parent_metrics["bits_per_byte"],
                 "source_losses": parent_validation["source_losses"],
                 "checkpoint_sha256": sha256(parent_path)})
    for alpha in (.10, .25, .50):
        name = f"alpha_{int(alpha * 100):02d}"
        checkpoint_path = out / f"{name}.pt"
        wiki_path = out / f"{name}_wikitext103.json"
        validation_path = out / f"{name}_validation.json"
        if not checkpoint_path.exists():
            state = {}
            for key, base_value in parent["model"].items():
                child_value = child["model"][key]
                state[key] = (base_value + alpha * (child_value - base_value)
                              if torch.is_floating_point(base_value) else child_value.clone())
            torch.save({"schema": "posthack-interpolation-v1", "config": cfg.to_dict(),
                        "variant": variant, "model": state, "alpha": alpha,
                        "parent_checkpoint_sha256": sha256(parent_path),
                        "child_checkpoint_sha256": sha256(P1)}, checkpoint_path)
        if not validation_path.exists():
            model = build_posthack_model(cfg, variant)
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
            source_losses, paired = evaluate(model, valid, cfg.context_length)
            atomic_json(validation_path, {"alpha": alpha, "source_losses": source_losses,
                                          "paired_losses": paired})
        if not wiki_path.exists():
            invoke([sys.executable, "scripts/evaluate_wikitext103.py",
                    "--checkpoint", str(checkpoint_path), "--tokenizer", str(TOKENIZER),
                    "--output", str(wiki_path), "--threads", "4"],
                   out / "evaluate.log", f"interpolation {name} WikiText")
        rows.append({"alpha": alpha,
                     "bits_per_byte": json.loads(wiki_path.read_text())["bits_per_byte"],
                     "source_losses": json.loads(validation_path.read_text())["source_losses"],
                     "checkpoint_sha256": sha256(checkpoint_path),
                     "checkpoint_path": str(checkpoint_path.relative_to(ROOT))})
    rows.append({"alpha": 1.0, "bits_per_byte": child_metrics["bits_per_byte"],
                 "source_losses": child_validation["source_losses"],
                 "checkpoint_sha256": sha256(P1)})
    base_bpb = rows[0]["bits_per_byte"]
    candidates = [row for row in rows[1:] if row["bits_per_byte"] <= base_bpb * 1.02 and
                  all(row["source_losses"][source] <= rows[0]["source_losses"][source] * 1.05
                      for source in rows[0]["source_losses"])]
    selected = max(candidates, key=lambda row: row["alpha"]) if candidates else rows[0]
    if 0 < selected["alpha"] < 1:
        checkpoint_path = ROOT / selected["checkpoint_path"]
        output = out / "selected_capability_screen.json"
        samples = out / "selected_capability_samples.json"
        if not output.exists():
            invoke([sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", str(checkpoint_path),
                    "--tokenizer", str(TOKENIZER), "--output", str(output),
                    "--tasks", TASKS, "--limit", "1000", "--threads", "4",
                    "--predictions-output", str(samples)],
                   out / "evaluate.log", "selected interpolation capability")
    report = {"schema": "posthack-interpolation-v1", "created_utc": utc(),
              "parent_checkpoint_sha256": sha256(parent_path),
              "child_checkpoint_sha256": sha256(P1), "rows": rows,
              "selected": selected,
              "selection_rule": "largest P1 fraction with <=2% WikiText BPB loss and <=5% source-wise held-out regression; broad capability evaluated for non-endpoint selection",
              "recovery_training_launched": False}
    atomic_json(out / "report.json", report)
    archive = [out / "report.json"]
    if 0 < selected["alpha"] < 1:
        archive += [ROOT / selected["checkpoint_path"],
                    out / "selected_capability_screen.json", out / "selected_capability_samples.json"]
    upload(archive, "POSTTRAIN-INTERPOLATION")


if __name__ == "__main__":
    main()
