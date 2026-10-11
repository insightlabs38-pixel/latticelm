"""Common endpoint evaluation, with atomic publication and checkpoint identity."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from followup_common import ROOT, RESEARCH, atomic_json, registered, sha256, utc, verified_checkpoint


def evaluate(name):
    record = registered(name); run = RESEARCH / "runs" / name
    item = verified_checkpoint(run)
    if not item or item["logical_tokens"] != record["token_budget"]: raise RuntimeError("endpoint checkpoint is missing")
    checkpoint = run / item["file"]; tokenizer = ROOT / record["tokenizer_path"]
    outputs = []
    for script, filename, extra in (
        ("evaluate_wikitext103.py", "wikitext103.json", []),
        ("evaluate_gibc.py", "capability_full.json", ["--tasks", record.get("evaluation_tasks", "hellaswag,arc_easy,piqa,winogrande")])):
        output = run / filename
        if output.exists():
            data = json.loads(output.read_text())
            identity = data.get("checkpoint_sha256", data.get("phase6_metadata", {}).get("checkpoint_sha256"))
            if identity != item["sha256"]: raise RuntimeError("stale evaluation " + filename)
        else:
            temporary = run / ("." + filename + ".tmp")
            command = [sys.executable, str(ROOT / "scripts" / script), "--checkpoint", str(checkpoint),
                       "--tokenizer", str(tokenizer), "--output", str(temporary), "--threads", "16"] + extra
            if script == "evaluate_gibc.py": command += ["--predictions-output", str(run / ".capability_samples.json.tmp")]
            with (run / "evaluate.log").open("a") as log: subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            with temporary.open("rb") as handle: os.fsync(handle.fileno())
            os.replace(temporary, output)
            if script == "evaluate_gibc.py": os.replace(run / ".capability_samples.json.tmp", run / "capability_samples.json")
        outputs.append({"file": filename, "sha256": sha256(output), "size_bytes": output.stat().st_size})
    validation = json.loads((run / "latest_validation.json").read_text())
    if validation["checkpoint_sha256"] != item["sha256"]: raise RuntimeError("endpoint source evaluation mismatch")
    receipt = {"at": utc(), "status": "COMPLETED", "checkpoint_sha256": item["sha256"],
               "tokenizer_sha256": record["tokenizer_sha256"], "outputs": outputs,
               "capability_scope": "full official task evaluation", "source_losses": validation["source_losses"],
               "frozen_mixture_weighted_loss": validation["frozen_mixture_weighted_loss"]}
    atomic_json(run / "evaluation_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--experiment-id", required=True)
    print(json.dumps(evaluate(p.parse_args().experiment_id)))
