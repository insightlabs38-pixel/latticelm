"""Validate the historical 300M control using receipts and a short fresh replay."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.data_d import SourceStream
from latticelm.final_recipe import make_optimizer, wsd_lr_scale
from latticelm.posthack_variants import build_posthack_model
from posthack_campaign import atomic_json, sha256, utc
from posthack_train import next_batch_hash
from train_final_production import WeightedMixture, load_arrays

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "post_hackathon_followup"


def audit() -> dict:
    old = ROOT / "post_hackathon_research"
    name = "R32-CO4-CONTROL"
    run = old / "runs" / name
    record = next(r for r in map(json.loads, (old / "RUN_REGISTRY.jsonl").read_text().splitlines())
                  if r["experiment_id"] == name)
    cfg_path = old / "configs/32m_co4.json"
    cfg = LatticeConfig.from_json(cfg_path)
    ck_path = run / "latest.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    receipt = json.loads((run / "result.json").read_text())
    events = [r for r in map(json.loads, (old / "controller_events.jsonl").read_text().splitlines())
              if r.get("experiment_id") == name]
    counts = Counter(e["event"] for e in events)
    metrics = list(map(json.loads, (run / "metrics.jsonl").read_text().splitlines()))
    target = 300_000_000
    step_tokens = cfg.batch_size * cfg.context_length
    steps = math.ceil(target / step_tokens)
    def check(condition, message):
        if not condition:
            raise RuntimeError(message)
    digest = sha256(ck_path)
    check(digest == ck_path.with_suffix(".sha256").read_text().strip() == receipt["checkpoint_sha256"], "checkpoint hash")
    check(sha256(cfg_path) == record["config_sha256"], "config file hash")
    check(json.loads(json.dumps(ck["config"])) == record["config"] == json.loads(cfg_path.read_text()), "config identities")
    check(ck["target_tokens"] == record["token_budget"] == target, "target")
    check(ck["step"] == receipt["steps"] == steps and ck["logical_tokens"] == target, "final positions")
    check(counts["TRAIN_START"] == counts["TRAIN_END"] == 1 and counts["RESUMED"] == 0, "restart events")
    check([r["step"] for r in metrics] == list(range(1, steps + 1)), "noncontiguous step history")
    maximum_lr_error = 0.0
    labels_count = Counter()
    for row in metrics:
        s = row["step"]
        expected_lr = cfg.learning_rate * wsd_lr_scale(s, steps, cfg.warmup_fraction, cfg.stable_fraction, cfg.decay_fraction)
        maximum_lr_error = max(maximum_lr_error, abs(expected_lr - row["lr"]))
        check(row["logical_tokens"] == min(s * step_tokens, target) and row["processed_tokens"] == s * step_tokens, "token history")
        labels_count.update(row["source_labels"])
    check(maximum_lr_error <= 1e-15, "LR trajectory differs from a single 300M WSD schedule")
    trainer = ROOT / "scripts/posthack_train.py"
    trainer_hash = sha256(trainer)
    start = next(e for e in events if e["event"] == "TRAIN_START")
    check(trainer_hash == start["trainer_sha256"] == ck["trainer_sha256"], "trainer identity")
    source_hashes = {}
    with zipfile.ZipFile(old / "source-snapshot-final.zip") as snapshot:
        for rel in ("scripts/posthack_train.py", "scripts/train_final_production.py", "src/latticelm/model.py",
                    "src/latticelm/final_recipe.py", "src/latticelm/data_d.py", "src/latticelm/posthack_variants.py"):
            source = (snapshot.read(rel) if rel in snapshot.namelist() else
                      (ROOT.parent / "latticelm-frozen" / rel).read_bytes())
            expected = hashlib.sha256(source).hexdigest()
            check(sha256(ROOT / rel) == expected, "source snapshot differs: " + rel)
            source_hashes[rel] = expected
    manifest = ROOT / record["manifest_path"]
    tokenizer = ROOT / record["tokenizer_path"]
    check(sha256(manifest) == ck["manifest_sha256"] == record["sampling_manifest_sha256"], "manifest")
    check(sha256(tokenizer) == ck["tokenizer_sha256"] == record["tokenizer_sha256"], "tokenizer")
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    top, train, _ = load_arrays(manifest)  # Int32Shard verifies each immutable token file.
    def new_mixture():
        streams = {source: SourceStream(train[source], cfg.context_length, cfg.seed + 17 * index)
                   for index, source in enumerate(train)}
        return WeightedMixture(streams, top["mixture_definition"], cfg.batch_size)
    mixture = new_mixture()
    check(mixture.state_dict()["counter"] == 0 and all(v == 0 for v in mixture.state_dict()["draws"].values()), "initial sampler")
    initial_batch_hash = next_batch_hash(mixture)
    cycle = mixture.cycle
    for row in metrics:
        offset = (row["step"] - 1) * cfg.batch_size
        check(tuple(row["source_labels"]) == tuple(cycle[(offset+i) % len(cycle)] for i in range(cfg.batch_size)), "source order")
    check(ck["mixture"] == {"counter": steps * cfg.batch_size, "draws": dict(labels_count)}, "checkpoint sampler cursor")
    check(tuple(ck["sampling_cycle"]) == cycle and not ck["sampling_annealed"], "sampling cycle")
    end_mixture = new_mixture(); end_mixture.load_state_dict(ck["mixture"])
    check(next_batch_hash(end_mixture) == ck["next_batch_sha256"], "saved next batch replay")
    model = build_posthack_model(cfg, "baseline")
    check(sum(p.numel() for p in model.parameters()) == record["parameter_count"], "parameters")
    optimizer, partition = make_optimizer(model, cfg.optimizer, cfg.learning_rate, cfg.weight_decay,
                                          (cfg.adam_beta1, cfg.adam_beta2))
    check(partition == record["optimizer_partition"] == ck["optimizer_partition"], "optimizer partition")
    check(all(not o["state"] for o in optimizer.state_dict()), "fresh optimizer states")
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    replay = []
    for step in range(1, 4):
        scale = wsd_lr_scale(step, steps, cfg.warmup_fraction, cfg.stable_fraction, cfg.decay_fraction)
        for group, lr in zip(optimizer.param_groups, base_lrs): group["lr"] = lr * scale
        x, y, labels = mixture.batch()
        optimizer.zero_grad()
        logits, _ = model(torch.from_numpy(x.astype(np.int64)))
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), torch.from_numpy(y.astype(np.int64)).reshape(-1))
        error = abs(float(loss.detach()) - metrics[step-1]["train_loss"])
        check(error <= 3e-5, f"fresh initialization replay mismatch at step {step}: {error}")
        check(list(labels) == metrics[step-1]["source_labels"], "replay source labels")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optimizer.step()
        replay.append({"step": step, "loss": float(loss.detach()), "recorded_loss": metrics[step-1]["train_loss"], "absolute_error": error})
    result = {"schema": "r32-control-provenance-v1", "at": utc(), "passed": True, "control": name,
              "checkpoint_sha256": digest, "config_sha256": sha256(cfg_path), "source_hashes": source_hashes,
              "event_counts": dict(counts), "total_steps": steps, "maximum_lr_error": maximum_lr_error,
              "logical_tokens": target, "processed_tokens": ck["processed_tokens"], "seed": cfg.seed,
              "initial_batch_sha256": initial_batch_hash, "saved_next_batch_sha256": ck["next_batch_sha256"],
              "source_order": list(train), "source_seed_rule": "seed + 17 * index in manifest-loaded source order",
              "mixture_cycle": list(cycle), "fresh_optimizer_empty": True, "fresh_replay": replay,
              "evidence_limit": "No initial checkpoint was archived; three-step fresh replay, matching trainer source, contiguous receipts and exact whole-run LR/cursor checks establish provenance within FP32 tolerance."}
    atomic_json(OUT / "R32_CO4_PROVENANCE_AUDIT.json", result)
    (OUT / "R32_CO4_PROVENANCE_AUDIT.md").write_text(
        "# Historical 300M Co4 provenance\n\nPASS: one fresh initialization and one 300M WSD trajectory.\n\n"
        f"Verified {steps:,} contiguous steps, every recorded LR (max error {maximum_lr_error}), "
        "one TRAIN_START / TRAIN_END, no RESUMED event, source/config/checkpoint hashes, "
        "all data-shard hashes, exact final sampler cursor and next-batch replay. A fresh model plus empty "
        "Muon/AdamW optimizers reproduces the first three recorded losses within 3e-5.\n\n"
        "No initial checkpoint was archived. This conclusion rests on the converging evidence above, "
        "rather than an unrecorded assertion about initialization. See the companion JSON for values and hashes.\n")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
