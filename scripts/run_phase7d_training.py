"""Run the controlled Phase 7D Co4-S/Co4-L DATA-C training lineages."""
from __future__ import annotations

import argparse, csv, io, json, math, os, random, resource, time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.final_data import PermutedBlocks
from latticelm.model import build_model
from run_phase7c_final import digest, evaluate, roll_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/data/phase7a"
ORIGINAL_MANIFEST = ROOT / "artifacts/gcp_final_10m_manifest.json"
FIELDS = ["checkpoint", "nominal_tokens", "training_tokens", "step", "train_loss",
          "common_validation_loss", "common_validation_perplexity", "babylm_validation_loss",
          "finewebedu_validation_loss", "cumulative_training_seconds", "interval_training_seconds",
          "tokens_per_second", "peak_rss_bytes", "learning_rate", "babylm_tokens",
          "finewebedu_tokens", "babylm_sequences_drawn", "finewebedu_sequences_drawn",
          "checkpoint_sha256", "preemptions", "infrastructure_downtime_seconds"]


def load(name: str) -> torch.Tensor:
    return torch.tensor(np.memmap(DATA / f"{name}.int32", mode="r", dtype=np.int32), dtype=torch.long)


def payload(model, optimizer, scheduler, config, streams, lineage, step, tokens,
            baby_tokens, web_tokens, elapsed, last_eval, best, preemptions):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "config": config.to_dict(), "source": "DATA-C",
            "lineage": lineage, "step": step, "tokens_seen": tokens,
            "baby_tokens": baby_tokens, "finewebedu_tokens": web_tokens,
            "data_stream_state": {k: v.draws for k, v in streams.items()},
            "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
            "cumulative_training_seconds": elapsed, "last_evaluation_seconds": last_eval,
            "best_validation_loss": best, "preemptions": preemptions,
            "phase7c_manifest_sha256": digest(ORIGINAL_MANIFEST)}


def upload_recovery(path: Path, checksum: str, lineage: str, step: int) -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None
    base = f"recovery/{lineage}"
    operations = [CommitOperationAdd(path_in_repo=f"{base}/latest.pt", path_or_fileobj=str(path)),
                  CommitOperationAdd(path_in_repo=f"{base}/latest.sha256",
                                     path_or_fileobj=io.BytesIO((checksum + "\n").encode()))]
    return HfApi(token=token).create_commit(
        repo_id=os.environ.get("LATTICELM_HF_REPO", "insightlabs38-pixel/LatticeLM-research"),
        repo_type="model", operations=operations,
        commit_message=f"Recovery {lineage} step {step}").oid


def masked_loss(model, pairs, baby_needed: int, web_needed: int) -> torch.Tensor:
    x = torch.stack([p[0] for p in pairs]); y = torch.stack([p[1] for p in pairs])
    logits = model(x)[0]
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                             reduction="none").view(8, 128)
    mask = torch.zeros_like(losses)
    for start, needed in ((0, baby_needed), (6, web_needed)):
        for row in range(start, start + (6 if start == 0 else 2)):
            take = min(128, needed); mask[row, :take] = 1; needed -= take
    return (losses * mask).sum() / mask.sum()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("s", "l"), required=True)
    p.add_argument("--fresh", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--target-tokens", type=int, default=25_000_000)
    p.add_argument("--stop-tokens", type=int); p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()
    if args.fresh == args.resume: p.error("choose exactly one of --fresh or --resume")
    config_path = ROOT / ("configs/phase7d_co4_s_25m.json" if args.model == "s" else "configs/phase7d_co4_l.json")
    config = LatticeConfig.from_json(config_path)
    lineage = "final-gcp-s-25m" if args.model == "s" else "co4-l-data-rich-25m"
    recovery = ROOT / "artifacts/checkpoints" / lineage
    curve = ROOT / ("artifacts/co4_s_25m_curve.csv" if args.model == "s" else "artifacts/co4_l_data_rich_curve.csv")
    log = ROOT / "artifacts/logs" / f"{lineage}.jsonl"
    torch.set_num_threads(config.num_threads); torch.set_num_interop_threads(1)
    random.seed(config.seed); torch.manual_seed(config.seed)
    baby, web = load("babylm_train"), load("finewebedu_train")
    baby_val, web_val, common = load("babylm_validation"), load("finewebedu_validation"), load("common_validation")
    streams = {"baby": PermutedBlocks(baby, 128, config.seed + 11),
               "web": PermutedBlocks(web, 128, config.seed + 29)}
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay,
                                  betas=(config.adam_beta1, config.adam_beta2), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    latest = recovery / "latest.pt"
    step = tokens = baby_tokens = web_tokens = 0
    prior_elapsed = last_eval = 0.0; best = float("inf"); preemptions = 0
    if args.model == "s" and args.resume and not latest.exists():
        # Seed the new rolling directory from the immutable, verified 10M recovery state.
        latest = ROOT / "artifacts/checkpoints/final-gcp-10m/latest.pt"
    if args.fresh and latest.exists():
        raise RuntimeError("refusing to overwrite an existing lineage")
    if args.resume:
        sidecar = latest.with_suffix(".sha256")
        if not sidecar.exists() or digest(latest) != sidecar.read_text().strip():
            raise RuntimeError("missing or invalid recovery checksum")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["source"] != "DATA-C": raise RuntimeError("recovery source is not DATA-C")
        valid_lineages = {lineage} | ({"final-gcp-10m"} if args.model == "s" else set())
        if state["lineage"] not in valid_lineages: raise RuntimeError("wrong recovery lineage")
        expected = config.to_dict(); observed = state["config"]
        for key in expected:
            if key not in {"max_steps"} and expected[key] != observed[key]:
                raise RuntimeError(f"frozen config mismatch: {key}")
        model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); random.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        step, tokens = int(state["step"]), int(state["tokens_seen"])
        baby_tokens, web_tokens = int(state["baby_tokens"]), int(state["finewebedu_tokens"])
        prior_elapsed = float(state["cumulative_training_seconds"]); last_eval = float(state["last_evaluation_seconds"])
        best = float(state["best_validation_loss"]); preemptions = int(state.get("preemptions", 0))
        for k, draws in state["data_stream_state"].items(): streams[k].draws = int(draws)
    target = args.stop_tokens or args.target_tokens
    if target > 25_000_000: raise RuntimeError("Phase 7D must stop before 50M")
    if target <= tokens: raise RuntimeError("target must exceed recovered token count")
    milestones = {1_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000,
                  15_000_000, 20_000_000, 25_000_000}
    evaluated = {m for m in milestones if m <= tokens}
    session_elapsed = 0.0; session_start_tokens = tokens; train_loss = float("nan")
    while tokens < target:
        started = time.perf_counter(); remaining = target - tokens
        pairs = [streams["baby"].one() for _ in range(6)] + [streams["web"].one() for _ in range(2)]
        increment = min(1024, remaining)
        baby_inc = increment * 3 // 4; web_inc = increment - baby_inc
        loss = model(torch.stack([z[0] for z in pairs]), torch.stack([z[1] for z in pairs]))[1] if increment == 1024 else masked_loss(model, pairs, baby_inc, web_inc)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if not all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters()):
            raise FloatingPointError(f"non-finite gradient at step {step + 1}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip); optimizer.step(); scheduler.step()
        session_elapsed += time.perf_counter() - started; step += 1; tokens += increment
        baby_tokens += baby_inc; web_tokens += web_inc; train_loss = float(loss.detach())
        pending = milestones - evaluated
        crossed = min(pending, key=lambda m: abs(tokens-m)) if pending else None
        # The terminal milestone is evaluated only at its exact masked endpoint.
        evaluation = tokens == target or (crossed is not None and crossed != target and abs(tokens-crossed) <= 512)
        periodic = step % config.checkpoint_interval == 0
        if not (periodic or evaluation): continue
        elapsed = prior_elapsed + session_elapsed
        state = payload(model, optimizer, scheduler, config, streams, lineage, step, tokens,
                        baby_tokens, web_tokens, elapsed, last_eval, best, preemptions)
        path, checksum = roll_checkpoint(recovery, state, evaluation)
        if not evaluation:
            if not args.no_upload and step % 1954 == 0: upload_recovery(path, checksum, lineage, step)
            continue
        bl = evaluate(model, baby_val, 8, 128); wl = evaluate(model, web_val, 8, 128); cl = evaluate(model, common, 8, 128)
        best = min(best, cl); elapsed = prior_elapsed + session_elapsed
        state = payload(model, optimizer, scheduler, config, streams, lineage, step, tokens,
                        baby_tokens, web_tokens, elapsed, elapsed, best, preemptions)
        path, checksum = roll_checkpoint(recovery, state, True)
        remote = None
        if not args.no_upload and (evaluation or step % 1954 == 0): remote = upload_recovery(path, checksum, lineage, step)
        nominal = min(milestones, key=lambda m: abs(tokens-m))
        evaluated.add(nominal)
        row = {"checkpoint": f"{lineage}-{nominal}", "nominal_tokens": nominal, "training_tokens": tokens,
               "step": step, "train_loss": train_loss, "common_validation_loss": cl,
               "common_validation_perplexity": math.exp(cl), "babylm_validation_loss": bl,
               "finewebedu_validation_loss": wl, "cumulative_training_seconds": elapsed,
               "interval_training_seconds": elapsed-last_eval,
               "tokens_per_second": (tokens-session_start_tokens)/max(session_elapsed, 1e-9),
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               "learning_rate": optimizer.param_groups[0]["lr"], "babylm_tokens": baby_tokens,
               "finewebedu_tokens": web_tokens, "babylm_sequences_drawn": streams["baby"].draws,
               "finewebedu_sequences_drawn": streams["web"].draws, "checkpoint_sha256": checksum,
               "preemptions": preemptions, "infrastructure_downtime_seconds": 0.0}
        exists = curve.exists(); curve.parent.mkdir(parents=True, exist_ok=True)
        with curve.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            if not exists: w.writeheader()
            w.writerow(row)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f: f.write(json.dumps({**row, "remote_recovery_revision": remote}) + "\n")
        print(json.dumps({**row, "remote_recovery_revision": remote}), flush=True); last_eval = elapsed
    if tokens != target: raise RuntimeError(f"ended at {tokens}, expected {target}")


if __name__ == "__main__": main()
