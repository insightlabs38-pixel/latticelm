"""Run the immutable Phase 7B DATA-C lineage from scratch or exact resume."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch

from latticelm.config import LatticeConfig
from latticelm.model import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/data/phase7a"
TOKENS_PER_STEP = 8 * 128
# Nearest whole frozen batch at or above each requested decimal-token point.
EVALUATION_STEPS = (977, 2930, 4883, 7325, 9766, 14649, 19532, 24415,
                    34180, 39063, 48829)
MAJOR_STEPS = {9766: "final-10m", 24415: "final-25m", 48829: "final-50m"}
CURVE_FIELDS = ["checkpoint", "nominal_tokens", "training_tokens", "step", "train_loss",
                "common_validation_loss", "common_validation_perplexity",
                "babylm_validation_loss", "finewebedu_validation_loss", "cumulative_wall_seconds",
                "interval_wall_seconds", "tokens_per_second", "peak_rss_bytes", "learning_rate",
                "babylm_tokens", "finewebedu_tokens", "babylm_sequences", "finewebedu_sequences",
                "babylm_repeating", "finewebedu_repeating", "checkpoint_sha256"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load(name: str) -> torch.Tensor:
    return torch.from_numpy(np.memmap(DATA / f"{name}.int32", mode="r", dtype=np.int32)).long()


class PermutedBlocks:
    """Deterministic without-replacement block stream serialized by draw count."""
    def __init__(self, tokens: torch.Tensor, context: int, seed: int):
        self.tokens, self.context, self.seed, self.draws = tokens, context, seed, 0
        self.blocks = (len(tokens) - 1) // context

    def one(self) -> tuple[torch.Tensor, torch.Tensor]:
        epoch, position = divmod(self.draws, self.blocks)
        rng = random.Random(self.seed + epoch)
        multiplier = rng.randrange(1, self.blocks)
        while math.gcd(multiplier, self.blocks) != 1:
            multiplier = (multiplier + 1) % self.blocks or 1
        offset = rng.randrange(self.blocks)
        block = (multiplier * position + offset) % self.blocks
        self.draws += 1
        start = block * self.context
        return (self.tokens[start:start + self.context],
                self.tokens[start + 1:start + self.context + 1])


def evaluate(model: torch.nn.Module, tokens: torch.Tensor, config: LatticeConfig) -> float:
    generator = torch.Generator().manual_seed(424242)
    losses = []
    model.eval()
    with torch.inference_mode():
        for _ in range(16):
            starts = torch.randint(0, len(tokens) - config.context_length - 1,
                                   (config.batch_size,), generator=generator)
            x = torch.stack([tokens[i:i + config.context_length] for i in starts.tolist()])
            y = torch.stack([tokens[i + 1:i + config.context_length + 1] for i in starts.tolist()])
            losses.append(float(model(x, y)[1]))
    model.train()
    return sum(losses) / len(losses)


def save(path: Path, model, optimizer, scheduler, step: int, streams: dict[str, PermutedBlocks],
         wall: float, best: float, last_eval_wall: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step, "tokens_seen": step * TOKENS_PER_STEP,
                "config": model.config.to_dict(), "source": "DATA-C", "seed": model.config.seed,
                "source_draws": {key: stream.draws for key, stream in streams.items()},
                "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
                "data_stream_state": {key: stream.draws for key, stream in streams.items()},
                "cumulative_wall_seconds": wall, "best_validation_loss": best,
                "last_evaluation_wall_seconds": last_eval_wall}, path)
    return digest(path)


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    parser.add_argument("--stop-step", type=int, default=48829)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = LatticeConfig.from_json(ROOT / "configs/phase7b_final.json")
    if args.smoke:
        args.stop_step = 2
    if args.stop_step > config.max_steps:
        parser.error("stop-step exceeds the frozen 50M endpoint")
    torch.set_num_threads(config.num_threads)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    baby, web = load("babylm_train"), load("finewebedu_train")
    baby_val, web_val, common = load("babylm_validation"), load("finewebedu_validation"), load("common_validation")
    streams = {"baby": PermutedBlocks(baby, 128, config.seed + 11),
               "web": PermutedBlocks(web, 128, config.seed + 29)}
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay,
                                  betas=(config.adam_beta1, config.adam_beta2), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    start_step, previous_wall, best, last_eval_wall = 0, 0.0, float("inf"), 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config.to_dict() or checkpoint["source"] != "DATA-C":
            raise RuntimeError("resume configuration or data regime differs from frozen lineage")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        previous_wall = float(checkpoint["cumulative_wall_seconds"])
        best = float(checkpoint["best_validation_loss"])
        last_eval_wall = float(checkpoint["last_evaluation_wall_seconds"])
        random.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        for key, draws in checkpoint["data_stream_state"].items():
            streams[key].draws = int(draws)
    started = time.perf_counter()
    train_loss = float("nan")
    evaluation_steps = set(EVALUATION_STEPS) | {args.stop_step}
    for step in range(start_step + 1, args.stop_step + 1):
        kinds = ("baby",) * 6 + ("web",) * 2
        pairs = [streams[kind].one() for kind in kinds]
        x, y = torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        train_loss = float(loss.detach())
        if step not in evaluation_steps:
            continue
        now_wall = previous_wall + time.perf_counter() - started
        baby_loss = evaluate(model, baby_val, config)
        web_loss = evaluate(model, web_val, config)
        common_loss = evaluate(model, common, config)
        improved = common_loss < best
        best = min(best, common_loss)
        name = MAJOR_STEPS.get(step, f"recovery-{step * TOKENS_PER_STEP}")
        checkpoint = ROOT / "artifacts/checkpoints" / f"phase7b_{name}_step{step}.pt"
        checkpoint_hash = save(checkpoint, model, optimizer, scheduler, step, streams, now_wall,
                               best, now_wall)
        nominal = next((value for value, label in ((10_000_000, "final-10m"),
                       (25_000_000, "final-25m"), (50_000_000, "final-50m"))
                       if label == name), step * TOKENS_PER_STEP)
        row = {"checkpoint": name, "nominal_tokens": nominal, "training_tokens": step * TOKENS_PER_STEP,
               "step": step, "train_loss": train_loss, "common_validation_loss": common_loss,
               "common_validation_perplexity": math.exp(common_loss),
               "babylm_validation_loss": baby_loss, "finewebedu_validation_loss": web_loss,
               "cumulative_wall_seconds": now_wall, "interval_wall_seconds": now_wall - last_eval_wall,
               "tokens_per_second": (step - start_step) * TOKENS_PER_STEP / max(1e-9, now_wall - previous_wall),
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "babylm_tokens": streams["baby"].draws * 128,
               "finewebedu_tokens": streams["web"].draws * 128,
               "babylm_sequences": streams["baby"].draws, "finewebedu_sequences": streams["web"].draws,
               "babylm_repeating": streams["baby"].draws > streams["baby"].blocks,
               "finewebedu_repeating": streams["web"].draws > streams["web"].blocks,
               "checkpoint_sha256": checkpoint_hash}
        append_csv(ROOT / "artifacts/final_training_curve.csv", row)
        with (ROOT / "artifacts/logs/phase7b_final.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        last_eval_wall = now_wall
        print(json.dumps(row), flush=True)
        if improved:
            (ROOT / "artifacts/checkpoints/phase7b_best.txt").write_text(str(checkpoint) + "\n")


if __name__ == "__main__":
    main()
