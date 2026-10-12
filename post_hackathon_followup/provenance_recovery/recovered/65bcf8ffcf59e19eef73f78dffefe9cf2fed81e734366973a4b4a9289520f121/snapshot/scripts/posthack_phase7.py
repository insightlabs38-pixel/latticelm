"""Time-gated nCo4 and Differential-Co4 8M pilot screens."""
from __future__ import annotations

import json
import sys
import time

import torch

from posthack_archive import upload
from posthack_calibrate import benchmark
from posthack_campaign import RESEARCH, ROOT, STATE, event, sha256, utc
from posthack_phase1 import invoke
from posthack_phase5 import endpoint
from posthack_single_run import TASKS, register_screen, run_screen
from train_final_production import load_arrays

ADAMW = RESEARCH / "configs/8m_adamw.json"
CO4 = RESEARCH / "configs/8m_co4.json"
MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"


def hours_left() -> float:
    return (json.loads(STATE.read_text())["archive_start_epoch"] - time.time()) / 3600


def calibrated(variant: str) -> dict:
    state = json.loads(STATE.read_text())
    arch = "adamw" if variant == "baseline" else variant
    if not all(f"8m/{arch}/{threads}" in state["throughput_calibration"] for threads in (12, 16)):
        if variant == "differential_co4":
            invoke([sys.executable, "scripts/posthack_calibrate.py", "--regimes", "8m",
                    "--architectures", "co4", "--threads", "12", "16",
                    "--variant", variant], RESEARCH / "diffco4_calibration.log",
                   "Differential-Co4 calibration")
        else:
            top, train, _ = load_arrays(MANIFEST)
            for threads in (12, 16):
                state = json.loads(STATE.read_text())
                if f"8m/{arch}/{threads}" not in state["throughput_calibration"]:
                    result = benchmark(ADAMW, threads, top, train, 5, 12, variant)
                    print(json.dumps(result, sort_keys=True), flush=True)
    state = json.loads(STATE.read_text())
    return max((state["throughput_calibration"][f"8m/{arch}/{threads}"] for threads in (12, 16)),
               key=lambda row: row["tokens_per_second"])


def stretch_run(name: str, config, variant: str, parent: str,
                calibrated_row: dict, hypothesis: str) -> None:
    register_screen(experiment_id=name, config_path=config, variant=variant,
                    budget=50_000_000, parent_control=parent, hypothesis=hypothesis,
                    throughput_tps=calibrated_row["tokens_per_second"] * .9,
                    estimate_provenance=calibrated_row["experiment_id"],
                    promotion_criterion="Repeatable or large quality gain at matched wall/FLOP with stable optimization",
                    abort_criterion="Numerical instability, >4h engineering gate, or archival reserve",
                    extra_fields={"engineering_gate_hours": 4,
                                  "fairness_axes": ["same_tokenizer", "same_data_order", "same_seed",
                                                    "same_optimizer", "equal_tokens", "wall_clock"]})
    run_screen(name, config, calibrated_row["threads"], calibrated_row["tokens_per_second"] * .9)


def nco4() -> None:
    started = time.time()
    control = calibrated("baseline")
    candidate = calibrated("nco4")
    if time.time() - started > 4 * 3600:
        raise RuntimeError("nCo4 engineering/calibration gate exceeded four hours")
    stretch_run("S1-8M-ADAMW-CONTROL", ADAMW, "baseline", "A8-CO4-S1337", control,
                "Matched AdamW control isolates hyperspherical geometry from optimizer choice")
    stretch_run("S1-8M-NCO4", ADAMW, "nco4", "S1-8M-ADAMW-CONTROL", candidate,
                "Unit-direction weights and hidden states with learned residual steps improve Co4 learning speed")
    base = endpoint("S1-8M-ADAMW-CONTROL")
    new = endpoint("S1-8M-NCO4")
    lines = ["# Hyperspherical nCo4 pilot", "",
             "The 8M pilot implements unit-direction embedding/hidden/head and projection weights, post-MOD unit Q/K directions with learned logit scale, learned attention/FFN spherical interpolation steps, and learned FFN/logit scaling. Normalization is functional in the forward pass, so no in-place optimizer projection or optimizer-state replacement occurs. Both arms use the same AdamW 3e-4 recipe, 4K tokenizer, data order, seed, batch and 50M tokens. This is a bounded nGPT-style mechanism screen, not a claim to reproduce every reference implementation detail.", "",
             "| Arm | Held-out loss | WikiText BPB | Four-task raw accuracy | 12/16-thread calibrated TPS |",
             "|---|---:|---:|---:|---:|",
             f"| AdamW Co4 | {base[0]:.5f} | {base[1]:.5f} | {base[2]:.4f} | {control['tokens_per_second']:.0f} |",
             f"| nCo4 | {new[0]:.5f} | {new[1]:.5f} | {new[2]:.4f} | {candidate['tokens_per_second']:.0f} |",
             "", "A positive signal requires quality at matched wall/FLOP, not simply a lower loss under a different optimizer. Tests cover unit hidden norms, finite gradients, checkpoint loading, and the common evaluator.", "",
             "Primary reference: [nGPT](https://arxiv.org/abs/2410.01131).", ""]
    (RESEARCH / "STRETCH_NCO4.md").write_text("\n".join(lines))
    upload([RESEARCH / "STRETCH_NCO4.md"], "NCO4-REPORT")


def baseline_50m() -> tuple[float, float]:
    run = RESEARCH / "runs/A8-CO4-S1337"
    checkpoint = run / "milestones/50m.pt"
    if not checkpoint.exists():
        raise RuntimeError("matched 50M Co4 milestone is missing")
    result = next((json.loads(line) for line in (run / "evaluations.jsonl").open()
                   if json.loads(line)["logical_tokens"] >= 50_000_000), None)
    if result is None:
        raise RuntimeError("matched 50M validation record is missing")
    wiki = run / "milestones/50m_wikitext103.json"
    if not wiki.exists():
        invoke([sys.executable, "scripts/evaluate_wikitext103.py",
                "--checkpoint", str(checkpoint), "--tokenizer", str(TOKENIZER),
                "--output", str(wiki), "--threads", "4"], run / "evaluate.log", "50M Co4 WikiText")
        upload([wiki], "A8-CO4-50M-WIKITEXT")
    mixture = json.loads(MANIFEST.read_text())["mixture_definition"]
    heldout = sum(mixture[source] * result["source_losses"][source] for source in mixture)
    return heldout, json.loads(wiki.read_text())["bits_per_byte"]


def differential() -> None:
    started = time.time()
    candidate = calibrated("differential_co4")
    if time.time() - started > 4 * 3600:
        raise RuntimeError("Differential-Co4 engineering/calibration gate exceeded four hours")
    stretch_run("S2-8M-DIFFCO4", CO4, "differential_co4", "A8-CO4-S1337",
                candidate, "Two post-MOD causal attention maps with learned differential lambda reduce common-mode noise")
    base_loss, base_bpb = baseline_50m()
    new = endpoint("S2-8M-DIFFCO4")
    lines = ["# Differential-Co4 pilot", "",
             "The same 8M Co4 backbone projects Q/K/V once, applies MOD once, QK-Norm and RoPE once, then splits each head into two Q/K subspaces. Two causal softmax maps read the same value stream. Their difference uses the published exponential lambda reparameterization, with lambda initialized at 0.8 and fixed scale correction. Same total Q/K/V widths and the modern Muon/AdamW optimizer are used. The matched Co4 50M milestone has the same seed, data order and token budget.", "",
             "| Arm at about 50M tokens | Held-out loss | WikiText BPB | Calibration TPS |",
             "|---|---:|---:|---:|",
             f"| Co4 baseline milestone | {base_loss:.5f} | {base_bpb:.5f} | see baseline calibration |",
             f"| Differential-Co4 | {new[0]:.5f} | {new[1]:.5f} | {candidate['tokens_per_second']:.0f} |",
             "", "The two attention-map calls increase attention work; a lower per-token loss would not by itself establish FLOP or wall efficiency. Tests cover the disabled exact-control path, causal future-token isolation, finite gradients and checkpoint loading.", "",
             "Primary reference: [Differential Transformer](https://arxiv.org/abs/2410.05258).", ""]
    (RESEARCH / "STRETCH_DIFFCO4.md").write_text("\n".join(lines))
    upload([RESEARCH / "STRETCH_DIFFCO4.md"], "DIFFCO4-REPORT")


def main() -> None:
    torch.set_num_interop_threads(1)
    if not (RESEARCH / "PROMOTION_32M.md").exists():
        raise RuntimeError("32M promotion prerequisite missing")
    if hours_left() < 36:
        reason = f"only {hours_left():.1f} hours before archival reserve"
        (RESEARCH / "STRETCH_NCO4.md").write_text(f"# Hyperspherical nCo4 pilot\n\nNot launched: {reason}.\n")
        (RESEARCH / "STRETCH_DIFFCO4.md").write_text(f"# Differential-Co4 pilot\n\nNot launched: {reason}.\n")
        event("S1-S2", "STRETCH_GATED_OUT", reason=reason)
        return
    if not (RESEARCH / "STRETCH_NCO4.md").exists():
        try:
            nco4()
        except (FloatingPointError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and not any(
                    phrase in str(exc).lower() for phrase in
                    ("gate exceeded", "nonfinite", "does not fit", "after archival reserve")):
                raise
            reason = f"four-hour engineering/numerical gate: {type(exc).__name__}: {exc}"
            (RESEARCH / "STRETCH_NCO4.md").write_text(f"# Hyperspherical nCo4 pilot\n\nStopped: {reason}.\n")
            event("S1-8M-NCO4", "STRETCH_GATED_OUT", reason=reason)
    if hours_left() < 20:
        reason = f"only {hours_left():.1f} hours before archival reserve"
        (RESEARCH / "STRETCH_DIFFCO4.md").write_text(f"# Differential-Co4 pilot\n\nNot launched: {reason}.\n")
        event("S2-8M-DIFFCO4", "STRETCH_GATED_OUT", reason=reason)
        return
    if not (RESEARCH / "STRETCH_DIFFCO4.md").exists():
        try:
            differential()
        except (FloatingPointError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and not any(
                    phrase in str(exc).lower() for phrase in
                    ("gate exceeded", "nonfinite", "does not fit", "after archival reserve")):
                raise
            reason = f"four-hour engineering/numerical gate: {type(exc).__name__}: {exc}"
            (RESEARCH / "STRETCH_DIFFCO4.md").write_text(f"# Differential-Co4 pilot\n\nStopped: {reason}.\n")
            event("S2-8M-DIFFCO4", "STRETCH_GATED_OUT", reason=reason)


if __name__ == "__main__":
    main()
