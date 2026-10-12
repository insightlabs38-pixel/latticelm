"""Matched 4K/8K ablation and evidence-gated 16K screen on one raw subset."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from posthack_archive import upload
from posthack_calibrate import benchmark
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, can_launch, event, sha256, utc
from posthack_phase1 import invoke
from posthack_single_run import TASKS, register_screen, remaining_tokens, run_screen
from train_final_production import load_arrays

OUT = RESEARCH / "retokenized_subset"
CONFIGS = {4096: RESEARCH / "configs/16m_4k_subset.json",
           8192: RESEARCH / "configs/16m_8k_subset.json",
           16384: RESEARCH / "configs/16m_16k_subset.json"}
TOKENIZERS = {4096: ROOT / "artifacts/tokenizers/final_corpus_4k.json",
              8192: ROOT / "artifacts/tokenizers/final_corpus_8k.json",
              16384: RESEARCH / "tokenizers/final_corpus_16k.json"}
IDS = {4096: "C16-4K-SUBSET", 8192: "C16-8K-SUBSET",
       16384: "C16-16K-SUBSET"}
BUDGET = 150_000_000


def weighted_bpt(report: dict, size: int) -> float:
    mixture = json.loads((OUT / "4k/manifest.json").read_text())["mixture_definition"]
    metrics = report["tokenizations"][str(size)]["source_metrics"]
    return sum(weight * metrics[source]["train"]["bytes_per_token"]
               for source, weight in mixture.items())


def weighted_rate(report: dict, size: int, key: str) -> float:
    mixture = json.loads((OUT / "4k/manifest.json").read_text())["mixture_definition"]
    metrics = report["tokenizations"][str(size)]["source_metrics"]
    return sum(weight * metrics[source]["train"][key]
               for source, weight in mixture.items())


def benchmark_prompts(size: int) -> dict:
    if size == 4096:
        path = RESEARCH / "prompt_truncation_audit.json"
    else:
        path = RESEARCH / f"prompt_truncation_audit_{size//1024}k.json"
        detail = RESEARCH / f"prompt_request_lengths_{size//1024}k.jsonl"
        if not path.exists():
            invoke([sys.executable, "scripts/posthack_prompt_audit.py",
                    "--tokenizer", str(TOKENIZERS[size]),
                    "--output", str(path), "--detail-output", str(detail)],
                   OUT / "benchmark_prompt_audit.log", f"{size//1024}K benchmark prompt audit")
            upload([path, detail], f"TOKENIZER-{size//1024}K-PROMPT-AUDIT")
    result = json.loads(path.read_text())
    if result["limited_examples_per_task"] is not None:
        raise RuntimeError("tokenizer benchmark prompt audit must use full task requests")
    return result


def calibrate(size: int, manifest: Path) -> dict:
    state = json.loads(STATE.read_text())
    label = f"{size//1024}k"
    if not all(f"16m/{label}/{threads}" in state["throughput_calibration"] for threads in (12, 16)):
        top, train, _ = load_arrays(manifest)
        for threads in (12, 16):
            state = json.loads(STATE.read_text())
            if f"16m/{label}/{threads}" not in state["throughput_calibration"]:
                result = benchmark(CONFIGS[size], threads, top, train, 5, 12,
                                   manifest_path=manifest,
                                   tokenizer_path=TOKENIZERS[size])
                print(json.dumps(result, sort_keys=True), flush=True)
    state = json.loads(STATE.read_text())
    return max((state["throughput_calibration"][f"16m/{label}/{threads}"] for threads in (12, 16)),
               key=lambda row: row["tokens_per_second"])


def evaluate_byte_milestone(checkpoint: Path, size: int) -> dict:
    run = checkpoint.parents[1]
    wiki = run / "byte_matched_wikitext103.json"
    capability = run / "byte_matched_capability_screen.json"
    samples = run / "byte_matched_capability_samples.json"
    if not wiki.exists():
        invoke([sys.executable, "scripts/evaluate_wikitext103.py",
                "--checkpoint", str(checkpoint), "--tokenizer", str(TOKENIZERS[size]),
                "--output", str(wiki), "--threads", "4"], run / "evaluate.log", "byte-matched WikiText")
    if not capability.exists():
        invoke([sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", str(checkpoint),
                "--tokenizer", str(TOKENIZERS[size]), "--output", str(capability),
                "--tasks", TASKS, "--limit", "1000", "--threads", "4",
                "--predictions-output", str(samples)], run / "evaluate.log", "byte-matched capability")
    upload([wiki, capability, samples], f"C16-{size//1024}K-BYTE-MATCHED-EVAL")
    return json.loads(wiki.read_text())


def sixteen_k_decision(bpt4: float, bpt8: float, bpt16: float,
                       wiki4: dict, wiki8: dict) -> dict:
    """Freeze the evidence and time gate before any 16K model training."""
    path = OUT / "16k_decision.json"
    if path.exists():
        return json.loads(path.read_text())
    compression_gain = bpt16 / bpt8 - 1
    eight_k_bpb_change = wiki8["bits_per_byte"] / wiki4["bits_per_byte"] - 1
    evidence = compression_gain >= .04 or eight_k_bpb_change <= -.01
    state = json.loads(STATE.read_text())
    reference = state["throughput_calibration"]["16m/8k/16"]
    estimated_tps = reference["tokens_per_second"] * .65
    fits, finish = can_launch(BUDGET, estimated_tps, 7200, 1800, 3600)
    result = {"created_utc": utc(), "run_16k": bool(evidence and fits),
              "compression_gain_over_8k": compression_gain,
              "8k_bpb_change_vs_4k": eight_k_bpb_change,
              "estimated_16k_tps": estimated_tps,
              "estimated_finish_utc": utc(finish),
              "criterion": "16K bytes/token improves >=4% over 8K or 8K BPB improves >=1% over 4K; predicted finish must precede archival reserve",
              "reason": ("evidence and time gate passed" if evidence and fits else
                         "compression and 8K result did not justify 16K training" if not evidence else
                         "16K run would cross archival reserve")}
    atomic_json(path, result)
    event(IDS[16384], "OPTIONAL_RUN_DECISION", **result)
    return result


def main() -> None:
    report_path = OUT / "report.json"
    if not report_path.exists():
        invoke([sys.executable, "scripts/posthack_retokenize_subset.py"],
               OUT / "build.log", "common raw-text retokenization")
        upload([report_path], "TOKENIZER-SUBSET-REPORT")
    report = json.loads(report_path.read_text())
    bpt4, bpt8, bpt16 = (weighted_bpt(report, size) for size in (4096, 8192, 16384))
    prompts = {size: benchmark_prompts(size) for size in (4096, 8192, 16384)}
    if bpt8 <= bpt4:
        raise RuntimeError("8K tokenizer failed expected compression ordering")
    step_tokens = 32 * 256
    byte_targets = {size: math.ceil((BUDGET * bpt4 / weighted_bpt(report, size)) / step_tokens) * step_tokens
                    for size in (8192, 16384)}
    torch.set_num_interop_threads(1)
    rows = {}
    for size in (4096, 8192):
        manifest = OUT / f"{size//1024}k/manifest.json"
        best = calibrate(size, manifest)
        name = IDS[size]
        extra = {"raw_subset_report_sha256": sha256(report_path),
                 "fairness_axes": ["equal_parameters", "equal_tokenizer_tokens", "approximate_equal_raw_bytes", "wall_clock"]}
        if size == 8192:
            extra["extra_checkpoint_tokens"] = {"byte_matched": byte_targets[size]}
        register_screen(experiment_id=name, config_path=CONFIGS[size], variant="baseline",
                        budget=BUDGET,
                        parent_control=IDS[4096] if size == 8192 else "A16-CO4-S1337",
                        hypothesis="An 8K vocabulary improves BPB or downstream capability at equal parameters and raw-document order"
                                   if size == 8192 else "Common-subset 4K reference isolates tokenizer changes",
                        throughput_tps=best["tokens_per_second"],
                        estimate_provenance=best["experiment_id"],
                        manifest=manifest, tokenizer=TOKENIZERS[size],
                        extra_fields=extra)
        run_screen(name, CONFIGS[size], best["threads"], best["tokens_per_second"])
        rows[size] = json.loads((RESEARCH / "runs" / name / "wikitext103.json").read_text())
    milestone = RESEARCH / "runs" / IDS[8192] / "milestones/byte_matched.pt"
    matched_wiki = evaluate_byte_milestone(milestone, 8192)
    decision = sixteen_k_decision(bpt4, bpt8, bpt16, rows[4096], rows[8192])
    matched_16k = None
    if decision["run_16k"]:
        size = 16384
        manifest = OUT / "16k/manifest.json"
        best = calibrate(size, manifest)
        fits, finish = (True, None) if (RESEARCH / "runs" / IDS[size] / "hf_archive_receipt.json").exists() else can_launch(
            remaining_tokens(IDS[size], BUDGET), best["tokens_per_second"], 7200, 1800, 3600)
        if not fits:
            event(IDS[size], "RUN_REJECTED", reason="measured 16K throughput crosses reserve",
                  predicted_finish_utc=utc(finish))
            decision["run_16k"] = False
            decision["reason"] = "measured 16K throughput crosses archival reserve"
            atomic_json(OUT / "16k_decision.json", decision)
        else:
            register_screen(experiment_id=IDS[size], config_path=CONFIGS[size],
                            variant="baseline", budget=BUDGET,
                            parent_control=IDS[4096],
                            hypothesis="A parameter-matched 16K vocabulary improves BPB or capability per raw byte",
                            throughput_tps=best["tokens_per_second"],
                            estimate_provenance=best["experiment_id"],
                            manifest=manifest, tokenizer=TOKENIZERS[size],
                            extra_fields={"raw_subset_report_sha256": sha256(report_path),
                                          "fairness_axes": ["equal_parameters", "equal_tokenizer_tokens",
                                                            "approximate_equal_raw_bytes", "wall_clock"],
                                          "extra_checkpoint_tokens": {"byte_matched": byte_targets[size]}})
            run_screen(IDS[size], CONFIGS[size], best["threads"], best["tokens_per_second"])
            rows[size] = json.loads((RESEARCH / "runs" / IDS[size] / "wikitext103.json").read_text())
            matched_16k = evaluate_byte_milestone(
                RESEARCH / "runs" / IDS[size] / "milestones/byte_matched.pt", size)
    comparison = {"4k_bpt_on_common_raw_subset": bpt4,
                  "8k_bpt_on_common_raw_subset": bpt8,
                  "16k_bpt_on_common_raw_subset": bpt16,
                  "per_source_tokenizer_metrics": {str(size): report["tokenizations"][str(size)]["source_metrics"]
                                                   for size in (4096, 8192, 16384)},
                  "weighted_characters_per_token": {str(size): weighted_rate(report, size, "characters_per_token")
                                                     for size in (4096, 8192, 16384)},
                  "weighted_words_per_token": {str(size): weighted_rate(report, size, "words_per_token")
                                                for size in (4096, 8192, 16384)},
                  "benchmark_prompt_lengths": {str(size): prompts[size]["tasks"]
                                               for size in (4096, 8192, 16384)},
                  "16k_training_decision": decision,
                  "8k_byte_matched_checkpoint_target_tokens": byte_targets[8192],
                  "8k_byte_matched_actual_tokens": int(torch.load(milestone, map_location="cpu", weights_only=False)["logical_tokens"]),
                  "endpoint_wikitext_bpb": {str(size): rows[size]["bits_per_byte"] for size in rows},
                  "8k_byte_matched_wikitext_bpb": matched_wiki["bits_per_byte"],
                  "16k_byte_matched_wikitext_bpb": matched_16k["bits_per_byte"] if matched_16k else None,
                  "parameter_count": {str(size): next(json.loads(line)["parameter_count"]
                                      for line in (RESEARCH / "RUN_REGISTRY.jsonl").open()
                                      if json.loads(line)["experiment_id"] == IDS[size])
                                      for size in rows},
                  "source_reuse_factors_at_150m": {
                      str(size): {
                          source: BUDGET * json.loads((OUT / f"{size//1024}k/manifest.json").read_text())["mixture_definition"][source]
                                  / max(1, report["tokenizations"][str(size)]["source_metrics"][source]["train"]["tokens"])
                          for source in report["tokenizations"][str(size)]["source_metrics"]}
                      for size in rows}}
    atomic_json(OUT / "comparison.json", comparison)
    bpb16_display = f"{rows[16384]['bits_per_byte']:.5f}" if 16384 in rows else "gated"
    lines = ["# Tokenizer and context efficiency", "",
             "The models use the same raw document IDs, source order, train/validation split, optimizer, batch semantics, and approximately 16M trainable parameters. Untied embedding/head costs are offset by FFN widths 1024, 752, and 205 for 4K, 8K, and 16K. The source sample is the pinned phase7f raw cache plus the pinned 40MB FineMath sample; this controlled subset is distinct from full DATA-D-v4, so its 4K arm is the proper direct comparator.", "",
             "| Measure | 4K | 8K | 16K |", "|---|---:|---:|---:|",
             f"| Trainable parameters | {comparison['parameter_count']['4096']:,} | {comparison['parameter_count']['8192']:,} | {comparison['parameter_count'].get('16384', 'gated')} |",
             f"| Mixture-weighted raw bytes/token | {bpt4:.4f} | {bpt8:.4f} | {bpt16:.4f} |",
             f"| Mixture-weighted characters/token | {comparison['weighted_characters_per_token']['4096']:.4f} | {comparison['weighted_characters_per_token']['8192']:.4f} | {comparison['weighted_characters_per_token']['16384']:.4f} |",
             f"| Mixture-weighted words/token | {comparison['weighted_words_per_token']['4096']:.4f} | {comparison['weighted_words_per_token']['8192']:.4f} | {comparison['weighted_words_per_token']['16384']:.4f} |",
             f"| WikiText BPB at 150M tokenizer tokens | {rows[4096]['bits_per_byte']:.5f} | {rows[8192]['bits_per_byte']:.5f} | {bpb16_display} |",
             f"| 8K BPB near 4K's 150M-token raw-byte exposure | — | {matched_wiki['bits_per_byte']:.5f} | — |",
             "", f"The byte-matched 8K checkpoint was captured at {comparison['8k_byte_matched_actual_tokens']:,} tokenizer tokens; the target was {byte_targets[8192]:,}. Raw-byte matching uses the mixture-weighted subset compression ratio, and is approximate because each source has a distinct ratio. Equal-token endpoints use 150M tokenizer tokens and process different byte volumes.", "",
             f"The 16K training gate: {decision['reason']}. Its bytes/token gain over 8K was {decision['compression_gain_over_8k']:.2%}; the 8K BPB change versus 4K was {decision['8k_bpb_change_vs_4k']:.2%}.", "",
             "The raw subset is smaller than full DATA-D-v4. Source streams repeat deterministically once their blocks are exhausted; per-source target exposure divided by available train tokens is recorded as `source_reuse_factors_at_150m` in the comparison JSON. Any quality gain therefore applies to this repeated common subset and needs larger-corpus confirmation.", "",
             f"The full eight-task prompt audit found a maximum meaningful truncation fraction of {json.loads((RESEARCH / 'prompt_truncation_audit.json').read_text())['max_task_meaningful_truncation_fraction_at_256']:.2%} at context 256. The registered context decision and any 512-token screen are in CONTEXT_512.md.", ""]
    lines.extend(["## Benchmark request lengths", "",
                  "Each cell gives mean prompt-plus-continuation tokens under the same full zero-shot task requests. The JSON evidence also records per-task p95, maxima, crossings at 128/256/512/1024, and actual left truncation under context 256.", "",
                  "| Task | 4K | 8K | 16K |", "|---|---:|---:|---:|"])
    for task in prompts[4096]["tasks"]:
        values = [prompts[size]["tasks"][task]["mean_total_tokens"]
                  for size in (4096, 8192, 16384)]
        lines.append(f"| {task} | {values[0]:.1f} | {values[1]:.1f} | {values[2]:.1f} |")
    lines.extend(["", "## Embedding parameter economics", "",
                  "The primary comparison keeps untied input and output matrices. As a secondary accounting exercise, tying them would save one vocabulary-by-328 matrix; no tied model is trained here.", "",
                  "| Vocabulary | Untied embedding/head parameters | Tied parameters | Savings if tied |",
                  "|---:|---:|---:|---:|"])
    for size in (4096, 8192, 16384):
        lines.append(f"| {size:,} | {2*size*328:,} | {size*328:,} | {size*328:,} |")
    lines.append("")
    if 16384 in rows:
        lines.extend(["## Conditional 16K screen", "",
                      f"The matched 16K model has {comparison['parameter_count']['16384']:,} parameters, {bpt16:.4f} mixture-weighted bytes/token, and {rows[16384]['bits_per_byte']:.5f} WikiText BPB at 150M tokenizer tokens.",
                      f"Its byte-matched BPB was {matched_16k['bits_per_byte']:.5f} at {byte_targets[16384]:,} target tokens. Tying the 16K embedding/head would save {16384 * 328:,} parameters; weight tying was not changed in the primary ablation.", ""])
    (RESEARCH / "TOKENIZER_CONTEXT.md").write_text("\n".join(lines))
    upload([OUT / "comparison.json", OUT / "16k_decision.json",
            RESEARCH / "TOKENIZER_CONTEXT.md"], "TOKENIZER-STUDY-REPORT")
    event(IDS[8192], "TOKENIZER_STUDY_ANALYZED", comparison_sha256=sha256(OUT / "comparison.json"))


if __name__ == "__main__":
    main()
