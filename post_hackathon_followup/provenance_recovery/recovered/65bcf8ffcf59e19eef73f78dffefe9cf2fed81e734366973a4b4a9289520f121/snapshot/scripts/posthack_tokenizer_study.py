"""Matched 4K/8K 16M ablation on a single retokenized raw subset."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from posthack_archive import upload
from posthack_calibrate import benchmark
from posthack_campaign import RESEARCH, ROOT, STATE, event, sha256
from posthack_phase1 import invoke
from posthack_single_run import TASKS, register_screen, run_screen
from train_final_production import load_arrays

OUT = RESEARCH / "retokenized_subset"
CONFIGS = {4096: RESEARCH / "configs/16m_4k_subset.json",
           8192: RESEARCH / "configs/16m_8k_subset.json"}
TOKENIZERS = {4096: ROOT / "artifacts/tokenizers/final_corpus_4k.json",
              8192: ROOT / "artifacts/tokenizers/final_corpus_8k.json"}
IDS = {4096: "C16-4K-SUBSET", 8192: "C16-8K-SUBSET"}
BUDGET = 150_000_000


def weighted_bpt(report: dict, size: int) -> float:
    mixture = json.loads((OUT / "4k/manifest.json").read_text())["mixture_definition"]
    metrics = report["tokenizations"][str(size)]["source_metrics"]
    return sum(weight * metrics[source]["train"]["bytes_per_token"]
               for source, weight in mixture.items())


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


def evaluate_byte_milestone(checkpoint: Path) -> dict:
    run = checkpoint.parents[1]
    wiki = run / "byte_matched_wikitext103.json"
    capability = run / "byte_matched_capability_screen.json"
    samples = run / "byte_matched_capability_samples.json"
    if not wiki.exists():
        invoke([sys.executable, "scripts/evaluate_wikitext103.py",
                "--checkpoint", str(checkpoint), "--tokenizer", str(TOKENIZERS[8192]),
                "--output", str(wiki), "--threads", "4"], run / "evaluate.log", "byte-matched WikiText")
    if not capability.exists():
        invoke([sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", str(checkpoint),
                "--tokenizer", str(TOKENIZERS[8192]), "--output", str(capability),
                "--tasks", TASKS, "--limit", "1000", "--threads", "4",
                "--predictions-output", str(samples)], run / "evaluate.log", "byte-matched capability")
    upload([wiki, capability, samples], "C16-8K-BYTE-MATCHED-EVAL")
    return json.loads(wiki.read_text())


def main() -> None:
    report_path = OUT / "report.json"
    if not report_path.exists():
        invoke([sys.executable, "scripts/posthack_retokenize_subset.py"],
               OUT / "build.log", "common raw-text retokenization")
        upload([report_path], "TOKENIZER-SUBSET-REPORT")
    report = json.loads(report_path.read_text())
    bpt4, bpt8 = weighted_bpt(report, 4096), weighted_bpt(report, 8192)
    if bpt8 <= bpt4:
        raise RuntimeError("8K tokenizer failed expected compression ordering")
    step_tokens = 32 * 256
    byte_target = math.ceil((BUDGET * bpt4 / bpt8) / step_tokens) * step_tokens
    torch.set_num_interop_threads(1)
    rows = {}
    for size in (4096, 8192):
        manifest = OUT / f"{size//1024}k/manifest.json"
        best = calibrate(size, manifest)
        name = IDS[size]
        extra = {"raw_subset_report_sha256": sha256(report_path),
                 "fairness_axes": ["equal_parameters", "equal_tokenizer_tokens", "approximate_equal_raw_bytes", "wall_clock"]}
        if size == 8192:
            extra["extra_checkpoint_tokens"] = {"byte_matched": byte_target}
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
    matched_wiki = evaluate_byte_milestone(milestone)
    comparison = {"4k_bpt_on_common_raw_subset": bpt4,
                  "8k_bpt_on_common_raw_subset": bpt8,
                  "8k_byte_matched_checkpoint_target_tokens": byte_target,
                  "8k_byte_matched_actual_tokens": int(torch.load(milestone, map_location="cpu", weights_only=False)["logical_tokens"]),
                  "endpoint_wikitext_bpb": {str(size): rows[size]["bits_per_byte"] for size in rows},
                  "8k_byte_matched_wikitext_bpb": matched_wiki["bits_per_byte"],
                  "parameter_count": {str(size): next(json.loads(line)["parameter_count"]
                                      for line in (RESEARCH / "RUN_REGISTRY.jsonl").open()
                                      if json.loads(line)["experiment_id"] == IDS[size])
                                      for size in IDS},
                  "source_reuse_factors_at_150m": {
                      str(size): {
                          source: BUDGET * json.loads((OUT / f"{size//1024}k/manifest.json").read_text())["mixture_definition"][source]
                                  / max(1, report["tokenizations"][str(size)]["source_metrics"][source]["train"]["tokens"])
                          for source in report["tokenizations"][str(size)]["source_metrics"]}
                      for size in IDS}}
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    lines = ["# Tokenizer and context efficiency", "",
             "The 4K and 8K models use the same raw document IDs, source order, train/validation split, optimizer, batch semantics, and approximately 16M trainable parameters. Their untied embedding/head costs are offset by FFN widths 1024 and 752 respectively. The source sample is the pinned phase7f raw cache plus the pinned 40MB FineMath sample; this controlled subset is distinct from full DATA-D-v4, so its 4K arm is the proper direct comparator.", "",
             "| Measure | 4K | 8K |", "|---|---:|---:|",
             f"| Trainable parameters | {comparison['parameter_count']['4096']:,} | {comparison['parameter_count']['8192']:,} |",
             f"| Mixture-weighted raw bytes/token | {bpt4:.4f} | {bpt8:.4f} |",
             f"| WikiText BPB at 150M tokenizer tokens | {rows[4096]['bits_per_byte']:.5f} | {rows[8192]['bits_per_byte']:.5f} |",
             f"| 8K BPB near 4K's 150M-token raw-byte exposure | — | {matched_wiki['bits_per_byte']:.5f} |",
             "", f"The byte-matched 8K checkpoint was captured at {comparison['8k_byte_matched_actual_tokens']:,} tokenizer tokens; the target was {byte_target:,}. Raw-byte matching uses the mixture-weighted subset compression ratio, and is approximate because each source has a distinct ratio. Equal-token endpoints use 150M tokenizer tokens and process different byte volumes.", "",
             "The raw subset is smaller than full DATA-D-v4. Source streams repeat deterministically once their blocks are exhausted; per-source target exposure divided by available train tokens is recorded as `source_reuse_factors_at_150m` in the comparison JSON. Any quality gain therefore applies to this repeated common subset and needs larger-corpus confirmation.", "",
             "The separate prompt-length audit found at most 0.299% of four competition task requests crossing context 256. This did not clear the registered 1–2% context-training gate. The transfer-suite prompt audit and full conclusion are recorded separately.", ""]
    (RESEARCH / "TOKENIZER_CONTEXT.md").write_text("\n".join(lines))
    upload([OUT / "comparison.json", RESEARCH / "TOKENIZER_CONTEXT.md"], "TOKENIZER-STUDY-REPORT")
    event(IDS[8192], "TOKENIZER_STUDY_ANALYZED", comparison_sha256=sha256(OUT / "comparison.json"))


if __name__ == "__main__":
    main()
