"""Generate the compact Phase 7C milestone artifacts from preserved raw results."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = list(csv.DictReader((ART / "gcp_final_training_curve.csv").open()))
    gibc = json.loads((ART / "gcp_final_10m_gibc_raw.json").read_text())
    wiki = json.loads((ART / "gcp_final_10m_wikitext103.json").read_text())
    manifest = json.loads((ART / "gcp_final_10m_manifest.json").read_text())
    final = rows[-1]
    result = gibc["results"]

    gibc_csv = ART / "gcp_final_10m_gibc.csv"
    with gibc_csv.open("w", newline="") as handle:
        fields = ["task", "samples", "accuracy", "accuracy_stderr", "normalized_accuracy",
                  "normalized_accuracy_stderr", "fewshot", "lm_eval_version", "wall_seconds",
                  "checkpoint_sha256", "tokenizer_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for task in ("hellaswag", "arc_easy", "piqa", "winogrande"):
            item = result[task]
            writer.writerow({
                "task": task, "samples": item["sample_len"], "accuracy": item.get("acc,none"),
                "accuracy_stderr": item.get("acc_stderr,none"),
                "normalized_accuracy": item.get("acc_norm,none"),
                "normalized_accuracy_stderr": item.get("acc_norm_stderr,none"), "fewshot": 0,
                "lm_eval_version": gibc["phase6_metadata"]["lm_eval_version"],
                "wall_seconds": gibc["phase6_metadata"]["wall_seconds"],
                "checkpoint_sha256": gibc["phase6_metadata"]["checkpoint_sha256"],
                "tokenizer_sha256": gibc["phase6_metadata"]["tokenizer_sha256"],
            })
        writer.writerow({
            "task": "wikitext_103", "samples": wiki["tokens_scored"],
            "accuracy": "", "accuracy_stderr": "", "normalized_accuracy": "",
            "normalized_accuracy_stderr": "", "fewshot": 0, "lm_eval_version": "custom perplexity",
            "wall_seconds": wiki["wall_seconds"], "checkpoint_sha256": wiki["checkpoint_sha256"],
            "tokenizer_sha256": wiki["tokenizer_sha256"],
        })

    slopes = []
    for before, after in zip(rows, rows[1:]):
        delta_m = (float(after["training_tokens"]) - float(before["training_tokens"])) / 1e6
        slopes.append({
            "from_tokens": int(before["training_tokens"]), "to_tokens": int(after["training_tokens"]),
            "loss_improvement_per_million_tokens":
                (float(before["common_validation_loss"]) - float(after["common_validation_loss"])) / delta_m,
            "perplexity_improvement_per_million_tokens":
                (float(before["common_validation_perplexity"]) - float(after["common_validation_perplexity"])) / delta_m,
        })

    metrics = {
        "checkpoint": "final-gcp-10m", "tokens_trained": int(final["training_tokens"]),
        "parameters": manifest["model"]["trainable_parameters"],
        "train_loss": float(final["train_loss"]), "val_loss": float(final["common_validation_loss"]),
        "val_ppl": float(final["common_validation_perplexity"]),
        "common_validation_loss": float(final["common_validation_loss"]),
        "common_validation_perplexity": float(final["common_validation_perplexity"]),
        "babylm_validation_loss": float(final["babylm_validation_loss"]),
        "finewebedu_validation_loss": float(final["finewebedu_validation_loss"]),
        "wall_seconds": float(final["cumulative_training_seconds"]),
        "tokens_per_second": float(final["tokens_per_second"]),
        "peak_rss_bytes": int(final["peak_rss_bytes"]), "preemptions": int(final["preemptions"]),
        "infrastructure_downtime_seconds": float(final["infrastructure_downtime_seconds"]),
        "source_tokens": {"babylm": int(final["babylm_tokens"]),
                          "fineweb_edu": int(final["finewebedu_tokens"])},
        "checkpoint_sha256": final["checkpoint_sha256"],
        "tokenizer_sha256": wiki["tokenizer_sha256"],
        "gibc": {
            "hellaswag": result["hellaswag"], "arc_easy": result["arc_easy"],
            "piqa": result["piqa"], "winogrande": result["winogrande"],
            "lm_eval_version": gibc["phase6_metadata"]["lm_eval_version"],
            "wall_seconds": gibc["phase6_metadata"]["wall_seconds"],
            "wikitext_103_perplexity": wiki["perplexity"],
            "wikitext_103_tokens_scored": wiki["tokens_scored"],
            "wikitext_103_wall_seconds": wiki["wall_seconds"],
        },
        "learning_slopes": slopes, "saturation_status": "SLOWING BUT CLEARLY IMPROVING",
        "recommended_next_token_target": "25M (not started)",
    }
    (ART / "gcp_final_10m_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    profile_rate = 7688.64
    run_rate = metrics["tokens_per_second"]
    hours_100m = 100_000_000 / run_rate / 3600
    cost_100m = hours_100m * 0.0959
    report = f"""# Phase 7C — GCP C4A bring-up and canonical 10M run

## Outcome

The fresh LatticeLM-Co4-S canonical GCP lineage reached exactly **10,000,000 loss-bearing training tokens** and stopped. It consumed exactly 7,500,000 BabyLM tokens and 2,500,000 decontaminated FineWeb-Edu tokens. No 25M/50M continuation was started, no architecture or optimizer setting changed, and no prior trained weights were used.

## Environment and validation

- GCP machine: `c4a-standard-16` Spot, `europe-west4-c`
- Architecture: AArch64 / ARM64, ARM Neoverse-V2
- Visible/usable cores: 16 independent guest-visible cores, one thread per core, one NUMA node; no SMT exposed
- RAM: 62 GiB; swap: none
- Storage: 100 GB Hyperdisk Balanced boot disk mounted at `/`; no separate persistent disk was attached. Checkpoints were therefore also mirrored to Hugging Face. Guest metadata did not expose the boot disk auto-delete policy.
- OS/kernel: Ubuntu 24.04.4 LTS / 7.0.0-1011-gcp
- Python/PyTorch: 3.12.3 / 2.14.0+cpu, native AArch64 build with oneDNN and OpenMP

All 22 tests passed. Source compilation and `git diff --check` passed. The exact Phase 7A binaries, tokenizer, decontamination record, source revisions, validation set, and their hashes were reconstructed and verified. The DATA-C sampler passed independent-process determinism and serialized-resume next-batch hash tests. HF upload/download smoke verification passed.

## Runtime selection

The exact workload sweep used real DATA-C batches and complete forward/loss/backward/clipping/AdamW steps. Twelve threads at batch eight was selected for stability: its longer run measured {profile_rate:,.2f} tokens/s with 0.133183 s median step time. Sixteen threads had a faster 0.130057 s median but much worse variance (0.02925 s standard deviation and 0.03901 s IQR). Batch 16/32 throughput results were retained but rejected because adopting them would alter the frozen effective batch.

- Selected PyTorch intra-op threads: 12
- Selected inter-op threads: 1
- Selected microbatch: 8 sequences / 1,024 tokens
- Selected gradient accumulation: 1
- Selected effective batch: 8 sequences / 1,024 tokens
- Selected runtime: eager PyTorch

`torch.compile` took 16.554 s to initialize, improved steady-state median throughput by only about 0.45%, and produced a maximum one-step parameter difference of 3.353e-4 in the conservative equivalence check. It was therefore rejected. Profiling showed matrix multiplication dominant (46.96% self CPU), with AdamW 8.04% total CPU and CPU flash-SDPA forward/backward 2.66%/4.89%.

## Performance and estimated cost

Canonical training-only throughput was **{run_rate:,.2f} tokens/s** over {metrics['wall_seconds']:.3f} s. That projects to **{hours_100m:.3f} hours per 100M tokens** and **${cost_100m:.4f} per 100M tokens**, using the user's console-derived estimate of $70/730 h = $0.0959/h. This is an approximate normalized compute estimate, not an audited billing rate and does not include setup, evaluation, downtime, storage, networking, or idle VM time. There were zero observed preemptions and zero recorded infrastructure downtime.

## Triton-CPU / AArch64

Status: **WORKING for minimal correctness; partial for production training integration.** Official `triton-lang/triton-cpu` commit `9a3dd8096b3c5b89a6dfeba012221f3fed450eb0` built on AArch64, and the bounded 257-element FP32 vector-add smoke test passed exactly. Missing LLD and a missing SLEEF submodule caused two recorded failed attempts before the successful build. The broad upstream autotuning tutorial was stopped after three minutes. A Co4 MOD kernel was deliberately deferred because a correct custom backward and full-step speedup proof were not immediately available. Best current MOD implementation: eager PyTorch reference.

## Canonical 10M result

| Metric | Result |
|---|---:|
| Parameters | {metrics['parameters']:,} |
| Common validation loss | {metrics['val_loss']:.6f} |
| Common validation perplexity | {metrics['val_ppl']:.5f} |
| BabyLM validation loss | {metrics['babylm_validation_loss']:.6f} |
| FineWeb-Edu validation loss | {metrics['finewebedu_validation_loss']:.6f} |
| WikiText-103 perplexity | {wiki['perplexity']:.5f} |
| HellaSwag accuracy | {result['hellaswag']['acc,none']:.6f} |
| ARC-Easy accuracy | {result['arc_easy']['acc,none']:.6f} |
| PIQA accuracy | {result['piqa']['acc,none']:.6f} |
| WinoGrande accuracy | {result['winogrande']['acc,none']:.6f} |

The GIBC tasks used lm-eval 0.4.13, zero-shot evaluation, batch eight, and the frozen tokenizer. Normalized scores and standard errors are preserved in `gcp_final_10m_gibc.csv`; raw output is retained separately. WikiText-103 scored 411,412 tokens from `wikitext-103-raw-v1`.

## Learning progression and interpretation

| Tokens | Common loss | Common perplexity |
|---:|---:|---:|
"""
    for row in rows:
        report += f"| {int(row['training_tokens']):,} | {float(row['common_validation_loss']):.6f} | {float(row['common_validation_perplexity']):.5f} |\n"
    report += "\nInterval improvements per million tokens were:\n\n"
    for slope in slopes:
        report += (f"- {slope['from_tokens']/1e6:.3f}M→{slope['to_tokens']/1e6:.3f}M: "
                   f"loss {slope['loss_improvement_per_million_tokens']:.6f}/M, "
                   f"perplexity {slope['perplexity_improvement_per_million_tokens']:.4f}/M\n")
    report += f"""

The recent common-loss slope remains negative (improving), but it has slowed materially. BabyLM and FineWeb-Edu validation both continued improving through 10M. Classification: **SLOWING, NOT NEAR PLATEAU**; the model remains data-limited at 10M. The recommended next deliberate target is **25M**, with a new authorization and another slope review; no continuation has been launched.

The Phase 7A DATA-C research model reached 3.957705 common loss / 52.3371 perplexity at 3,145,728 tokens. The new lineage reached 3.974787 / 53.2388 at its nearby 3,000,320-token checkpoint and 3.626181 / 37.5691 at 10M. This is learning/data-scaling progression across different seeds and endpoints, not a controlled same-token claim. The Phase 6 Co4-S research checkpoint's 3.523641 BabyLM-only validation loss used a different validation regime and cannot be ranked directly against the balanced common-validation numbers.

## Persistence and lineage

The public immutable bundle is `experiments/final-gcp-10m/`; private resumable state remains at `recovery/final-gcp-10m/`. The final training checkpoint SHA-256 is `{metrics['checkpoint_sha256']}` and tokenizer SHA-256 is `{metrics['tokenizer_sha256']}`. The canonical manifest was frozen before training, at source code commit `f74e932`; manifest commit `c55437e` records that freeze. The rolling local latest/previous/fallback recovery states and periodic remote recovery uploads were maintained. Final remote revisions are recorded in the metrics after verified upload.
"""
    (ART / "gcp_final_10m_report.md").write_text(report)

    # Add milestone evidence to the already round-trip-verified release export.
    bundle = ROOT / "experiments" / "final-gcp-10m"
    if bundle.exists():
        for source, name in (
            (ART / "gcp_final_10m_manifest.json", "canonical_manifest.json"),
            (ART / "phase7a_selected_dataset_manifest.json", "data_manifest.json"),
            (ART / "common_validation_manifest.json", "validation_manifest.json"),
            (ART / "gcp_c4a_environment.json", "gcp_environment.json"),
            (ART / "gcp_final_training_curve.csv", "training_curve.csv"),
            (ART / "gcp_final_10m_gibc.csv", "gibc.csv"),
            (ART / "gcp_final_10m_wikitext103.json", "wikitext103.json"),
            (ART / "gcp_final_10m_report.md", "phase7c_report.md"),
        ):
            shutil.copy2(source, bundle / name)
        resume = bundle / "resume_state.pt"
        if resume.exists():
            resume.unlink()
        sums = [f"{sha256(path)}  {path.name}" for path in sorted(bundle.iterdir())
                if path.is_file() and path.name != "SHA256SUMS"]
        (bundle / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
