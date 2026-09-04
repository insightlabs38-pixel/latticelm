#!/usr/bin/env bash
# Complete the canonical lineage sequentially. Credentials are environment-only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
: "${HF_TOKEN:?HF_TOKEN is required}"
export LATTICELM_HF_REPO="${LATTICELM_HF_REPO:-insightlabs38-pixel/LatticeLM-research}"

resume="${1:-artifacts/checkpoints/phase7b_recovery-1000448_step977.pt}"

metrics_for() {
  local checkpoint="$1" output="$2"
  python - "$checkpoint" "$output" <<'PY'
import csv, json, sys
checkpoint, output = sys.argv[1:]
rows = list(csv.DictReader(open("artifacts/final_training_curve.csv")))
row = next(item for item in reversed(rows) if item["checkpoint"] == checkpoint)
json.dump({"tokens_trained": int(row["training_tokens"]),
           "wall_seconds": float(row["cumulative_wall_seconds"]),
           "val_loss": float(row["common_validation_loss"]),
           "val_ppl": float(row["common_validation_perplexity"])}, open(output, "w"), indent=2)
PY
}

persist() {
  local label="$1" step="$2"
  local checkpoint="artifacts/checkpoints/phase7b_final-${label}_step${step}.pt"
  local staging="artifacts/hf_staging/phase7b-${label}"
  local metrics="/tmp/phase7b-${label}-metrics.json"
  metrics_for "final-${label}" "$metrics"
  export_one() {
    local role="$1" remote="$2"
    python scripts/export_release_model.py --checkpoint "$checkpoint" --output "$staging" \
      --experiment "phase7b-final-${label}" --role "$role" \
      --tokenizer artifacts/tokenizers/babylm_2026_4k.json \
      --tokenizer-report artifacts/tokenizers/babylm_2026_4k.report.json \
      --dataset-manifest artifacts/phase7a_selected_dataset_manifest.json --metrics "$metrics"
    python scripts/upload_hf_checkpoint.py --directory "$staging" --path "$remote"
  }
  export_one latest latest
  if python - "final-${label}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open("artifacts/final_training_curve.csv")))
target = next(row for row in reversed(rows) if row["checkpoint"] == sys.argv[1])
raise SystemExit(float(target["common_validation_loss"]) != min(float(row["common_validation_loss"]) for row in rows))
PY
  then
    export_one best best
  fi
  export_one named "experiments/final-${label}"
  resume="$checkpoint"
}

python scripts/run_phase7b_final.py --resume "$resume" --stop-step 9766
persist 10m 9766
python scripts/evaluate_gibc.py --checkpoint "$resume" \
  --tokenizer artifacts/tokenizers/babylm_2026_4k.json \
  --output artifacts/raw_phase7b_gibc_10m.json --tasks hellaswag,arc_easy,piqa,winogrande
python scripts/evaluate_wikitext103.py --checkpoint "$resume" \
  --tokenizer artifacts/tokenizers/babylm_2026_4k.json --output artifacts/raw_phase7b_wikitext_10m.json

python scripts/run_phase7b_final.py --resume "$resume" --stop-step 24415
persist 25m 24415

python scripts/run_phase7b_final.py --resume "$resume" --stop-step 48829
persist 50m 48829
python scripts/evaluate_gibc.py --checkpoint "$resume" \
  --tokenizer artifacts/tokenizers/babylm_2026_4k.json \
  --output artifacts/raw_phase7b_gibc_50m.json --tasks hellaswag,arc_easy,piqa,winogrande
python scripts/evaluate_wikitext103.py --checkpoint "$resume" \
  --tokenizer artifacts/tokenizers/babylm_2026_4k.json --output artifacts/raw_phase7b_wikitext_50m.json
