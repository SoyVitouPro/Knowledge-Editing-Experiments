#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/baselines/alpha_edit${PYTHONPATH:+:$PYTHONPATH}"

EDIT_LIMIT="${ALPHAEDIT_EDIT_LIMIT:-2000}"

run_dataset() {
  local dataset_type="$1"
  local data_file="$2"
  local run_label="$3"
  shift 3

  python "$ROOT/src/baselines/alpha_edit/run_experiment.py" \
    --model google/gemma-4-E2B \
    --dataset-type "$dataset_type" \
    --edit-data "$data_file" \
    --edit-limit "$EDIT_LIMIT" \
    --run-label "$run_label" \
    --dataset wikimedia/wikipedia \
    --dataset-config 20231101.en \
    --samples 100000 \
    "$@"
}

run_dataset counterfact \
  "$ROOT/data/counterfact_2000/train.jsonl" \
  counterfact_2000 \
  "$@"
run_dataset zsre \
  "$ROOT/data/zsre_2000/train.jsonl" \
  zsre_2000 \
  "$@"
run_dataset mquake \
  "$ROOT/data/mquake_2000/CF3k.jsonl" \
  mquake_2000 \
  "$@"