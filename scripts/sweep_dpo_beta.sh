#!/usr/bin/env bash
#
# Phase 3.2 — DPO beta sweep.
#
# Trains one DPO run per beta value, each from the SAME SFT checkpoint and the
# SAME preference pairs, writing to its own output directory so the results are
# directly comparable (clean attribution). Each beta is a separate process, so
# there is no model-state leakage between runs.
#
# Usage:
#   scripts/sweep_dpo_beta.sh [PAIRS_PATH] [SFT_CHECKPOINT] [BETAS...]
#
# Examples:
#   scripts/sweep_dpo_beta.sh
#   scripts/sweep_dpo_beta.sh outputs/pairs/dpo_pairs.jsonl outputs/sft/checkpoint-final
#   scripts/sweep_dpo_beta.sh outputs/pairs/dpo_pairs.jsonl outputs/sft/checkpoint-final 0.05 0.1 0.2 0.5
#
set -euo pipefail

PAIRS_PATH="${1:-outputs/pairs/dpo_pairs.jsonl}"
SFT_CHECKPOINT="${2:-outputs/sft/checkpoint-final}"
shift "$(( $# < 2 ? $# : 2 ))" || true
BETAS=("$@")
if [ "${#BETAS[@]}" -eq 0 ]; then
    BETAS=(0.05 0.1 0.3)
fi

echo "DPO beta sweep"
echo "  pairs:      ${PAIRS_PATH}"
echo "  sft ckpt:   ${SFT_CHECKPOINT}"
echo "  betas:      ${BETAS[*]}"
echo

for BETA in "${BETAS[@]}"; do
    OUT_DIR="outputs/dpo_beta${BETA}"
    echo "=================================================================="
    echo "  DPO beta=${BETA}  ->  ${OUT_DIR}"
    echo "=================================================================="
    python scripts/run_pipeline.py dpo \
        --device cuda \
        --beta "${BETA}" \
        --pairs-path "${PAIRS_PATH}" \
        --sft-checkpoint "${SFT_CHECKPOINT}" \
        --output-dir "${OUT_DIR}"
done

echo
echo "Sweep complete. Compare per-beta BFCL results:"
echo "  outputs/dpo_beta*/dpo_bfcl_results.json"
