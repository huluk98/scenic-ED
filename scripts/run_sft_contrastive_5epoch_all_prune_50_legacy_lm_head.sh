#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/run_sft_contrastive_5epoch_all_prune_50_legacy_lm_head.sh <base-model-path-or-hf-id> [extra scenic_prune_eval.py args]

Example:
  bash scripts/run_sft_contrastive_5epoch_all_prune_50_legacy_lm_head.sh charent/ChatLM-mini-Chinese

This is the isolated fallback for the older full-linear 50% workflow. It prunes
all eligible encoder/decoder linear layers plus lm_head, which should produce
about 50% full-checkpoint sparsity on ChatLM-mini-Chinese.

Useful env overrides:
  REUSE_LAST_RUN=1     # reuse the latest legacy run directory and skip training
  SKIP_TRAIN=1         # reuse regular/contrastive checkpoints and only prune/eval
  PRUNE_METHODS=gradient # run just one method, or use the default full method set
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

BASE_MODEL="$1"
SAFE_BASE="$(basename "$BASE_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_BASE="${SAFE_BASE%_}"
SAFE_BASE="${SAFE_BASE:-chatlm}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

export PRUNE_LM_HEAD="${PRUNE_LM_HEAD:-1}"
export PRUNE_SCOPE="${PRUNE_SCOPE:-all-linear}"
export SPARSITY_BASIS="${SPARSITY_BASIS:-targeted-linear}"
export LATEST_RUN_FILE="${LATEST_RUN_FILE:-prune_eval_outputs/.latest_sft_contrastive5_all50_legacy_lm_head}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-prune_eval_outputs/${SAFE_BASE}_sft_contrastive5_all50_legacy_lm_head_${RUN_ID}}"

echo "Legacy lm_head pruning fallback enabled: PRUNE_LM_HEAD=${PRUNE_LM_HEAD}"
echo "Legacy output root: ${OUTPUT_ROOT}"

exec bash "$(dirname "$0")/run_sft_contrastive_5epoch_all_prune_50.sh" "$@"
