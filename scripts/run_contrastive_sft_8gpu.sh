#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

torchrun --standalone --nproc_per_node=8 contrastive_sft.py "$@"
