#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_gradient50_onnx_quant_baseline.sh [base-model-path-or-hf-id] [accuracy eval args]

Default:
  base model: charent/ChatLM-mini-Chinese

This launcher builds the sparse quantized ASIC baseline:
  1. Fine-tune the base model for 5 epochs with regular SFT.
  2. Create the 50% gradient one-shot pruned checkpoint.
  3. Export dense and gradient-pruned checkpoints to ONNX FP16.
  4. Export FP32 ONNX sources and dynamic-quantize them to ONNX INT8.
  5. Evaluate PyTorch FP16, ONNX FP16, and ONNX INT8 on benchmark/training EM1/EM5.
  6. Benchmark isolated latency, p95 latency, TPS, peak memory, and model size.

Useful overrides:
  OUTPUT_ROOT=onnx_eval_outputs/gradient50_asic_baseline
  NPROC_PER_NODE=8
  ACCURACY_GPU_IDS=0,1,2,3,4,5,6,7
  MAX_BENCHMARK_EXAMPLES=200
  MAX_TRAIN_EXAMPLES=
  LATENCY_QUERIES=200
  LATENCY_SEQ_LENGTHS="64 128"
  INT8_ONNX_PROVIDER=CUDAExecutionProvider

Final report:
  <OUTPUT_ROOT>/all_deployment_em_latency_report.json
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

BASE_MODEL="${1:-charent/ChatLM-mini-Chinese}"
if [[ $# -gt 0 ]]; then
  shift
fi

export PRUNE_METHOD="${PRUNE_METHOD:-gradient}"
export SPARSITY="${SPARSITY:-0.5}"
export PRUNED_VARIANT="${PRUNED_VARIANT:-gradient50}"
export PRUNED_PRETTY_LABEL="${PRUNED_PRETTY_LABEL:-50% gradient pruned}"
export RUN_INT8="${RUN_INT8:-1}"
export RUN_TENSORRT="${RUN_TENSORRT:-0}"
export TENSORRT_SPARSITY_ENABLE="${TENSORRT_SPARSITY_ENABLE:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-$NPROC_PER_NODE}"
export RUN_ACCURACY_PARALLEL="${RUN_ACCURACY_PARALLEL:-1}"
export ACCURACY_GPU_IDS="${ACCURACY_GPU_IDS:-0,1,2,3,4,5,6,7}"
export INT8_ONNX_PROVIDER="${INT8_ONNX_PROVIDER:-CUDAExecutionProvider}"

exec bash scripts/run_onnx_precision_prune_eval.sh "$BASE_MODEL" "$@"
