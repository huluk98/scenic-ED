# SCENIC ED

Verified H20 ONNX Runtime GPU setup:

```bash
git pull
python - <<'PY'
import sys, torch, onnxruntime as ort
print("python:", sys.executable)
print("torch cuda:", torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("onnxruntime:", ort.__version__, ort.__file__)
print("providers:", ort.get_available_providers())
PY
```

Expected provider list after the fixed install:

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

If PyTorch reports 9 GPUs but you want only the 8 H20s, pin the run to GPUs 0-7:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
ACCURACY_GPU_IDS=0,1,2,3,4,5,6,7 \
FP16_ONNX_PROVIDER=CUDAExecutionProvider \
bash scripts/run_gradient50_onnx_quant_baseline.sh charent/ChatLM-mini-Chinese
```

ONNX FP16 should use `CUDAExecutionProvider`. Dynamic ONNX INT8 may still be slower or fall back because CUDA EP does not accelerate every quantized operator. If you only need benchmark accuracy and want to skip full training-data EM, add `MAX_TRAIN_EXAMPLES=0` to the command.

SCENIC edge-device training utilities and compact dataset artifacts for ChatLM-mini-Chinese SFT experiments.

## Kept Dataset Files

- `data/SCENIC_full_training_dataset.json`: regular SFT prompt/response records.
- `data/SCENIC_full_anchor_positive_negative.json`: anchor/positive/negative contrastive tuples with target responses.
- `generated/iot_instruction_benchmark_200.json`: 200-example IoT instruction benchmark.

Older intermediate datasets, audits, reports, and expansion files are intentionally not stored in this repo.

## Fine-Tune ChatLM-Mini-Chinese

`scripts/scenic_train_chatlm_sft.py` has two entry points for `charent/ChatLM-mini-Chinese`:

- `train_regular_sft(...)` trains normal prompt/response SFT for a configurable number of epochs.
- `train_contrastive_triplet_sft(...)` trains the compatibility-aware triplet SFT objective with anchor, positive, negative, and response tuples.

Install dependencies:

```bash
conda create -n scenic-ed python=3.10 -y
conda activate scenic-ed
python -m pip install -U pip
python -m pip uninstall -y torch torchvision torchaudio onnxruntime onnxruntime-gpu
python -m pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.8 PyTorch wheel on Linux, which is the right match for NVIDIA H20 machines with driver `570.124.06`.

For the H20 ONNX/TensorRT deployment launcher, make sure the active Python environment exposes the GPU ONNX Runtime provider:

```bash
python - <<'PY'
import sys
import torch
import onnxruntime as ort
print("python:", sys.executable)
print("torch cuda:", torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("onnxruntime:", ort.__version__, ort.__file__)
print("providers:", ort.get_available_providers())
PY
```

The provider list must include `CUDAExecutionProvider`. If it only shows `AzureExecutionProvider` and `CPUExecutionProvider`, that environment is using CPU-only ONNX Runtime or a conflicting install. Fix it in the same conda environment with:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install --extra-index-url https://pypi.nvidia.com "onnxruntime-gpu[cuda,cudnn]>=1.19" "optimum[onnxruntime-gpu]>=1.23" "onnx>=1.16" "onnxscript>=0.3" "safetensors>=0.4.5" "tensorrt>=10"
```

The native TensorRT benchmark also needs NVIDIA `trtexec` on `PATH`; check it with `trtexec --version`. If `trtexec` is missing, install TensorRT from NVIDIA packages or use an NVIDIA TensorRT container before running `scripts/run_h20_encoder_decoder_sft_prune_trt24.sh`.

The H20 launcher also exports calibrated static QDQ INT8 ONNX artifacts from real calibration examples, not random ranges:

- `onnx/dense_sft_int8_qdq/model.onnx`
- `onnx/nvidia_2_4_sft_int8_qdq/model.onnx`
- `reports/int8_status.json`

These files are enough to test ONNX Runtime GPU provider behavior for INT8. Treat NVIDIA 2:4 sparse hardware acceleration as proven only when the backend exposes evidence that sparse kernels or sparse tactics were actually used.

If your base model is a Hugging Face repo id, pass it directly. The H20 launcher downloads that repo into `<output_dir>/base_model` first, then trains from the local snapshot with `trust_remote_code=True`, so the original Hugging Face `modeling.py` and related custom-code files are used and preserved for later pruning/export steps:

```bash
bash scripts/run_h20_encoder_decoder_sft_prune_trt24.sh \
  --base_model charent/ChatLM-mini-Chinese
```

Verify the CUDA wheel after install:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Expected on the H20 box: `torch.__version__` contains `+cu128`, `torch.version.cuda` is `12.8`, and `gpu count` is `8`.

Run quick data checks:

```bash
python3 scripts/scenic_train_chatlm_sft.py --mode regular --dry-run
python3 scripts/scenic_train_chatlm_sft.py --mode contrastive --dry-run
python3 contrastive_sft.py --dry-run
python3 -m pytest
```

Run a tiny smoke training job before a full epoch run:

```bash
python3 scripts/scenic_train_chatlm_sft.py --mode regular --epochs 1 --max-examples 16 --batch-size 2
```

If Hugging Face download fails with an SSL certificate or hostname mismatch error, first refresh the local Python certificate stack:

```bash
python3 -m pip install -U certifi huggingface_hub transformers requests urllib3
```

If the network still rewrites Hugging Face certificates, download the model once from a working network and train from the local directory:

```bash
hf download charent/ChatLM-mini-Chinese --local-dir models/ChatLM-mini-Chinese
python3 scripts/scenic_train_chatlm_sft.py --mode regular --model models/ChatLM-mini-Chinese --local-files-only --epochs 1 --max-examples 16
```

If an 8-GPU run fails at startup with `Cannot send a request, as the client has been closed`, download the model once before launching DDP and then run with `--local-files-only`:

```bash
hf download charent/ChatLM-mini-Chinese --local-dir models/ChatLM-mini-Chinese
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
torchrun --standalone --nproc_per_node=8 scripts/scenic_train_chatlm_sft.py \
  --mode regular \
  --model models/ChatLM-mini-Chinese \
  --local-files-only \
  --epochs 5 \
  --fp16 \
  --batch-size 16
```

Use the same local/offline pattern if loading weights succeeds but then fails on an SSL hostname mismatch for `additional_chat_templates`; that is a late Transformers Hub lookup, not a training failure.

If the model is already in the Hugging Face cache and `hf download` fails, skip the download and find the cached snapshot:

```bash
python scripts/find_chatlm_cache.py
```

Then use the printed snapshot path with `--model <snapshot-path> --local-files-only`.

If Transformers still tries `HEAD generate.py` or another custom-code file, build a self-contained local model directory from the existing cache:

```bash
python scripts/prepare_chatlm_local_model.py
```

Then use the printed `models/ChatLM-mini-Chinese-local` path with `--local-files-only`.

Run training:

```bash
python3 scripts/scenic_train_chatlm_sft.py --mode regular
python3 scripts/scenic_train_chatlm_sft.py --mode contrastive --alignment-weight 0.1 --margin 0.5
python3 contrastive_sft.py --alignment-weight 0.1 --margin 0.5
```

Regular SFT now follows the older `sf-2.py` recipe by default: 5 epochs, 256-token inputs, 128-token targets, fp16, fixed-length tokenization, step checkpoints every 500 steps, and final output in `sft5`.

On a multi-GPU NVIDIA machine, launch with `torchrun`. `--batch-size` is per GPU, so this example has a global batch of `16 * 8 = 128` before gradient accumulation:

```bash
torchrun --standalone --nproc_per_node=8 scripts/scenic_train_chatlm_sft.py \
  --mode regular \
  --epochs 5 \
  --fp16 \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --num-workers 4
```

Contrastive triplet SFT uses the same launcher:

```bash
torchrun --standalone --nproc_per_node=8 scripts/scenic_train_chatlm_sft.py \
  --mode contrastive \
  --epochs 5 \
  --fp16 \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --alignment-weight 0.1 \
  --margin 0.5 \
  --num-workers 4
```

Or use the dedicated contrastive-only file:

```bash
bash scripts/run_contrastive_sft_8gpu.sh
```

For `contrastive_sft.py`, the model, dataset, and output paths are provided directly in the `MODEL_PATH`, `TRAIN_JSON`, and `OUTPUT_DIR` variables at the top of the file. By default it expects the local ChatLM model at `models/ChatLM-mini-Chinese-local`, trains for 5 epochs from `data/SCENIC_full_anchor_positive_negative.json`, writes to `models/chatlm_scenic_triplet_sft`, uses fp16, and loads local files only.

The triplet objective is pair-balanced: for each tuple it averages the anchor and positive generation losses, then adds `alignment_weight * max(0, margin + d(anchor, positive) - d(anchor, negative))` using cosine distance over L2-normalized encoder representations.

`contrastive_sft.py` also enables NCCL async error handling, uses a 10-minute DDP timeout, and cleans up the process group plus CUDA cache on normal exit, Ctrl-C, or termination. If a previous failed run already left Python ranks alive on the server, kill those stale processes once before relaunching:

```bash
pkill -f "torchrun.*contrastive_sft.py"
pkill -f "contrastive_sft.py"
nvidia-smi
```

If `nvidia-smi` still shows old Python PIDs after that, terminate those PIDs directly with `kill -9 <pid>`. No Python cleanup hook can run after `kill -9` or a driver-level hard hang, but GPU memory is released when the owning process is gone.

The dedicated `contrastive_sft.py` saves only the final model by default. Per-epoch checkpoint saves can look like a hang because rank 0 is writing model files while the other ranks wait at a DDP synchronization point. To enable epoch checkpoints anyway, pass `--epoch-checkpoints`; to disable them in the shared trainer, pass `--no-epoch-checkpoints`.

For the final save, all ranks now synchronize once, DDP is shut down, non-main ranks release GPU memory, and only rank 0 writes the model. The default final save moves the model to CPU first, so `nvidia-smi` may look mostly idle while the model is being written. If safetensors is slow on your filesystem, retry with:

```bash
torchrun --standalone --nproc_per_node=8 contrastive_sft.py --no-safe-serialization
```

For a quick 8-GPU smoke test:

```bash
torchrun --standalone --nproc_per_node=8 scripts/scenic_train_chatlm_sft.py \
  --mode regular \
  --epochs 1 \
  --max-examples 128 \
  --fp16 \
  --batch-size 4
```

Check CUDA before training:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Single-process `python scripts/scenic_train_chatlm_sft.py ...` uses one GPU. Use `torchrun --nproc_per_node=8` to use all 8 H20 GPUs.

The training scripts now default to fp16. Set `PRECISION=bf16` for the shell launchers or pass `--bf16` directly if you want bfloat16 instead.

The model path, dataset paths, and output directories can also be changed directly at the top of `scripts/scenic_train_chatlm_sft.py` or the contrastive-only `contrastive_sft.py`.

## Original ChatLM Baseline

To test the untouched Hugging Face ChatLM model on the same benchmark and full training dataset, run:

```bash
python scripts/evaluate_original_chatlm.py \
  charent/ChatLM-mini-Chinese \
  --output-json prune_eval_outputs/original_chatlm_eval_report.json
```

To run the same baseline across all 8 NVIDIA H20 GPUs, use the launcher:

```bash
NPROC_PER_NODE=8 \
bash scripts/run_original_chatlm_eval_8gpu.sh charent/ChatLM-mini-Chinese
```

If Hugging Face download/cache access is failing with `LocalEntryNotFoundError` or SSL EOF errors, point the launcher at the local model directory instead:

```bash
NPROC_PER_NODE=8 \
bash scripts/run_original_chatlm_eval_8gpu.sh /path/to/ChatLM-mini-Chinese
```

You can also keep the source id in the report while evaluating a local folder:

```bash
HF_MODEL_PATH=/path/to/ChatLM-mini-Chinese \
NPROC_PER_NODE=8 \
bash scripts/run_original_chatlm_eval_8gpu.sh charent/ChatLM-mini-Chinese
```

The 8-GPU launcher now tries paths in this order: a local first argument, `HF_MODEL_PATH`, a self-contained copy built from the local Hugging Face cache, then network download when `LOCAL_FILES_ONLY=0` or the default `auto` permits it. Set `LOCAL_FILES_ONLY=1` to force offline/local-only evaluation. Override `OUTPUT_ROOT` or `OUTPUT_JSON` if you want a fixed report path.

The report includes benchmark/training EM@1 and EM@5 plus an `eos` block. By default the script ensures the tokenizer EOS id is present in both the model config and generation config before evaluation, then samples generated benchmark beams to check whether outputs terminate with EOS. Use `--local-files-only` with a local model path if the base model is already downloaded.

## 50% Prune And Evaluate

After regular SFT or contrastive SFT finishes, run a 50% prune/eval pass with one model path:

```bash
bash scripts/run_prune_eval_50.sh models/chatlm_scenic_triplet_sft
```

The launcher writes one combined JSON report containing the original pre-prune EM@1/EM@5, the pruned EM@1/EM@5, benchmark results, full training-set results, model identity metadata, pruning stats, and predictions. By default it uses magnitude pruning and auto-detects available NVIDIA GPUs for evaluation.

### Gradient-50 ONNX Quantized Baseline

If you only need the accuracy change from dense regular SFT to 50% gradient-pruned regular SFT, and your ONNX Runtime does not expose `CUDAExecutionProvider`, use the 8-GPU accuracy-only launcher:

```bash
NPROC_PER_NODE=8 bash scripts/run_gradient50_accuracy_only_8gpu.sh charent/ChatLM-mini-Chinese
```

It skips ONNX export, INT8, and latency/TPS entirely. The launcher writes `<OUTPUT_ROOT>/gradient50_accuracy_delta_summary.json` with dense EM@1/EM@5, gradient-50 EM@1/EM@5, deltas, retention, target sparsity, and whole-model sparsity. This is the fastest path when the goal is just benchmark/training accuracy preservation.

For the sparse quantized ASIC baseline, run the one-command launcher from the original Hugging Face model:

```bash
NPROC_PER_NODE=8 ACCURACY_GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/run_gradient50_onnx_quant_baseline.sh charent/ChatLM-mini-Chinese
```

This trains regular SFT for 5 epochs, creates the 50% gradient one-shot pruned checkpoint, exports dense and pruned ONNX FP16, exports FP32 ONNX sources, dynamic-quantizes them to ONNX INT8, evaluates benchmark/training EM@1 and EM@5, then benchmarks isolated latency, p95 latency, TPS, peak memory, and model size. Accuracy evaluations are fanned out across the listed GPUs, while latency is run sequentially so the timing numbers are not contaminated by parallel jobs. The final JSON is `<OUTPUT_ROOT>/all_deployment_em_latency_report.json` and includes `accuracy_delta_table`, `model_size_table`, `runtime_benchmark`, and target/whole-model sparsity from the pruning summary.

This is intended as a sparse quantized baseline for true ASIC comparison, not as a complete edge deployment claim. Gradient pruning is unstructured, so `TENSORRT_SPARSITY_ENABLE` defaults to `0`; only set it to `1` for a structured NVIDIA 2:4 run.

Choose another 50% pruning method with `METHOD`:

```bash
METHOD=wanda bash scripts/run_prune_eval_50.sh models/chatlm_scenic_triplet_sft
METHOD=gradient bash scripts/run_prune_eval_50.sh models/chatlm_scenic_triplet_sft
METHOD=nvidia bash scripts/run_prune_eval_50.sh models/chatlm_scenic_triplet_sft
```

Override outputs or limit rows for a smoke test:

```bash
bash scripts/run_prune_eval_50.sh models/chatlm_scenic_triplet_sft \
  --output-json prune_eval_outputs/triplet_50/prune_eval_report.json \
  --pruned-output-dir prune_eval_outputs/triplet_50/pruned_model \
  --max-train-examples 128 \
  --max-benchmark-examples 50
```

`scripts/scenic_prune_eval.py` supports `magnitude`, `gradient`, `wanda`, and NVIDIA `2:4` pruning. The report stores `accuracy` as exact-match@1 so you can verify the checkpoint you passed in before comparing the pruned model.

By default, the prune/eval launchers now use `SPARSITY_BASIS=targeted-linear` and `PRUNE_SCOPE=all-linear`, so `SPARSITY=0.5` targets 50% sparsity in each selected encoder/decoder linear layer while leaving `lm_head` dense. This protects the final vocabulary projection from pruning. For an encoder-only reference run, set `PRUNE_SCOPE=encoder-linear`; to intentionally include the output head, set `PRUNE_LM_HEAD=1` or pass `--prune-lm-head`.

For unstructured `magnitude`, `gradient`, and `wanda`, pruning is per selected linear layer, so each pruned layer lands near the requested sparsity instead of borrowing zeros from another layer. NVIDIA `2:4` remains structured by definition, so eligible linear weights are pruned with the 2-of-4 pattern; the final sparsity check reports any layer that does not reach the expected 50%.

### Legacy lm_head Full-Linear Fallback

If the protected-`lm_head` run underperforms, use the isolated legacy launcher to reproduce the earlier full-linear 50% workflow:

```bash
bash scripts/run_sft_contrastive_5epoch_all_prune_50_legacy_lm_head.sh charent/ChatLM-mini-Chinese
```

To reuse the latest legacy run and only redo pruning/eval:

```bash
REUSE_LAST_RUN=1 SKIP_TRAIN=1 \
bash scripts/run_sft_contrastive_5epoch_all_prune_50_legacy_lm_head.sh charent/ChatLM-mini-Chinese
```

This wrapper sets `PRUNE_LM_HEAD=1`, `PRUNE_SCOPE=all-linear`, and `SPARSITY_BASIS=targeted-linear`, then writes into a separate `*_legacy_lm_head_*` run directory. On ChatLM-mini-Chinese it should prune 161 targeted linear layers, including `lm_head`, and land near 50% full-checkpoint sparsity. The default protected-head workflow should prune 160 targeted linear layers and land near 44% full-checkpoint sparsity. New legacy reports record this automatically; for older reports without `prune_lm_head` metadata, pass `--include-lm-head` to `scripts/check_pruned_model_sparsity.py` when manually verifying.

The all-in-one launchers run this sparsity check automatically at the end and write `sparsity_check.json` in the run directory. To verify a run manually, use:

```bash
python scripts/check_pruned_model_sparsity.py \
  --report-json prune_eval_outputs/<run>/all_sft_contrastive_pruning_em_report.json \
  --output-json prune_eval_outputs/<run>/sparsity_check.json
```

For the default per-layer runs, `expected_scope_is_50_percent_sparse` and `expected_scope_layers_are_50_percent_sparse` should both be true. `full_model_is_50_percent_sparse` may be false because `lm_head`, embeddings, biases, and other non-linear parameters are intentionally left dense.

To inspect active/nonzero parameters for one model path directly:

```bash
python scripts/check_pruned_model_sparsity.py \
  --model-path prune_eval_outputs/<run>/contrastive_sft/gradient_50/pruned_model \
  --output-json prune_eval_outputs/<run>/contrastive_gradient_active_params.json
```

For a simpler edit-and-run version, set `MODEL_PATH` at the top of `scripts/count_active_params_simple.py`, then run `python scripts/count_active_params_simple.py`.

To run the full pipeline from the base ChatLM model in one command, use:

```bash
bash scripts/run_contrastive_5epoch_all_prune_50.sh charent/ChatLM-mini-Chinese
```

This trains contrastive triplet SFT for 5 epochs, then runs 50% per-layer `magnitude`, `wanda`, `gradient`, and NVIDIA `2:4` pruning. It writes one final JSON at `prune_eval_outputs/<run>/all_pruning_em_report.json` with benchmark and training-set EM@1/EM@5 for the original contrastive model and each pruned model, then writes `sparsity_check.json` and fails the shell if any generated pruned checkpoint is not at the expected 50% sparsity basis or per-layer check. If you already have the base model downloaded locally, pass that directory or force offline loading with `LOCAL_FILES_ONLY=1`.

To train and compare both regular SFT and contrastive SFT from the same original ChatLM Hugging Face id, use:

```bash
bash scripts/run_sft_contrastive_5epoch_all_prune_50.sh charent/ChatLM-mini-Chinese
```

This uses `data/SCENIC_full_training_dataset.json` for regular SFT and evaluation, `data/SCENIC_full_anchor_positive_negative.json` for contrastive SFT, and `generated/iot_instruction_benchmark_200.json` for benchmark EM. It writes one combined JSON at `prune_eval_outputs/<run>/all_sft_contrastive_pruning_em_report.json`, then writes `sparsity_check.json` and fails the shell if any generated pruned checkpoint is not at the expected 50% sparsity basis or per-layer check.

To reuse the latest combined run in one command and skip both SFT training steps, run:

```bash
REUSE_LAST_RUN=1 bash scripts/run_sft_contrastive_5epoch_all_prune_50.sh charent/ChatLM-mini-Chinese
```

The shell launchers default to `PRECISION=fp16`; set `PRECISION=bf16` or `PRECISION=fp32` only when you want to override that.

For Chinese command responses, the launcher uses whitespace-insensitive exact match by default (`IGNORE_SPACES=1`) so tokenizer-inserted spaces do not count as wrong actions. Set `IGNORE_SPACES=0` if you need strict string equality.

When you pass the Hugging Face id directly, the launcher first downloads it into the run directory and trains from that local copy. This preserves the original tokenizer assets for pruning reloads and avoids fast-tokenizer backend errors such as `TokenizersBackend` or `argument 'vocab': 'dict' object cannot be converted to 'Sequence'`.

If the per-method outputs already exist and you only need to merge them into one JSON, run:

```bash
python scripts/aggregate_prune_eval_reports.py \
  --run-dir prune_eval_outputs/<run-directory>
```

This discovers `magnitude_50`, `wanda_50`, `gradient_50`, and `nvidia24_50` reports and writes `all_pruning_em_report.json` in the same run directory.
