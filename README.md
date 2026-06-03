# SCENIC ED

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
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.8 PyTorch wheel on Linux, which is the right match for NVIDIA H20 machines with driver `570.124.06`.

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
  --epochs 3 \
  --bf16 \
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
python3 scripts/scenic_train_chatlm_sft.py --mode regular --epochs 3
python3 scripts/scenic_train_chatlm_sft.py --mode contrastive --epochs 3 --alignment-weight 0.1 --margin 0.5
python3 contrastive_sft.py --epochs 3 --alignment-weight 0.1 --margin 0.5
```

On a multi-GPU NVIDIA machine, launch with `torchrun`. `--batch-size` is per GPU, so this example has a global batch of `16 * 8 = 128` before gradient accumulation:

```bash
torchrun --standalone --nproc_per_node=8 scripts/scenic_train_chatlm_sft.py \
  --mode regular \
  --epochs 3 \
  --bf16 \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --num-workers 4
```

Contrastive triplet SFT uses the same launcher:

```bash
torchrun --standalone --nproc_per_node=8 scripts/scenic_train_chatlm_sft.py \
  --mode contrastive \
  --epochs 3 \
  --bf16 \
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

For `contrastive_sft.py`, the model, dataset, and output paths are provided directly in the `MODEL_PATH`, `TRAIN_JSON`, and `OUTPUT_DIR` variables at the top of the file. By default it expects the local ChatLM model at `models/ChatLM-mini-Chinese-local`, trains from `data/SCENIC_full_anchor_positive_negative.json`, writes to `models/chatlm_scenic_triplet_sft`, uses bf16, and loads local files only.

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
  --bf16 \
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

For H20 speed, prefer `--bf16`. The trainer loads model weights in bfloat16 and pads batches to multiples of 8 when bf16/fp16 is enabled, matching the faster path used by the Encoder-Decoder training scripts.

The model path, dataset paths, and output directories can also be changed directly at the top of `scripts/scenic_train_chatlm_sft.py` or the contrastive-only `contrastive_sft.py`.

## 50% Prune And Evaluate

After regular SFT or contrastive SFT finishes, run a 50% prune/eval pass with one model path:

```bash
bash scripts/run_prune_eval_50.sh models/chatlm_scenic_triplet_sft
```

The launcher writes one combined JSON report containing the original pre-prune EM@1/EM@5, the pruned EM@1/EM@5, benchmark results, full training-set results, model identity metadata, pruning stats, and predictions. By default it uses magnitude pruning and auto-detects available NVIDIA GPUs for evaluation.

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

By default, the prune/eval launchers now use `SPARSITY_BASIS=full-model` and `PRUNE_SCOPE=all-linear`, so `SPARSITY=0.5` targets 50% sparsity across the whole loaded checkpoint, not only the selected encoder weights. For the older reference-style encoder-only run, set `SPARSITY_BASIS=targeted-linear PRUNE_SCOPE=encoder-linear`.

After a run finishes, verify both full-model and targeted linear sparsity with:

```bash
python scripts/check_pruned_model_sparsity.py \
  --report-json prune_eval_outputs/<run>/all_sft_contrastive_pruning_em_report.json \
  --output-json prune_eval_outputs/<run>/sparsity_check.json
```

For the new full-model runs, `full_model_is_50_percent_sparse` should be true. For encoder-only reference runs, `expected_scope_is_50_percent_sparse` may be true while `full_model_is_50_percent_sparse` is false because decoder/output parameters are intentionally left dense.

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

This trains contrastive triplet SFT for 5 epochs, then runs 50% `magnitude`, `wanda`, `gradient`, and NVIDIA `2:4` pruning. It writes one final JSON at `prune_eval_outputs/<run>/all_pruning_em_report.json` with benchmark and training-set EM@1/EM@5 for the original contrastive model and each pruned model. If you already have the base model downloaded locally, pass that directory or force offline loading with `LOCAL_FILES_ONLY=1`.

To train and compare both regular SFT and contrastive SFT from the same original ChatLM Hugging Face id, use:

```bash
bash scripts/run_sft_contrastive_5epoch_all_prune_50.sh charent/ChatLM-mini-Chinese
```

This uses `data/SCENIC_full_training_dataset.json` for regular SFT and evaluation, `data/SCENIC_full_anchor_positive_negative.json` for contrastive SFT, and `generated/iot_instruction_benchmark_200.json` for benchmark EM. It writes one combined JSON at `prune_eval_outputs/<run>/all_sft_contrastive_pruning_em_report.json`.

For Chinese command responses, the launcher uses whitespace-insensitive exact match by default (`IGNORE_SPACES=1`) so tokenizer-inserted spaces do not count as wrong actions. Set `IGNORE_SPACES=0` if you need strict string equality.

When you pass the Hugging Face id directly, the launcher first downloads it into the run directory and trains from that local copy. This preserves the original tokenizer assets for pruning reloads and avoids fast-tokenizer backend errors such as `TokenizersBackend` or `argument 'vocab': 'dict' object cannot be converted to 'Sequence'`.

If the per-method outputs already exist and you only need to merge them into one JSON, run:

```bash
python scripts/aggregate_prune_eval_reports.py \
  --run-dir prune_eval_outputs/<run-directory>
```

This discovers `magnitude_50`, `wanda_50`, `gradient_50`, and `nvidia24_50` reports and writes `all_pruning_em_report.json` in the same run directory.
