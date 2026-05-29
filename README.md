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
huggingface-cli download charent/ChatLM-mini-Chinese --local-dir models/ChatLM-mini-Chinese
python3 scripts/scenic_train_chatlm_sft.py --mode regular --model models/ChatLM-mini-Chinese --local-files-only --epochs 1 --max-examples 16
```

Run training:

```bash
python3 scripts/scenic_train_chatlm_sft.py --mode regular --epochs 3
python3 scripts/scenic_train_chatlm_sft.py --mode contrastive --epochs 3 --alignment-weight 0.1 --margin 0.5
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

The model path, dataset paths, and output directories can also be changed directly at the top of `scripts/scenic_train_chatlm_sft.py`.
