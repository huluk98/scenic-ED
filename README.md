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
python3 -m pip install -r requirements.txt
```

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

Run training:

```bash
python3 scripts/scenic_train_chatlm_sft.py --mode regular --epochs 3
python3 scripts/scenic_train_chatlm_sft.py --mode contrastive --epochs 3 --alignment-weight 0.1 --margin 0.5
```

The model path, dataset paths, and output directories can also be changed directly at the top of `scripts/scenic_train_chatlm_sft.py`.
