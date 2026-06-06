# Linear Sparsity and Progressive Recovery Experiments

## Purpose

This experiment block addresses the revision question of whether SCENIC pruning conclusions remain stable across multiple sparsity levels and whether progressive recovery fine-tuning improves accuracy retention. It evaluates dense 0% sparsity, one-shot magnitude pruning at 30% and 50%, and progressive magnitude pruning with recovery fine-tuning at 30% and 50%.

## Experimental Conditions

- `dense`: no pruning, target sparsity 0.0.
- `oneshot`: selected `torch.nn.Linear.weight` tensors are pruned once after checkpoint loading. This is the original manuscript-style pruning condition.
- `progressive`: selected Linear weights are pruned through staged masks. By default, the added progressive method performs one recovery epoch after each pruning stage plus one final recovery epoch after all stages.

The default sparsity levels are 0%, 30%, and 50%. These add a moderate pruning point and preserve the manuscript's current 50% setting so the paper can show whether conclusions are stable under less aggressive compression.

The full revision launcher also adds a legacy one-shot 30% follow-up for the named pruning methods that can genuinely hit 30% targeted sparsity: `magnitude`, `wanda`, and `gradient`. `nvidia24` is not included in that 30% add-on by default because 2:4 structured pruning is effectively 50% selected-weight sparsity.

## Pruning Scope

The runner prunes only Linear weights by default. It excludes biases, embeddings, normalization parameters, `lm_head`, classifier heads, response heads, final projection heads, and other output-head-like modules.

Use `--prune_output_heads` only for an explicit ablation. The default is false.

The outputs report both:

- `targeted_linear_sparsity_actual`: sparsity over the selected Linear weights only.
- `whole_model_sparsity_actual`: sparsity over all model parameters.

These differ because embeddings, norms, heads, and other excluded parameters remain dense.

## Pruning Methods

The new controlled experiment uses magnitude pruning:

- Per-layer unstructured pruning by default.
- Lowest absolute values are masked to reach the requested sparsity in every selected Linear layer.
- `--global_pruning` switches to a global threshold over all selected Linear weights.

Masks are saved with each pruned checkpoint and enforced after each recovery optimizer step. Regrowth is disabled by default; use `--regrowth` only for an explicit regrowth experiment.

## Progressive Schedule

For target 30%:

- 10%, 20%, 30%

For target 50%:

- 10%, 20%, 30%, 40%, 50%

At each stage, the runner updates masks, performs the configured recovery fine-tuning, reapplies masks after optimizer steps, and logs stage sparsity, validation EM@1, validation EM@5, and loss. The default keeps `--recovery_epochs_per_stage 1` and `--final_recovery_epochs 1`, so target 30% gets recovery after 10%, 20%, and 30% plus one final recovery epoch, while target 50% gets recovery after 10%, 20%, 30%, 40%, and 50% plus one final recovery epoch.

## EM@1 and EM@5

The evaluator normalizes predictions and targets before exact match:

- strip whitespace
- Unicode NFKC normalization
- collapse duplicated spaces
- standardize common punctuation variants
- preserve Chinese characters
- optionally remove all spaces with `--normalization_mode ignore_spaces`

For encoder-decoder and decoder-only models, the runner uses deterministic beam generation and forces at least five return sequences for EM@5. For encoder-only models, it expects `model.config.id2label` so top logits can map to canonical responses.

## Difficulty Labels

Difficulty is required for final reports. The evaluator first uses a benchmark field named `difficulty`, `complexity`, or `level`. If the benchmark lacks such a field, pass `--benchmark_difficulty_path`.

Supported external difficulty file formats are CSV, JSON, or JSONL with:

- `id,difficulty`
- `sample_id,difficulty`
- `input,difficulty`

The join order is sample id first, then exact input string. The runner raises an error rather than guessing labels.

Create a template with:

```bash
python scripts/create_benchmark_difficulty_template.py \
  --benchmark_path generated/iot_instruction_benchmark_200.json \
  --output_dir results/difficulty_labels
```

Fill `difficulty` with `easy`, `medium`, or `hard`.

## Reproduce Runs

Run the complete revised experiment suite from the original Hugging Face model with:

```bash
bash scripts/run_full_revision_experiments.sh
```

The default base model is `charent/ChatLM-mini-Chinese`. The launcher fine-tunes both regular SFT and contrastive SFT for five epochs, runs the existing one-shot 50% pruning/eval suite, runs the added one-shot 30% pruning/eval suite, and runs the new 0/30/50 sparsity matrix for both checkpoints.

By default the full launcher is now pruning-focused and skips ONNX (`RUN_ONNX=0`). The SFT and legacy pruning/eval steps use `NPROC_PER_NODE`/`torchrun`, and the added linear/progressive pruning jobs are split across visible GPUs by default with `RUN_SPARSITY_PARALLEL=1`. On an 8x H20 box, run:

```bash
NPROC_PER_NODE=8 SPARSITY_GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/run_full_revision_experiments.sh
```

Set `RUN_ONNX=1` only when ONNX precision/runtime tables are needed.

If a full run has already completed and its 0%/50% results look good, add only the 30% follow-up without retraining or rerunning the whole matrix:

```bash
bash scripts/run_30pct_revision_experiments.sh results/ChatLM-mini-Chinese_full_revision_YYYYMMDDTHHMMSSZ
```

By default this only runs the missing legacy 30% one-shot methods and writes `legacy_oneshot_30/`. Set `RUN_SPARSITY_30=1` if you also need a standalone 30%-only linear `oneshot`/`progressive` rerun; otherwise the main full launcher already includes `oneshot_30` and `progressive_30` in `linear_sparsity_0_30_50/`.

To pass a different original model:

```bash
bash scripts/run_full_revision_experiments.sh another/model-or-local-path
```

Example encoder-decoder run:

```bash
python scripts/run_sparsity_experiments.py \
  --experiment_name scenic_linear_sparsity_0_30_50 \
  --model_family encoder_decoder \
  --model_checkpoint PATH_TO_CHECKPOINT \
  --benchmark_path generated/iot_instruction_benchmark_200.json \
  --benchmark_difficulty_path results/difficulty_labels/benchmark_difficulty_template.csv \
  --sparsity_levels 0 0.3 0.5 \
  --pruning_modes dense oneshot progressive \
  --prune_scope linear_weights \
  --prune_method magnitude \
  --recovery_epochs_per_stage 1 \
  --final_recovery_epochs 1 \
  --num_beams 5 \
  --num_return_sequences 5 \
  --seed 42 \
  --output_dir results/scenic_linear_sparsity_0_30_50
```

Run one command per model family/checkpoint. Use `--model_family encoder_only`, `decoder_only`, or `encoder_decoder` as appropriate.

## Outputs

The runner writes:

- `final_revision_summary.json` from the full launcher, consolidating regular SFT and contrastive SFT outputs, the 7 original pruning-method outputs per model, the 2 added progressive outputs per model, training/benchmark EM@1/EM@5 for legacy pruning reports, and benchmark easy/medium/hard breakdowns when difficulty labels are available
- `predictions_{model_family}_{pruning_mode}_{sparsity}_{seed}.csv`
- `summary_metrics.csv`
- `paper_table_sparsity_difficulty.csv`
- `progressive_logs_{model_family}_{target_sparsity}_{seed}.csv`
- `checkpoints/.../linear_weight_masks.pt`
- saved pruned/recovered checkpoints
- `experiment_config.json`

Create figures with:

```bash
python scripts/plot_sparsity_results.py \
  --summary_csv results/scenic_linear_sparsity_0_30_50/summary_metrics.csv
```

Figures are saved under `results/{experiment_name}/figures/`.

## Paper Use

Use `paper_table_sparsity_difficulty.csv` for the main revised-paper table. It reports overall, easy, medium, and hard EM@1/EM@5 together with targeted Linear sparsity and whole-model sparsity. Use `summary_metrics.csv` for confidence intervals and accuracy-retention values relative to dense 0%.
