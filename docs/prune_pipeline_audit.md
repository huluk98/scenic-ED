# Prune Pipeline Audit

Audit date: 2026-06-08

This is an independent source-level audit of the current SCENIC pruning pipeline. I checked the launchers, pruning implementations, aggregation code, bundled benchmark data, and the focused pruning tests. There are no completed JSON reports in the local `prune_eval_outputs/` folders at audit time, so this is not a numerical post-run audit.

## What A Default Full Run Generates

Entry point: `bash scripts/run_full_revision_experiments.sh`

Default output root:

```text
results/${SAFE_BASE}_full_revision_${RUN_ID}/
```

For the default base model, that is shaped like:

```text
results/ChatLM-mini-Chinese_full_revision_<timestamp>/
```

The full launcher creates these major outputs:

1. SFT checkpoints under `legacy_regular_contrastive_5epoch_prune50/`
   - `regular_sft_5epoch/`
   - `contrastive_sft_5epoch/`

2. Legacy 50% one-shot prune/eval outputs under `legacy_regular_contrastive_5epoch_prune50/`
   - Per model: `regular_sft` and `contrastive_sft`
   - Per method: `magnitude_50`, `wanda_50`, `gradient_50`, `nvidia24_50`
   - Each method directory writes:
     - `prune_eval_report.json`
     - `pruned_model/`
   - Aggregate outputs:
     - `all_sft_contrastive_pruning_em_report.json`
     - `sparsity_check.json`

3. Legacy 30% one-shot follow-up under `legacy_oneshot_30/`
   - Per model: `regular_sft` and `contrastive_sft`
   - Per method by default: `magnitude_30`, `wanda_30`, `gradient_30`
   - `nvidia24` is excluded by default because 2:4 is effectively 50% selected-weight sparsity.
   - Each method directory writes:
     - `prune_eval_report.json`
     - `pruned_model/`
   - Aggregate outputs:
     - `all_sft_contrastive_pruning_em_report_30.json`
     - `sparsity_check_30.json`

4. Linear sparsity matrix under `linear_sparsity_0_30_50/`
   - Per model: `regular_sft/` and `contrastive_sft/`
   - Default conditions per model:
     - `dense_0`
     - `oneshot_30`
     - `oneshot_50`
     - `progressive_30`
     - `progressive_50`
   - If `RUN_SPARSITY_PARALLEL=1` and multiple GPUs are detected, each condition first writes under `jobs/<condition>/`, then `aggregate_sparsity_job_summaries.py` writes combined CSVs to the model-level directory.
   - Each condition writes at least:
     - `experiment_config.json`
     - `summary_metrics.csv`
     - `paper_table_sparsity_difficulty.csv`
     - `predictions_{model_family}_{mode}_{sparsity}_{seed}.csv`
   - `oneshot` and `progressive` conditions also write:
     - `checkpoints/.../`
     - `checkpoints/.../linear_weight_masks.pt`
   - `progressive` conditions also write:
     - `progressive_logs_{model_family}_{target_sparsity}_{seed}.csv`

5. Final full-run summary and manifest
   - `final_revision_summary.json`
   - `full_revision_manifest.txt`

6. Optional ONNX precision/runtime outputs
   - Disabled by default with `RUN_ONNX=0`.
   - If enabled, outputs go under `onnx_precision/{regular_sft,contrastive_sft}/`.
   - This full-revision ONNX step is a dense precision benchmark. It is not the clean pruned ONNX deployment path.

Count-wise, the default full run plans, per model:

- 7 legacy one-shot method outputs: magnitude, Wanda, and gradient at 30% and 50%, plus NVIDIA 2:4 at 50%.
- 5 linear matrix rows: dense 0%, one-shot magnitude 30%, one-shot magnitude 50%, progressive magnitude 30%, progressive magnitude 50%.
- 11 saved pruned/recovered checkpoints per model: 7 legacy pruned checkpoints plus 4 matrix pruned/recovered checkpoints. Dense 0% is evaluated but does not save a separate dense checkpoint.

Across both regular SFT and contrastive SFT, that is 14 legacy pruned checkpoints, 8 linear-matrix pruned/recovered checkpoints, and 2 dense SFT checkpoints.

## What The Standalone Launchers Generate

`scripts/run_sft_contrastive_5epoch_all_prune_50.sh`

- Trains or reuses the two 5-epoch checkpoints.
- Runs the 50% legacy suite for `magnitude`, `wanda`, `gradient`, and `nvidia24`.
- Writes one per-method `prune_eval_report.json`, one per-method `pruned_model/`, the aggregate `all_sft_contrastive_pruning_em_report.json`, and `sparsity_check.json`.

`scripts/run_30pct_revision_experiments.sh`

- Reuses checkpoints from a full revision run.
- Runs 30% legacy one-shot methods for `magnitude`, `wanda`, and `gradient` by default.
- Optionally runs a 30%-only linear matrix if `RUN_SPARSITY_30=1`.

`scripts/run_prune_eval_50.sh`

- Despite the name, it can run any `SPARSITY` value supplied by env.
- For one input checkpoint and one method, it writes:
  - `${RUN_DIR}/prune_eval_report.json`
  - `${RUN_DIR}/pruned_model/`

`scripts/run_clean_h20_onnx_fp16_int8.sh` and `scripts/run_gradient50_onnx_quant_baseline.sh`

- These are the intended pruned ONNX FP16/INT8 export paths for the strongest 50% contrastive row.
- They default to `FINETUNE_MODE=contrastive`, `FINETUNE_TRAIN_JSON=data/SCENIC_full_anchor_positive_negative.json`, `PRUNE_METHOD=gradient`, and `SPARSITY=0.5`.
- They export `onnx/sft5_fp16_pruned` directly from the contrastive gradient-50 checkpoint, then create `onnx/sft5_int8_pruned` by dynamic-quantizing the pruned FP32 ONNX source.
- They now set `ENFORCE_CONTRASTIVE_GRADIENT50=1` by default, so accidental environment overrides to regular SFT, another train JSON, another method, or another sparsity fail unless that guard is intentionally disabled.
- New pruning summaries and final deployment reports record `fine_tune`, `pruning_contract`, and `onnx_artifacts` provenance fields.

## Pruning Guideline Compliance

The written guideline in `docs/linear_sparsity_progressive_experiments.md` says the controlled experiment should:

- prune selected `torch.nn.Linear.weight` tensors only;
- exclude biases, embeddings, norms, `lm_head`, classifiers, response heads, and final projections by default;
- use per-layer unstructured magnitude pruning by default;
- report targeted Linear sparsity separately from whole-model sparsity;
- save masks and enforce them after recovery optimizer steps;
- use staged progressive schedules of 10/20/30% and 10/20/30/40/50%.

The linear matrix implementation follows those rules:

- `collect_prunable_linear_modules()` only collects `torch.nn.Linear` modules, respects encoder/decoder scope, and excludes output-head-like modules unless `--prune_output_heads` is set.
- `apply_magnitude_masks()` does per-layer magnitude masking by default and supports global thresholding only if `--global_pruning` is explicitly enabled.
- `progressive_schedule()` implements 30% as 10/20/30 and 50% as 10/20/30/40/50.
- `recovery_finetune()` reapplies masks after each optimizer step unless `--regrowth` is enabled.
- `run_single_experiment()` records targeted Linear sparsity, whole-model sparsity, checkpoint path, mask path, predictions, summaries, and progressive logs.

The focused tests support these invariants:

- 20 focused pruning tests pass in `conda run -n scenic-ed`.
- They cover per-layer magnitude sparsity, default `lm_head` exclusion, Wanda smoke behavior, 30% and 50% targeted sparsity, and mask reapplication after optimizer steps.

## Method Behavior And Differences

The pipeline has two distinct pruning families.

Legacy one-shot prune/eval:

- `magnitude`: per-selected-linear-layer unstructured magnitude pruning.
- `gradient`: one-shot saliency pruning using `abs(weight * grad)` accumulated on calibration batches.
- `wanda`: one-shot activation-aware pruning using `abs(weight) * sqrt(mean(input_activation^2))`.
- `nvidia24`: structured 2:4 pruning within groups of four input-channel weights for eligible Linear layers.

Controlled linear matrix:

- Only supports `--prune_method magnitude`.
- Adds `dense`, `oneshot`, and `progressive` modes at 0/30/50%.
- The progressive rows are magnitude pruning plus recovery fine-tuning, not a separate named pruning method like Wanda or gradient pruning.

Important differences from a stricter or paper-style interpretation:

1. The default sparsity basis is targeted Linear sparsity, not whole-model sparsity.
   - This is intentional and documented.
   - Whole-model sparsity will be lower because embeddings, norms, heads, and other excluded parameters stay dense.

2. The legacy report phase name is always `pruned_after_50_percent`.
   - This key is still used when the 30% launcher runs with `SPARSITY=0.3`.
   - The underlying `pruning.sparsity` value is correct, but the phase label is misleading for 30% reports.

3. Legacy and matrix evaluators normalize text differently.
   - `scenic_prune_eval.py` only strips whitespace and optionally removes all spaces.
   - `run_sparsity_experiments.py` uses Unicode NFKC normalization, punctuation normalization, space collapsing, and optional no-space mode.
   - This means legacy EM numbers and linear-matrix EM numbers are close in intent but not exactly the same evaluator contract.

4. The legacy one-shot outputs and the linear matrix overlap conceptually but are not duplicate artifacts.
   - Legacy `magnitude_50` evaluates dense and pruned checkpoints on benchmark and training datasets in one report.
   - Matrix `oneshot_50` evaluates benchmark difficulty slices, saves masks, and contributes retention against the dense 0% matrix row.

5. NVIDIA 2:4 is pure 2:4 by default under `targeted-linear`, but can become hybrid if run with `--sparsity-basis full-model` and `--full-model-correction`.
   - That correction magnitude-prunes non-protected parameters to hit full-model sparsity.
   - The default full pipeline uses `targeted-linear`, so this hybrid behavior is not active by default.

6. `nvidia24` cannot honestly represent 30% sparsity.
   - The code and docs handle this correctly by excluding it from the default 30% follow-up.

## Bottom Line

The current default full pipeline follows the repo's documented pruning plan for the controlled linear sparsity experiment. The biggest thing to be precise about in writing is that the main comparison is targeted Linear sparsity, not full-model sparsity.

The meaningful differences are labeling and comparability issues, not core pruning failures:

- 30% legacy reports still use the `pruned_after_50_percent` phase key.
- Legacy EM normalization is simpler than the controlled matrix evaluator.
- Progressive pruning is magnitude-plus-recovery and should not be described as a separate pruning algorithm.
- Legacy 50% magnitude and matrix `oneshot_50` are related but generated by different runners with different reporting contracts.

Recommended cleanup before using this in a paper or final appendix:

- Rename or alias the legacy phase key to `pruned_after_prune` or `pruned_after_{sparsity}_percent` in future reports.
- State in method text that sparsity targets apply to selected Linear weights by default.
- Keep legacy method tables separate from the controlled linear sparsity/progressive matrix, or clearly label the evaluator differences.

## Fair Three-Way Evaluation Contract

The downloaded `final_revision_summary.json` was generated by the encoder-decoder full revision run:

```text
scripts/run_full_revision_experiments.sh
base_model=charent/ChatLM-mini-Chinese
run_root=results/ChatLM-mini-Chinese_full_revision_20260606T132621Z
```

That summary is an encoder-decoder result with two trained variants:

- `regular_sft`
- `contrastive_sft`

It should be compared to decoder-only and encoder-only runs through the linear sparsity matrix contract, not by mixing each repo's older custom report format.

For a fair three-way architecture comparison, keep these fixed:

- same benchmark: `generated/iot_instruction_benchmark_200.json`
- same train/eval split policy and no hidden extra examples
- same seed, default `42`
- same sparsity levels: `0`, `0.3`, `0.5`
- same pruning modes for the controlled table: `dense`, `oneshot`, `progressive`
- same pruning method for the controlled table: `magnitude`
- same sparsity basis: targeted Linear sparsity
- same output-head policy: exclude output heads by default
- same EM normalization: prefer the `run_sparsity_experiments.py` normalization contract for all three
- same difficulty labels and same overall/easy/medium/hard reporting

Architecture-specific differences are acceptable only where the model interface requires them:

- `encoder_decoder`: load with `AutoModelForSeq2SeqLM`; generate responses directly from encoded prompts.
- `decoder_only`: load with `AutoModelForCausalLM`; generate prompt plus completion, then score only new tokens after the prompt.
- `encoder_only`: load with `AutoModelForSequenceClassification`; map top logits through `config.id2label`, and ensure every target response exists in `config.label2id`.

The shared runner already has these branches through `--model_family encoder_decoder`, `--model_family decoder_only`, and `--model_family encoder_only`. That means the fairest three-way comparison is to run each architecture through `scripts/run_sparsity_experiments.py` with the same benchmark, sparsity levels, pruning modes, seed, and normalization mode, changing only `--model_family` and `--model_checkpoint`.

Do not compare the encoder-decoder legacy prune/eval table directly against a decoder-only or encoder-only table unless those repos use the same legacy settings. The legacy path evaluates both training and benchmark rows and uses a simpler text normalization path; the linear sparsity matrix is the cleaner cross-architecture comparison surface.
