# TabICLv2 Regressor — DIMER Model Card

## Base model

- Model family: TabICLv2
- Checkpoint identifier: `tabicl-regressor-v2-20260212.ckpt`
- Task: tabular regression
- Upstream package pinned by this pipeline: `tabicl[finetune]==2.1.1`
- Upstream project license: BSD 3-Clause for TabICL's core tabular implementation

## Adaptation mode

This DIMER pipeline performs real downstream fine-tuning through `tabicl.FinetunedTabICLRegressor`. The upstream fine-tuning wrapper trains with pinball/quantile loss and supports validation-based best-checkpoint selection using MSE, MAE, or R².

Fine-tuning is GPU-only in this DIMER implementation. Lack of CUDA is a run failure rather than an automatic change to zero-shot inference.

## Artifact

A trained artifact contains both:

1. `checkpoints/best.ckpt` — fine-tuned TabICLv2 weights;
2. `training_context.parquet` — the support/context table used when reloading the in-context regressor.

The training job verifies that `best.ckpt` can be loaded by `TabICLRegressor`, fitted on the saved training context, and used for prediction before declaring success.

## Evaluation

Validation metrics: MAE, MSE, RMSE, R².

When `test.csv` is present, it is scored only after fine-tuning and best-checkpoint selection. The pipeline does not clip predictions or assume a non-negative target domain.

For temporal/forecasting datasets, use explicit time-separated validation and test splits; the automatic fallback holdout is a generic IID random split.

## Provenance

Each successful `result.json` records the base checkpoint identifier, TabICL version, uploaded dataset SHA-256, and fine-tuned checkpoint SHA-256.

## Production status

Repository implementation is not equivalent to DIMER production validation. Production enablement still requires an on-platform GPU build, fine-tuning smoke test, persistence of both checkpoint and support context, and end-to-end inference-serving verification.
