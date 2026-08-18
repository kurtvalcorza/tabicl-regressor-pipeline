# TabICLv2 Regression Dataset Specification

## Files

A dataset is a directory or zip containing exactly one `train.csv` and optionally one `val.csv` and one `test.csv`. Duplicate split candidates are rejected.

All present splits must contain the same columns before configured exclusions.

## Target

- Default target column: `target`
- The target must be numeric and finite.
- Missing, non-numeric, `NaN`, and infinite target values are excluded from the usable-row count and removed before training.
- At least 50 usable training rows are required.
- At least two distinct finite target values are required.
- Negative targets are valid. Predictions are never clipped to zero by the pipeline.

## Features

Every column except the target and configured `drop_columns` is passed to TabICL. The initial DIMER operational profile permits up to 2,000 features.

TabICL accepts pandas DataFrames and performs its own tabular preprocessing and ensemble feature shuffling.

## Splitting and sampling

If `val.csv` is absent, the finetuner creates a deterministic random holdout using `validation_split` and `seed`.

If training rows exceed `max_train_rows`, the finetuner deterministically samples to the configured cap. The DIMER field range is 100–50,000 rows. TabICL's fine-tuning data pipeline internally chunks meta-datasets with a default `max_data_size` of 10,000.

`test.csv`, when supplied, is not used for optimization or early stopping. It is scored only after fine-tuning.

For forecasting/time-series problems, callers should supply an explicitly time-separated `val.csv` / `test.csv`; the generic random fallback split is intended for ordinary IID tabular regression, not temporal evaluation.

## Reported metrics

Validation and optional test results include:

- MAE
- MSE
- RMSE
- R²

The fine-tuner's best-checkpoint selection may be driven by `mae`, `mse`, or `r2`.

## Archive safety

Default operational guards:

- maximum total uncompressed dataset archive: 1 GiB
- maximum single CSV: 512 MiB
- duplicate split filenames rejected
- nested zip files rejected by the validator
- path-traversal archive members rejected

These limits may be overridden by platform environment variables when a larger deployment profile is intentional.
