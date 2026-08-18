# Deployment and operations — TabICLv2 Regressor

## Repositories

DIMER connects to two deployable repositories:

- validator: `kurtvalcorza/tabicl-regressor-dataset-validator`
- fine-tuner: `kurtvalcorza/tabicl-regressor-finetuner`

The pipeline task identity is `tabular_regression`.

## Runtime requirements

The validator is CPU-only.

The fine-tuner intentionally requires a CUDA GPU. It does **not** silently change a fine-tuning request into zero-shot inference when CUDA is unavailable. A no-GPU run fails clearly in `result.json`.

The fine-tuner image is based on PyTorch 2.6 / CUDA 12.4 and installs `tabicl[finetune]==2.1.1`.

## Base checkpoint

The pipeline uses TabICLv2's regression checkpoint identifier:

`tabicl-regressor-v2-20260212.ckpt`

TabICL is distributed under the BSD 3-Clause license (the upstream repository contains a separate Apache-2.0 notice for its forecasting-derived code; this pipeline uses the tabular regressor, not the forecasting module).

## Fine-tuned artifact

A successful run writes:

```text
tabicl_regressor/
├── artifact.json
├── training_context.parquet
└── checkpoints/
    └── best.ckpt
```

The training context is part of the serving contract. TabICL remains an in-context learner after downstream fine-tuning, so inference reloads the fine-tuned checkpoint and then fits the inference wrapper on the saved support/context table before predicting new rows.

Conceptually:

```python
from tabicl import TabICLRegressor
import pandas as pd

ctx = pd.read_parquet("training_context.parquet")
reg = TabICLRegressor(model_path="checkpoints/best.ckpt", allow_auto_download=False)
reg.fit(ctx.drop(columns=[target_column]), ctx[target_column])
y_pred = reg.predict(X_new)
```

The training job performs a reload-and-predict smoke check against the saved `best.ckpt` before it reports success.

## Metrics and provenance

`result.json` records validation metrics and, when supplied, post-training `test.csv` metrics: MAE, MSE, RMSE and R².

It also records:

- dataset SHA-256
- TabICL package version
- base checkpoint identifier
- fine-tuned checkpoint SHA-256
- target column and training hyperparameters

## DIMER acceptance test

Before enabling for production, verify the complete platform path:

1. build validator image;
2. build GPU fine-tuner image;
3. validate a small regression dataset;
4. fine-tune and confirm `best.ckpt` plus context artifact are persisted;
5. deploy the artifact through DIMER's inference-serving layer;
6. send a regression request and verify the service reloads both checkpoint and support context;
7. compare served predictions against an offline reload of the same artifact.

The repository-level code cannot prove step 5–7; those are platform integration checks.

## Operational note

The current GitHub visibility of `tabicl-regressor-finetuner` should be **private** to match the requested repository policy. If it is still public, change **Settings → General → Danger Zone → Change repository visibility → Make private** before treating the repository set as complete.
