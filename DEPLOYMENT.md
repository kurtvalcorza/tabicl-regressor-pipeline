# Deployment and operations — TabICLv2 Regressor

## Repositories

DIMER connects to two deployable repositories:

- validator: `kurtvalcorza/tabicl-regressor-dataset-validator`
- fine-tuner: `kurtvalcorza/tabicl-regressor-finetuner`

The pipeline task identity is `tabular_regression`.

## Runtime requirements

The validator is CPU-only.

The fine-tuner intentionally requires a CUDA GPU. It does **not** silently change a fine-tuning request into zero-shot inference when CUDA is unavailable. A no-GPU run fails clearly in `result.json`.

The fine-tuner image is based on **PyTorch 2.8 / CUDA 12.8** (which ships sm_120/Blackwell kernels; the earlier PyTorch 2.6 / CUDA 12.4 image failed on RTX 50-series / B200 hardware) and installs `tabicl[finetune]==2.1.1`.

## Base checkpoint and handoff

The pipeline is **fixed to the pinned TabICLv2 regression checkpoint** `tabicl-regressor-v2-20260212.ckpt` at Hugging Face revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, baked into the fine-tuner image at that revision (so fine-tuning performs no runtime download) and SHA-256-verified at fine-tune time.

Resolution precedence and provenance are exact:

| `baseModelSource` | when | verification | `baseModelRevision` |
|---|---|---|---|
| `dimer-provided` | `DIMER_BASE_MODEL_PATH` is set (DIMER operator override) | **must exist** — a configured-but-missing path fails the run; used as-is, SHA-256 recorded | pinned revision only if the bytes match the pinned default, else `null` |
| `pinned-baked` | default — the checkpoint baked into the image (`TABICL_BAKED_BASE_MODEL`) | SHA-256 hard-verified against the pinned value | pinned revision |
| `pinned-download` | no baked copy present | downloaded at the pinned revision, SHA-256 hard-verified | pinned revision |

A DIMER-selected Base Model deterministically controls the checkpoint actually loaded (no silent fallback), and a custom base never falsely claims the pinned revision. `model_id` is intentionally not a `dimer-pipeline.json` hyperparameter.

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
- base checkpoint identifier, **HF revision, SHA-256, and source** (`dimer-provided` / `pinned-baked` / `pinned-download`)
- fine-tuned checkpoint SHA-256 and training-context SHA-256
- target column and training hyperparameters

`artifact.json` is the complete inference contract: target/feature-column ordering, the `inference` block (`nEstimators`, `randomState`, device, `allowAutoDownload`), the persisted `categoricalEncoders`, and component `digests`. A fresh-environment loader must reconstruct the exact scored model from `artifact.json` plus the referenced files alone.

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
