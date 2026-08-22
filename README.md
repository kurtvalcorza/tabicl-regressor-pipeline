# TabICLv2 Regressor — DIMER Pipeline

A DIMER pipeline that fine-tunes [TabICLv2](https://huggingface.co/jingang/TabICL), a pretrained
tabular foundation model, on your own tabular-regression dataset. You supply a table of rows with
one finite numeric target column. The pipeline validates the table, fine-tunes TabICLv2 on a GPU,
and produces a saved model artifact plus holdout error metrics.

TabICL is [BSD-3-Clause](https://opensource.org/license/bsd-3-clause) licensed, so the trained
pipeline can be enabled and served without a usage restriction on the model itself. The training
data carries its own licence — see [Data licence governs the served model](#data-licence-governs-the-served-model).

For platform-administrator setup and operations — resource profiles, weights delivery, network
egress, base-model handoff, and the acceptance test — see [DEPLOYMENT.md](DEPLOYMENT.md). Model
provenance and adaptation details are in [MODEL_CARD.md](MODEL_CARD.md).

---

## The model: TabICLv2

TabICL is a tabular foundation model from the [Soda team at Inria](https://team.inria.fr/soda/),
released with open weights under BSD-3-Clause. Like TabPFN and Mitra, it is an **in-context
learner**: it reads a table of labelled examples as context and predicts on new rows. TabICLv2
(arXiv [2602.11139](https://arxiv.org/abs/2602.11139)) is the successor to the original TabICL
(ICML 2025, arXiv [2502.05564](https://arxiv.org/abs/2502.05564)), designed to be faster and to
scale to larger tables.

For regression, this pipeline fine-tunes through `tabicl.FinetunedTabICLRegressor`, whose
upstream wrapper trains with a **pinball / quantile loss** and supports validation-based
best-checkpoint selection on MSE, MAE, or R². The pipeline does not clip predictions or assume a
non-negative target domain — negative targets are valid. See [MODEL_CARD.md](MODEL_CARD.md) for
adaptation mode, and the [model card](weights/README.md) once the weights are staged locally (or
the upstream [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) repository) for provenance.

### Checkpoints

TabICLv2 ships classifier and regressor checkpoints in a single Hugging Face repository. This
pipeline uses the regressor.

| Checkpoint | Target type | Hugging Face id | File |
|---|---|---|---|
| Regressor (this pipeline) | numeric | [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) | `tabicl-regressor-v2-20260212.ckpt` |
| Classifier | categorical | [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) | `tabicl-classifier-v2-20260212.ckpt` |

The regressor checkpoint is pinned by Hugging Face **revision**
`4dcd344ece2c00be9e831fdd35bed57b5ad83e19` and hard-verified by **SHA-256**
`0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a` at fine-tune time, so every
run starts from identical weights (see [DEPLOYMENT.md → Base checkpoint and handoff](DEPLOYMENT.md)).

### Fine-tuning is GPU-only

This component is intentionally a **fine-tuner**, not a zero-shot server. It adapts the
pretrained weights to the uploaded table through `FinetunedTabICLRegressor`, which requires CUDA.
**If CUDA is unavailable the run fails** clearly in `result.json` rather than silently
downgrading a fine-tuning request into zero-shot inference. Because TabICL remains an in-context
model *after* fine-tuning, the served artifact carries both the fine-tuned checkpoint and the
training context — see [Outputs](#outputs).

---

## When to use this pipeline

Use this pipeline for tabular regression: predicting a continuous numeric value from a row of
features. Demand quantities, prices, durations, scores, and any row-per-record numeric prediction
fit here. For a categorical target, use the
[TabICLv2 classifier pipeline](https://github.com/kurtvalcorza/tabicl-classifier-pipeline). Do
not use it for images.

For temporal / forecasting datasets, supply explicit **time-separated** `val.csv` and `test.csv`
splits; the automatic fallback holdout is a generic IID random split, not a temporal one.

---

## Repositories

The pipeline is two deployable containers, one repository each, plus this umbrella repository for
the contract and docs. Each `Dockerfile` sits at its repository root.

| Component | Repository | Runs on |
|---|---|---|
| Validator | [`tabicl-regressor-dataset-validator`](https://github.com/kurtvalcorza/tabicl-regressor-dataset-validator) | CPU |
| Fine-tuner | [`tabicl-regressor-finetuner`](https://github.com/kurtvalcorza/tabicl-regressor-finetuner) | CUDA GPU |
| Umbrella (this repo) | `tabicl-regressor-pipeline` | Docs + contracts |

```
tabicl-regressor-dataset-validator/     (CPU)
├── Dockerfile
├── validate.py          DIMER-facing entrypoint (delegates to validator.py)
├── validator.py         validation implementation
├── requirements.txt
└── tests/

tabicl-regressor-finetuner/             (GPU)
├── Dockerfile           torch 2.8 / CUDA 12.8; bakes the pinned checkpoint
├── train.py             DIMER-facing entrypoint
├── dimer-pipeline.json  preprocessing + fine-tuning fields
├── requirements.txt
└── tests/
```

DIMER builds each repository from its root and launches the container by the portal naming
convention: `validate.py` for the validator and `train.py` for the fine-tuner. The validator's
tested logic lives in `validator.py`; `validate.py` is a thin entrypoint that delegates to it.

Keep `dimer-pipeline.json` at the fine-tuner repository root. It defines the preprocessing and
fine-tuning fields end users see. Without it, the workbench preprocessing step renders empty and
the fine-tuning step stays locked.

The authoritative dataset contract ([`TABULAR_REGRESSION_DATASET_SPEC.md`](TABULAR_REGRESSION_DATASET_SPEC.md)),
model card ([`MODEL_CARD.md`](MODEL_CARD.md)), and deployment notes ([`DEPLOYMENT.md`](DEPLOYMENT.md))
live in this umbrella repository, not in the container repositories.

---

## Creating the pipeline

Prerequisites: portal access as AI Engineer, and both repositories reachable by the portal's
GitHub App.

1. Open **AI Engineer → New Pipeline** and set these fields:

   | Field | Value |
   |---|---|
   | Pipeline Name | `TabICLv2 Tabular Regression` |
   | Description | `Fine-tune the TabICLv2 tabular foundation model for regression using your own tabular dataset. Supports dataset validation, configurable preprocessing, MAE/MSE/RMSE/R² evaluation, and export of the trained model.` |
   | Task Type | `Custom / Other` |
   | Base Model | `jingang/TabICL` (regressor checkpoint) |
   | Validator repository | `https://github.com/kurtvalcorza/tabicl-regressor-dataset-validator` |
   | Fine-tuner repository | `https://github.com/kurtvalcorza/tabicl-regressor-finetuner` |

2. Build both images (the fine-tuner build needs Hugging Face egress to bake the checkpoint — see
   [DEPLOYMENT.md → Network / Base checkpoint](DEPLOYMENT.md)).
3. Validate and fine-tune a small regression dataset.
4. Enable the pipeline **only after** the on-platform serving check in the acceptance test passes.

### Portal implementation notes

- **`Custom / Other` is the correct portal card** for tabular pipelines; DIMER has no native
  tabular task type. The fine-tuner image sets `DIMER_TASK_TYPE=tabular_regression` as its baked
  fallback and treats any value the platform sends as an override.
- **`dimer-pipeline.json` stays at the fine-tuner repository root.** The portal reads it there to
  render the preprocessing and fine-tuning fields.
- **`model_id` is not declared in `dimer-pipeline.json`.** The DIMER **Base Model** field is
  authoritative for the checkpoint that loads.

---

## The dataset

### Format

A zip of CSV files. The full contract is in
[`TABULAR_REGRESSION_DATASET_SPEC.md`](TABULAR_REGRESSION_DATASET_SPEC.md).

```
dataset.zip
├── train.csv          (required)   one row per example; one finite numeric target column
├── val.csv            (optional)   same columns as train; a random holdout is split if absent
└── test.csv           (optional)   scored after fine-tuning, never used for selection
```

The target column is named `target` by default; change it with the `target_column` preprocessing
field. Every other column, except those listed in `drop_columns`, is a feature; features may be
numeric or categorical. **Missing, non-numeric, `NaN`, and infinite target values are excluded**
from the usable-row count and removed before training. Duplicate split candidates in the archive
are rejected so the validator and fine-tuner cannot resolve different files.

### How to build a dataset

TabICL consumes a feature table, not raw records. Convert a time series or transaction log
(`entity, date, value`) into a training table by engineering one row per `(entity, date)`:

- **features** — history and context at that point: lags, rolling means and standard deviations,
  calendar fields, and any known covariates (promotions, holidays, weather, stock status);
- **target** — the numeric value to predict, e.g. the quantity a chosen number of days ahead.

For forecasting problems, use time-separated `val.csv` / `test.csv` rather than relying on the
generic random fallback holdout.

### Row and feature ceilings

The initial DIMER profile caps training at **50,000 rows** (`max_train_rows` range 100–50,000)
and **2,000 features**. At least **50 usable training rows** and **2 distinct finite target
values** are required. If `max_train_rows` is exceeded the fine-tuner deterministically samples
down to the cap; TabICL's own fine-tuning pipeline additionally chunks meta-datasets with a
default `max_data_size` of 10,000. Negative targets are valid and predictions are never clipped.
Archive guards: ≤1 GiB uncompressed total, ≤512 MiB per CSV, nested zips and path-traversal
members rejected. These limits are overridable by platform environment variables for an
intentionally larger profile.

### Data licence governs the served model

The model is BSD-3-Clause, but a served pipeline is also bound by the licence of the data it was
trained on. A model fine-tuned on non-commercial data — for example CC BY-NC — may not be
appropriate to expose as a hosted service. Confirm the licence of any corpus before you enable a
pipeline built from it.

---

## Configurable fields

Preprocessing (`datasetPreprocessing`):

| Field | Default | Purpose |
|---|---|---|
| `target_column` | `target` | Name of the finite numeric column to predict |
| `drop_columns` | — | Comma-separated columns to exclude from features (ids, raw dates) |
| `max_train_rows` | `10000` | Cap on training rows (range 100–50,000); larger tables are sampled to it |
| `validation_split` | `0.2` | Holdout fraction when the zip has no `val.csv` (range 0.05–0.4) |

Fine-tuning (`modelFinetuning`):

| Field | Default | Purpose |
|---|---|---|
| `epochs` | `30` | Maximum fine-tuning epochs; early stopping may stop sooner (range 1–100) |
| `learning_rate` | `1e-5` | AdamW learning rate (range 1e-7–1e-3) |
| `weight_decay` | `0.01` | AdamW weight decay (range 0–0.2) |
| `patience` | `8` | Non-improving epochs tolerated before early stopping (range 1–30) |
| `time_limit_seconds` | `1800` | Wall-clock fine-tuning budget (range 60–14,400) |
| `seed` | `0` | RNG seed for splitting, sampling, and TabICL; pin it for reproducible runs |
| `eval_metric` | `mae` | Validation metric for best-checkpoint selection (`mae`, `mse`, `r2`) |
| `n_estimators_finetune` | `2` | Ensemble members per fine-tuning meta-batch (range 1–8) |
| `n_estimators_validation` | `2` | Ensemble members for end-of-epoch validation (range 1–8) |
| `n_estimators_inference` | `8` | Ensemble members used by the final fitted regressor (range 1–32) |

---

## Outputs

A successful run writes the trained artifact and a `result.json` describing the run:

```
tabicl_regressor/
├── artifact.json            complete inference contract (schema, inference block, encoders, digests)
├── training_context.parquet the in-context support table used at inference time
└── checkpoints/
    └── best.ckpt            the fine-tuned TabICLv2 checkpoint
```

`result.json` records validation and optional post-training `test.csv` metrics — **MAE, MSE,
RMSE, R²** — plus the input dataset SHA-256, the `tabicl` package version, the base checkpoint
identifier with its HF revision / SHA-256 / source (`dimer-provided` / `pinned-baked` /
`pinned-download`), the fine-tuned checkpoint SHA-256, the training-context SHA-256, the target
column, and the training hyperparameters.

TabICL remains an in-context model after downstream fine-tuning, so **inference needs both the
fine-tuned checkpoint and the training context**. A fresh-environment loader reconstructs the
exact scored model from `artifact.json` plus its referenced files, then fits and predicts:

```python
from tabicl import TabICLRegressor
import pandas as pd

ctx = pd.read_parquet("training_context.parquet")
reg = TabICLRegressor(model_path="checkpoints/best.ckpt", allow_auto_download=False)
reg.fit(ctx.drop(columns=[target_column]), ctx[target_column])
y_pred = reg.predict(X_new)
```

The fine-tuner performs this reload-and-predict smoke check against `best.ckpt` before it reports
success. When the dataset zip includes a `test.csv`, it is scored after fitting and does **not**
influence checkpoint selection.

---

## Reproducibility

TabICL fine-tuning is stochastic: two runs on identical data can differ unless the seed is fixed.
`seed` is a first-class field and seeds the split, the sampling, and TabICL itself. GPU kernel
autotuning can still leave small residual variation, so runs are reproducible in ranking but not
guaranteed byte-identical. Starting from identical weights is guaranteed by the pinned revision +
SHA-256 baked into the image.

---

## Resource profile

Each fine-tuning run executes as a Kubernetes job under a **GPU** profile — there is no CPU
fallback. Start with a single modern CUDA GPU and the conservative default limits, then profile
representative workloads before increasing `max_train_rows`, the estimator counts, or `epochs`.
TabICL holds the training table in memory as in-context context, so its footprint grows with rows
and features; set the production GPU/RAM profile from measured peak usage during the acceptance
test (see [DEPLOYMENT.md → DIMER acceptance test](DEPLOYMENT.md)).

The fine-tuner image is built on **`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`**, which ships
sm_120 (Blackwell) kernels alongside sm_70–sm_100. Do not downgrade to cuda12.4/torch2.6 — that
build lacks sm_120 and dies with *"no kernel image is available"* on RTX 50-series / B200 GPUs.

---

## Provenance and traceability

### How this pipeline was authored

The validator, fine-tuner, configuration, and documentation in this repository were drafted with
AI assistance (Anthropic Claude, via Claude Code) and are pending human review before production
deployment. The following were verified by execution, not only generated: both container scripts
byte-compile; `dimer-pipeline.json` validates against the field schema; the validator's unit-test
suite covers its check set (numeric-target and usable-row thresholds, duplicate-split and
archive-safety rejection, the wrong-pipeline guidance, and the crash-path callback/metadata
contract); and the base weights' SHA-256 is verified before fitting.

Not yet verified, and requiring human sign-off: the DIMER portal image build, the on-platform GPU
smoke test, the resource-profile request, and the platform's **inference-serving integration**
for this custom TabICL artifact format. Treat the generated code as a reviewed draft, not audited
production code, and do not production-enable until the end-to-end serving check in
[DEPLOYMENT.md](DEPLOYMENT.md) passes on-platform.

### Model lineage

| Field | Value |
|---|---|
| Base model | [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) (regressor checkpoint) |
| Checkpoint file | `tabicl-regressor-v2-20260212.ckpt` |
| Pinned weights revision | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| Weights SHA-256 | `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a` |
| Licence | BSD-3-Clause |
| Origin | [Qu et al. (2025, ICML)](https://arxiv.org/abs/2502.05564); [TabICLv2 (2026)](https://arxiv.org/abs/2602.11139); Soda team, Inria |
| Framework | `tabicl[finetune]==2.1.1` |

Pinning the revision (fine-tuner `Dockerfile`) makes every run start from identical weights.
Without it, `tabicl` could fetch a moved `main` at runtime and the model could change between
builds.

### Data lineage

A trained model inherits the provenance and licence of the table it was fine-tuned on. Each
dataset should carry its source, its licence, and — for a derived table — the transformation that
produced it. Because the served artifact embeds the training context, treat it with the same
data-governance controls as the source training dataset.

### Per-run record

Every fine-tuning run writes a `result.json` that serves as the run's provenance record: the base
model and its resolved revision/SHA-256/source, the target and dropped columns, the seed, time
budget, and eval metric, the training device, row counts, the resulting error metrics, the
fine-tuned checkpoint SHA-256, the training-context SHA-256, and the input dataset digest. Paired
with the container image tag, this forms a chain from data to served model.

---

## References

- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025).
  [*TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*](https://arxiv.org/abs/2502.05564).
  International Conference on Machine Learning (ICML) 2025.
- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026).
  [*TabICLv2: A better, faster, scalable, and open tabular foundation model*](https://arxiv.org/abs/2602.11139).
- [TabICL source code](https://github.com/soda-inria/tabicl), Soda team, Inria (BSD-3-Clause).
- [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) model card and checkpoints, Hugging Face.
