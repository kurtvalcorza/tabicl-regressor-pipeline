---
license: bsd-3-clause
model_card_spec: "1.0"
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - tabular-foundation-model
  - in-context-learning
  - quantile-regression
  - tabicl
base_model: jingang/TabICL
---

# TabICLv2 Regressor

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jingang%2FTabICL-ffcc4d?style=flat)](https://huggingface.co/jingang/TabICL)
[![GitHub](https://img.shields.io/badge/GitHub-soda--inria%2Ftabicl-181717?style=flat&logo=github&logoColor=white)](https://github.com/soda-inria/tabicl)
[![arXiv](https://img.shields.io/badge/arXiv-2602.11139-b31b1b.svg)](https://arxiv.org/abs/2602.11139)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


###### Description

TabICLv2 Regressor packages the `tabicl-regressor-v2-20260212.ckpt` checkpoint from `jingang/TabICL` at Hugging Face revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, a pretrained tabular foundation model developed by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan of the Soda team at Inria, run through the `tabicl==2.1.1` reference implementation. The model is a three-stage Transformer for tables — a column-wise encoder that embeds each feature distribution, a row-wise encoder that builds one representation per observation, and a dataset-wise in-context-learning Transformer that attends from the labelled support rows to the query rows. Its regression head predicts 999 target quantiles (α = 0.001 … 0.999) trained with pinball loss, and the reference implementation averages them into the point estimate that `predict()` returns.

At inference the model conditions on the labelled training table as in-context support and emits one continuous estimate per query row. Adaptation happens through in-context conditioning by default and, in this pipeline, through gradient fine-tuning on the operator's table (`tabicl-regressor-finetuner/train.py`, `early_stopping=True`, learning rate 1e-5 by default). What this repository adds is the DIMER composition around those weights: the pipeline contract (`dimer-pipeline.json`, `DIMER_CONTRACT.md`), the dataset specification, the Colab artifact-inference tutorial, and the release conformance record; the validator and fine-tuner workers it composes live in the sibling `tabicl-regressor-dataset-validator` and `tabicl-regressor-finetuner` repositories. The upstream checkpoint is not modified by this repository, and the served contract exposes the point estimate only — the underlying quantiles are not surfaced.

#### Intended Use and Limitations

The use cases below are the ones envisioned during development; the limits are the ones the workers enforce.

###### Primary Intended Uses

Supervised prediction of a continuous numeric target from tabular features, where each observation is one row of numerical and categorical predictor columns. The pipeline takes a `train.csv` (optionally `val.csv`/`test.csv`) with a declared numeric target column and produces a fine-tuned TabICLv2 artifact, one point prediction per row, and holdout error metrics.

Concrete application domains envisioned during development: demand and quantity estimation, price and cost estimation, continuous risk or score prediction, scientific and engineering regression represented as feature tables, and resource-use estimation formulated as row-wise tabular prediction, where the operator wants strong performance with little or no task-specific hyperparameter search. The pipeline is meant to play the role of a strong zero-configuration baseline or a fine-tuned production model inside DIMER. Enforced ceilings: at most 2,000 feature columns (`MAX_FEATURES`) and a training cap of `max_train_rows` between 300 and 50,000 (default 10,000). Pretraining covered roughly 48,000 rows and 100 features, so larger tables are extrapolation the upstream authors report works but this pipeline does not separately validate.

###### Primary Intended Users

Machine-learning researchers, data scientists, machine-learning engineers, software developers, and scientific researchers building predictive systems from structured datasets. The envisioned deployment setting is internal enterprise or research use through the DIMER platform, where the fine-tuner runs as a CUDA worker and the validator as a CPU worker — not a public-facing service.

The pipeline assumes its users understand dataset provenance, holdout evaluation, leakage, and distribution shift, and know that the served prediction is a quantile-averaged point estimate with no attached interval, that MAE on a few hundred holdout rows has wide variance, that a sparse or heavy-tailed target needs a naive baseline for comparison, and that fine-tuning needs a CUDA GPU and will fail without one rather than fall back. A user who cannot tell a held-out error from an in-sample one is outside the assumed competency.

###### Out-of-scope use cases

- **Capability boundaries:** categorical classification (the sibling `tabicl-classifier-pipeline` does that); interval, quantile, or distributional output — the checkpoint computes quantiles internally but this pipeline's served contract returns the point estimate only; image, audio, video, or natural-language inputs; unsupervised clustering; causal-effect estimation; generative modelling; raw time-series forecasting without tabular feature construction.
- **Input boundaries:** a target column that is not numeric or not finite (validator checks `target_is_numeric`, `target_is_finite` refuse); a constant target (`target_has_variation` refuses); more than 2,000 feature columns (refused); fewer than 50 usable training rows (`MIN_TRAIN_ROWS`) or fewer than 10 evaluation rows (`MIN_EVAL_ROWS`); archives over 1 GiB uncompressed (refused); fine-tuning on a host without CUDA (refused with a clear error).
- **Decision boundaries:** autonomous high-impact decisions — pricing of credit or insurance, clinical dosing, safety margins — without application-specific validation and a human decision-maker; treating upstream benchmark rankings as a guarantee on a new dataset.

#### Factors

TabICLv2's behaviour varies with the structure of the table it is given, not with a physical capture condition; the three subsections below say what that means for groups, instruments, and environment.

###### Groups

This pipeline is not human-centric by construction: TabICLv2 was pretrained on synthetic tasks rather than any fixed human population, so no demographic group is an intrinsic evaluation group of the foundation model, and the pretraining corpus is not group-audited because it contains no people. Demographic fairness or subgroup error parity has therefore **not** been established for the checkpoint, and the pipeline measures no subgroup metric.

Where the operator's downstream table describes people, the obligation transfers to the operator: identify the relevant groups in their own data, compute per-group MAE and RMSE on the holdout split, and check for disparate error before deployment — a regressor that is unbiased on average can still be systematically high for one group and low for another. The validator result records row counts and target statistics, not any demographic structure.

###### Instrumentation

TabICLv2 consumes an abstract tabular representation rather than a raw sensor stream; the upstream pretraining did not depend on any real acquisition hardware. The instrument does not disappear because a table sits between it and the model: the operator's rows are produced by whatever systems fed the CSV — transactional databases, ETL pipelines, meters, sensors, survey instruments — and their sampling rate, resolution, calibration, units, and encoding of missing values determine both feature quality and the scale of the target.

Instrument error reaches the model as feature or target error. Drift, miscalibration, a unit change, or a changed collection procedure between training and inference is not detectable by this pipeline; the validator checks that the target is numeric, finite, and non-constant and that the schema is consistent, not whether a column's meaning or unit has changed. Operators should document the instrumentation of downstream datasets separately.

###### Environment

**Operating environment.** Fine-tuning is GPU-only: the fine-tuner raises if CUDA is unavailable — there is no CPU fine-tune path. Inference on the saved artifact runs on CPU or GPU through `tabicl==2.1.1`, batched at `PREDICT_BATCH_ROWS` (default 8,192) to bound memory. The validator is CPU-only. Precision follows the reference implementation's defaults.

**Data environment.** The reported error assumes the inference rows are drawn from the same distribution as the training table: same feature semantics, same units, same target range. Performance degrades, without warning from the pipeline, under geographic, temporal, institutional, or population shift, with heavy-tailed, sparse, or intermittent targets, and with feature count beyond the roughly 100 features and 48,000 rows of the pretraining regime. Predictions outside the training target range are extrapolations the model has no basis for. Robustness to arbitrary distribution shift has not been established.

#### Metrics

Metrics are chosen for a point-estimate regressor whose intended use spans targets of very different scales and tail behaviour.

###### Performance Measures

The fine-tuner scores the fine-tuned model on the held-out split (`tabicl-regressor-finetuner/train.py`) and writes `mae`, `rmse`, and `r2` into the result artifact, computed with scikit-learn on the raw point predictions. The headline metric is the DIMER hyperparameter `eval_metric` (default `mae`).

Why these: MAE is robust to outliers and interpretable in target units, which makes it the right default for a domain-agnostic pipeline whose targets range from counts to prices; RMSE weights large errors quadratically and is the informative one when a few big misses matter more than many small ones; R² normalises against the variance of the holdout target so that tables of different scale can be compared, at the cost of misleading on low-variance targets. Reading only MAE hides tail failures, reading only RMSE lets one outlier dominate, and reading only R² hides absolute error, which is why all three are written. Pinball loss and interval coverage — the natural metrics for the quantile head — are not reported because the served contract does not expose the quantiles. Upstream reports strong regression performance across TabArena and TALENT; that is a published relative ranking, not a number this pipeline measures or claims.

###### Decision thresholds

The pipeline applies no decision threshold and no clipping: `predict()` returns the quantile-averaged point estimate and the served artifact returns it unchanged, so the reported error reflects what a caller will actually receive. No acceptance threshold on MAE, RMSE, or R² was set during development, because the pipeline is domain-agnostic and the tolerable error is a property of the deployment.

Any cutoff that turns a prediction into an action — a reorder point, a price band, a tolerance margin — is the deployment owner's to define and to calibrate on their own held-out residuals. Set it from the asymmetric cost of over- versus under-prediction: where an under-estimate is the expensive error, place the operating value above the point prediction by a margin derived from the holdout residual distribution, and revisit it whenever the input distribution or the target scale shifts.

###### Approaches to uncertainty and variability

The pipeline's reported metrics come from a single holdout split of the operator's table (`validation_split`, default 0.2, used when `val.csv` is absent). No dispersion is reported alongside the point value: one split, one fine-tuning run, no confidence interval. Operators who need one should repeat the run across seeds or use cross-validation on their own side.

Sources of run-to-run variability: the holdout split, the training cap, TabICL's internal feature-permutation ensembling (`n_estimators_finetune`, `n_estimators_validation`, `n_estimators_inference`, defaults 2/2/8), and gradient fine-tuning with early stopping; all are driven by the DIMER `seed` hyperparameter, which the fine-tuner passes to the split, the cap, and TabICL's `random_state`. Non-deterministic CUDA kernels can still produce small run-to-run differences. The pipeline emits no confidence output: although the checkpoint predicts 999 quantiles internally, the served contract returns only their average, so no interval is available to calibrate. A caller who needs an interval must estimate one from their own holdout residuals and should treat it as valid only within the training target range; the upstream card is explicit that even the internal quantiles are not guaranteed calibrated for every downstream population.

#### Ethical considerations and biases

No external ethics board reviewed this pipeline, and no clearance testing with a specific group took place; the subsections record what the developers considered and what the repository actually does.

###### Data

TabICLv2 was pretrained exclusively on synthetic tasks, so the pretraining data do not consist of personally identifiable, health, biometric, financial, or classified records — known from the upstream disclosure, which ends at the description of the synthetic prior; the generated tables are not published. What this repository distributes: the pipeline contract, dataset specification, the Colab tutorial, and conformance records; it does **not** distribute the checkpoint (the fine-tuner downloads it at the pinned revision and verifies its SHA-256 before use, and `weights/` is gitignored) and ships no sample data of its own.

Operators may fine-tune or evaluate TabICLv2 on sensitive real-world tables. The pipeline does not audit the operator's data for personal, sensitive, or proprietary attributes — the validator checks structure and target type, not content — so the legality, privacy, consent, access control, and governance of downstream data remain with the application developer and data owner.

###### Human Life

The pipeline is not intended for decisions in health care, physical safety, criminal justice, legal rights, employment, credit, insurance, education access, or public benefits, and it has not been validated for any of them. The only validation performed is the contract and notebook conformance testing in `scripts/` and the DIMER holdout scoring on the operator's own table; no clinical, regulatory, or independent domain validation has been carried out by the developers or by any external body, and upstream benchmark rankings are not evidence of suitability.

Where such a use is foreseeable — a dosing or exposure regressor on a clinical feature table, or a credit-limit regressor — it would be admissible only with independent domain validation on that operator's population, a human decision-maker between the prediction and the action, subgroup error evaluation, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented in the composed workers, each inspectable in the named code:

- **Supply-chain integrity:** the base checkpoint is downloaded with `hf_hub_download(..., revision=BASE_MODEL_REVISION)` at `4dcd344e…`, its SHA-256 is computed and compared with `BASE_MODEL_SHA256` (`0db9cb53…`), and a mismatch raises unless the checkpoint was DIMER-provided, in which case the digest and pin status are recorded in provenance rather than enforced; `tabicl` is pinned to 2.1.1.
- **Input integrity:** the validator rejects archives over 1 GiB uncompressed, fewer than 50 usable training or 10 evaluation rows, more than 2,000 features, and a target that is non-numeric, non-finite, or constant, with wrong-pipeline guidance when the target looks categorical; the fine-tuner re-applies the row and feature limits.
- **Statistical mitigations:** prediction is batched at `PREDICT_BATCH_ROWS` to bound memory, and predictions are scored raw so that the reported error is the error a caller will see, not a clipped one.
- **Reproducibility:** `seed` drives the split, the cap, and TabICL's `random_state`; the result artifact records the checkpoint digest, revision, source, dataset digest, and the effective hyperparameters; the saved artifact is reloaded and smoke-predicted before the run is reported successful.
- **Refusals:** fine-tuning refuses any non-CUDA device rather than silently training on CPU; the served contract does not expose the internal quantiles, so no uncalibrated interval can be mistaken for a calibrated one.

###### Risks and harms

- **Extrapolation outside the training range** (model-intrinsic): a query row beyond the support of the training table receives a point estimate with no signal of its own unreliability; borne by whoever the operator's decision affects; likely under normal use as the deployment drifts; magnitude set by what the estimate gates.
- **Amplification of input bias** (model-intrinsic): a table whose target encodes a historical disparity — past pay, past pricing — yields a regressor that reproduces it; borne by the data subjects in the disadvantaged group; realised whenever such a table is used without subgroup error evaluation.
- **Tail and sparsity failure** (model-intrinsic): heavy-tailed, intermittent, or zero-inflated targets can produce low MAE and large individual misses; borne by the operator who reads MAE alone.
- **Small-sample variance** (model-intrinsic): a holdout MAE on a few hundred rows can move materially between seeds; borne by the operator who ships on one split.
- **Automation bias** (use-context): a numerically precise estimate displaces human judgement; borne by the data subject; likely wherever the number is surfaced without its error.
- **Undetected leakage** (use-context): a feature derived from the target collapses the holdout error and fails in production; the validator does not detect it; borne by the operator and downstream users.
- **Benchmark over-generalisation** (use-context): reading upstream relative rankings as an expected error level on a new table; borne by whoever sets expectations from them.

###### Use cases

Distinct from the capability and decision boundaries listed under *Out-of-scope use cases*, the developers consider the following uses prohibited even where the model would produce a numerically plausible estimate:

- surveillance, biometric or demographic profiling, or social scoring of individuals;
- unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access, including regression on a target that proxies a protected attribute (pay, premium, or limit set by group membership);
- deceptive, manipulative, or predatory applications, including exploitative price discrimination and presenting a point estimate as a certified measurement;
- clinical dosing, safety-margin, or legal-rights determinations without the validation and oversight described under *Human Life*;
- any use that violates the BSD-3-Clause terms of the upstream `jingang/TabICL` checkpoint and `tabicl` code, or the terms of the DIMER deployment.

---

## Model Details

- **Model name:** TabICLv2 Regressor
- **Model family:** TabICLv2
- **Checkpoint identifier:** `tabicl-regressor-v2-20260212.ckpt`
- **Code repository:** [soda-inria/tabicl](https://github.com/soda-inria/tabicl)
- **Hugging Face repository:** [jingang/TabICL](https://huggingface.co/jingang/TabICL)
- **Developers:** Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan; Soda team, Inria
- **Task:** tabular regression
- **Learning paradigm:** in-context learning / quantile regression
- **Checkpoint version date:** 12 February 2026
- **Associated paper first posted:** 11 February 2026
- **Reference implementation:** `tabicl`
- **Known-compatible library version used by this repository:** `tabicl[finetune]==2.1.1`
- **License:** BSD 3-Clause

## Checkpoint Provenance

This card documents the following released TabICLv2 regression checkpoint.

- **Checkpoint:** `tabicl-regressor-v2-20260212.ckpt`
- **Hugging Face repository:** [`jingang/TabICL`](https://huggingface.co/jingang/TabICL)
- **Pinned Hugging Face revision:** `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- **Direct download URL:** [`https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt`](https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt)
- **SHA-256:** `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`

The checkpoint revision and SHA-256 should be retained for strict model-version provenance. The regression and classification checkpoints are separately pretrained artifacts and are not interchangeable.

## Input

The regressor consumes a supervised tabular problem consisting conceptually of:

- labelled support or training observations;
- numerical and/or categorical feature columns;
- a finite continuous target for the support observations; and
- unseen feature rows requiring prediction.

TabICL uses the labelled support dataset as inference context. Calling `fit()` in the standard in-context-learning estimator primarily establishes this context. The pretrained model performs task-specific learning during prediction.

The model does not require the target to be non-negative. The numerical range and physical meaning of the target are downstream-task properties.

## Output

TabICLv2 Regressor internally predicts **999 target quantiles** corresponding to probability levels from approximately `0.001` to `0.999` at increments of `0.001`.

The reference model uses these quantiles to produce a continuous point estimate and quantile or distributional information where requested.

For standard point regression, the published approach averages the predicted quantiles. For probabilistic prediction, the quantile outputs can be transformed into a monotonic predictive distribution and used to derive quantities such as quantiles, probability density, cumulative distribution, moments, and uncertainty intervals.

## Architecture

TabICLv2 Regressor shares the principal three-stage architecture of the classification model:

1. **Column-wise embedding**
2. **Row-wise interaction**
3. **Dataset-wise in-context learning**

Its runtime complexity for a table containing `n` rows and `m` features is approximately `O(n² + n m²)`.

### Repeated Feature Grouping

Features are embedded using repeated circular groupings rather than entirely isolated column tokens. The published grouping pattern is `(0, 1, 3)`. This gives the model multiple local views of feature relationships while preserving the effective feature count.

### Target-Aware Embedding

Continuous target values from labelled support rows are projected through learned linear layers and injected into the early representation stages. Unlike the classifier, which uses discrete lookup embeddings, the regression checkpoint uses **linear target embeddings**.

### Column-Wise Transformer

- 3 induced self-attention blocks
- 128 inducing vectors
- model dimension: 128
- 8 attention heads

### Row-Wise Transformer

- 3 Transformer layers
- model dimension: 128
- 8 attention heads
- 4 learnable `[CLS]` tokens

The `[CLS]` representations aggregate feature-wise information into a row-level representation.

### Dataset-Wise ICL Transformer

- 12 Transformer layers
- model dimension: 512
- 8 attention heads

The labelled support rows provide contextual information from which predictions for new rows are generated.

### Prediction Head

The regression prediction head is a two-layer MLP with hidden dimension 1024 and output dimension **999**. Each output corresponds to one target quantile.

### Other Architectural Characteristics

The regression checkpoint uses bias-free pre-norm LayerNorm with learnable weights, GELU activations, 2× feed-forward expansion, rotary positional embeddings in the row Transformer, Query-Aware Scalable Softmax (QASSMax), standard residual initialization, and linear target embeddings for continuous labels.

The use of bias-free LayerNorm is an explicit architectural difference from the classification checkpoint.

## Quantile Regression

TabICLv2's regression formulation differs from foundation models that predict only a point estimate or discretize the target into categorical bins.

It predicts 999 quantiles:

```text
α = 0.001, 0.002, ..., 0.999
```

Training minimizes **pinball loss** across the quantiles.

The authors report that this approach performed better in their preliminary RMSE comparisons than both direct mean-squared-error prediction and bin-based regression approaches.

At inference time, the model can average quantiles to obtain a point estimate or use the full quantile set to construct a predictive distribution. This makes the checkpoint suitable for both point and distribution-aware regression, subject to downstream calibration and validation.

## Pretraining Data

TabICLv2 Regressor was pretrained entirely on **synthetically generated tabular regression problems**.

The synthetic prior generates diverse graph-structured relationships using mechanisms such as structural causal graphs, random neural functions, tree-based functions, random discretization, nonlinear transformations, correlated variables, feature interactions, and varying graph structures.

Approximately **35 million synthetic datasets** are processed across the complete three-stage curriculum.

For regression pretraining, the target remains continuous; numerical variables and targets undergo outlier handling and standard scaling within the prior-generation process; and filtering is used to remove a subset of synthetic datasets judged effectively unpredictable.

The use of fully synthetic pretraining data avoids directly training the foundation model on a fixed corpus of sensitive real-world tables.

## Pretraining Procedure

The regressor uses the same three-stage curriculum scale as the classifier but is trained as a separate model.

### Stage 1

- 500,000 steps
- 1,024 samples per synthetic dataset
- approximately 30–90% training context
- maximum learning rate: `8e-4`

### Stage 2

- 40,000 steps
- 400–10,240 samples per dataset
- approximately 80% training context
- maximum learning rate: `1e-4`

### Stage 3

- 10,000 steps
- 400–60,000 samples per dataset
- approximately 80% training context
- maximum learning rate: `2e-5`

Common settings include batch size 64, up to 100 features, Muon optimizer, cosine learning-rate schedule, gradient clipping, mixed-precision execution, and 8 attention heads in the principal Transformer modules.

Regression-specific settings include quantile-regression training, 999 quantiles, pinball loss, bias-free LayerNorm, and continuous linear target embeddings.

The paper reports approximately **24.5 H100 GPU-days of pretraining compute per model**.

## In-Context Learning and Fine-Tuning

The released TabICLv2 Regressor is primarily an in-context-learning model. A labelled support table conditions predictions without conventional from-scratch optimization on every new dataset.

TabICLv2 also supports explicit downstream fine-tuning through the upstream `FinetunedTabICLRegressor` implementation. Fine-tuned models should be documented as separate downstream model versions because their parameters and performance depend on the adaptation dataset and procedure.

## Evaluation

The upstream paper evaluates TabICLv2 regression on real-world tabular benchmarks including **TabArena** and **TALENT**.

The authors report that untuned TabICLv2 achieves stronger overall results than RealTabPFN-2.5 despite the latter using substantial downstream tuning and ensembling.

For TALENT regression tasks, the principal metric is **RMSE**. Supplementary regression metrics include MAE and R².

Aggregate benchmark rankings are useful for comparing general-purpose tabular methods but should not be interpreted as a fixed expected error for an arbitrary downstream regression task.

Regression metrics are scale and dataset dependent; there is therefore no meaningful universal "accuracy percentage" for this checkpoint.

## Uncertainty and Probabilistic Prediction

Because the regressor predicts a dense set of target quantiles, the model exposes more information than a single point estimate.

This can support prediction intervals, asymmetric uncertainty representation, probabilistic risk calculations, and distributional evaluation.

The presence of quantile outputs does **not** guarantee that the resulting intervals are well calibrated for every downstream population. Calibration, coverage, sharpness, and distribution shift should be evaluated on representative application data.

## Scalability

TabICLv2 introduces architectural and implementation changes specifically intended to improve performance on large tables. These include QASSMax for long-context generalization, efficient selective attention computation, CPU offloading, and disk-backed offloading.

The paper demonstrates million-scale inference while keeping resource consumption substantially below what would be required to hold all intermediate representations directly in accelerator or host memory.

A reported reference configuration processes a table of approximately one million samples and 500 features in roughly 450 seconds using high-end GPU hardware and offloading. These numbers characterize a published experimental environment rather than an application-level service guarantee.

## Reproducibility

### Exact Checkpoint Identity

For strict reproduction, preserve:

- **Checkpoint:** `tabicl-regressor-v2-20260212.ckpt`
- **Revision:** `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- **SHA-256:** `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`

### Released Pretraining Recipe Caveat

The current upstream TabICLv2 training scripts were migrated from the original private pretraining implementation and cross-checked against the released checkpoints.

The maintainers state that the migrated scripts have not yet been fully validated through an end-to-end reproduction of the original pretraining results.

The released checkpoint should therefore remain the canonical artifact for this exact version.

### Weight-Decay Implementation Detail

Although the paper describes cautious weight decay as part of the training methodology, the upstream reproduction scripts specify:

```text
use_cautious_wd = False
```

The project notes that cautious weight decay was not wired into Muon during the reference runs.

This implementation detail should be retained when attempting to reproduce the released checkpoint.

### Recommended Reproduction Record

For reproducible downstream use, record at minimum the exact checkpoint file and SHA-256, Hugging Face revision, `tabicl` package version, random seed, preprocessing, support/train split, inference estimator count, quantile or point-prediction settings, fine-tuning configuration if applicable, and hardware/software environment.

## Limitations

1. Regression performance is inherently dataset and target-scale dependent.
2. Pretraining used up to approximately 100 features; much larger feature spaces are outside the direct pretraining distribution.
3. The largest synthetic pretraining problems contained approximately 48,000 labelled training observations, although much larger contexts can be processed.
4. Quantile predictions require downstream calibration assessment when used as uncertainty estimates.
5. Strong benchmark performance does not guarantee low error on a particular application.
6. General demographic fairness has not been established.
7. Robustness under arbitrary distribution shift has not been established.
8. High-impact application suitability must be independently demonstrated.
9. Exact reproduction of the original pretraining run has not yet been demonstrated with the currently published reproduction scripts.
10. Point predictions can hide important uncertainty or tail behaviour; applications should use appropriate regression and probabilistic metrics where consequences warrant it.

## License

The core TabICL tabular implementation and released TabICLv2 checkpoints are distributed under the **BSD 3-Clause License**.

The upstream repository contains separately licensed functionality associated with forecasting-related components. This model card concerns the core TabICLv2 tabular regression checkpoint.

Licensing and governance requirements for downstream datasets and fine-tuned models must be considered independently.

## Model Ownership and Attribution

TabICLv2 was developed by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan of the Soda team at Inria. The upstream reference codebase is maintained at [soda-inria/tabicl](https://github.com/soda-inria/tabicl) and base model checkpoints are hosted on Hugging Face at [jingang/TabICL](https://huggingface.co/jingang/TabICL).

Downstream fine-tuned derivatives should clearly distinguish their modifications from the upstream pretrained checkpoint.

## Citation

Cite the TabICLv2 paper, the original TabICL foundation paper, and the upstream repository:

### Papers

- **TabICLv2 (2026):**  
  Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026). *TabICLv2: A better, faster, scalable, and open tabular foundation model.* ICML 2026. arXiv:2602.11139. https://doi.org/10.48550/arXiv.2602.11139

- **TabICL (2025):**  
  Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025). *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.* ICML 2025. arXiv:2502.05564. https://doi.org/10.48550/arXiv.2502.05564

### Upstream Repository

- **TabICL Codebase:**  
  Soda team, Inria. *TabICL: Open Tabular Foundation Models* [Software]. GitHub. https://github.com/soda-inria/tabicl

### BibTeX

```bibtex
@article{qu2026tabiclv2,
  title={{TabICLv2}: {A} better, faster, scalable, and open tabular foundation model},
  author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  journal={arXiv preprint arXiv:2602.11139},
  year={2026}
}

@inproceedings{qu2025tabicl,
  title={Tab{ICL}: {A} Tabular Foundation Model for In-Context Learning on Large Data},
  author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@misc{tabicl_repo,
  author = {Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  title = {Tab{ICL}: Open Tabular Foundation Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/soda-inria/tabicl}}
}

@misc{tabicl_hf,
  author = {Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  title = {{TabICL}: Open Tabular Foundation Models},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/jingang/TabICL}}
}
```

## Evaluation Status

### Established by the Upstream Work

The upstream evidence establishes general-purpose tabular regression, in-context learning, quantile-regression output, point and probabilistic prediction, optional downstream fine-tuning, strong evaluation on TabArena and TALENT, scalability beyond the direct pretraining context range, and open checkpoint availability.

### Application-Dependent or Not Generally Established

The upstream evidence does not establish universal RMSE, MAE, or R² on a particular downstream dataset, uncertainty calibration, subgroup parity, demographic fairness, robustness to arbitrary distribution shift, domain-specific safety, operational service levels, or suitability for high-impact autonomous decisions.

These properties must be evaluated for the downstream application.
