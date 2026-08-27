---
license: bsd-3-clause
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

## Description

TabICLv2 Regressor is a pretrained tabular foundation model developed by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan of the Soda team at Inria.

It performs supervised regression on structured or tabular data using **in-context learning (ICL)**. A labelled support table containing numerical targets is provided as context together with unseen observations, and the pretrained Transformer performs task adaptation during prediction.

The regression checkpoint is a distinct TabICLv2 model rather than the classification checkpoint with a changed output layer. It is separately pretrained for regression using **quantile regression**, predicting 999 target quantiles and optimizing them with pinball loss.

For ordinary point prediction, the reference implementation combines the predicted quantiles into a continuous numeric estimate. The model can also expose quantile information for probabilistic prediction.

## Model Details

- **Model name:** TabICLv2 Regressor
- **Model family:** TabICLv2
- **Checkpoint identifier:** `tabicl-regressor-v2-20260212.ckpt`
- **Hugging Face repository:** `jingang/TabICL`
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
- **Pinned Hugging Face revision:** `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- **SHA-256:** `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`

The checkpoint revision and SHA-256 should be retained for strict model-version provenance. The regression and classification checkpoints are separately pretrained artifacts and are not interchangeable.

## Intended Use and Limitations

### Primary Intended Uses

TabICLv2 Regressor is intended for supervised prediction of continuous numerical targets from structured or tabular features.

Appropriate uses include demand or quantity prediction, price estimation, duration prediction, score prediction, scientific or engineering regression, operational forecasting represented as a supervised feature table, resource or load estimation, and other row-per-observation continuous prediction problems.

The model can be used directly through in-context learning or adapted further through explicit downstream fine-tuning.

TabICLv2 was pretrained on synthetic datasets with sizes extending to approximately 48,000 labelled training examples and up to 100 features. The upstream implementation demonstrates useful generalization beyond this range, including substantially larger datasets and higher-dimensional tables, but such cases remain outside the direct pretraining distribution.

### Primary Intended Users

Primary users include data scientists, machine-learning researchers, machine-learning engineers, scientific researchers, software developers, and practitioners requiring flexible regression on structured datasets.

Users remain responsible for choosing meaningful evaluation metrics, constructing leak-free train/test splits, assessing distribution shift, and determining whether prediction error is acceptable for the intended application.

### Out-of-Scope Use Cases

The model is not intended for categorical classification, raw image/language/audio/video modelling, unsupervised clustering, direct causal inference, generative modelling, untransformed time-series forecasting where temporal structure is not appropriately represented, or autonomous high-impact decisions without domain-specific validation and governance.

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

## Factors

### Groups

The foundation model was pretrained on synthetic datasets rather than human demographic cohorts. No general demographic parity or subgroup-fairness property follows from the pretraining methodology. Human-centred applications require their own subgroup analysis.

### Instrumentation

TabICLv2 consumes tabular representations rather than raw measurements from a specific instrument. If features originate from sensors, laboratory instruments, surveys, administrative systems, or other acquisition processes, their limitations should be assessed separately.

### Environment

Temporal, geographic, institutional, measurement, or population changes may alter the data distribution and degrade regression performance. The model has no universal guarantee of invariance across such environments.

## Ethical Considerations and Risks

Potential downstream risks include numerically incorrect predictions, poorly calibrated uncertainty estimates, biased errors across relevant populations, distribution shift, training/evaluation leakage, use of sensitive features, automation bias, and inappropriate extrapolation from benchmark results.

For high-impact applications, numerical error should be evaluated in the context of its real-world consequences rather than only aggregate RMSE or R².

The model has not been established as universally suitable for autonomous medical, financial, legal, employment, safety-critical, or public-benefit decisions.

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

TabICLv2 was developed by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan of the Soda team at Inria.

Downstream fine-tuned derivatives should clearly distinguish their modifications from the upstream pretrained checkpoint.

## Citation

Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026). *TabICLv2: A better, faster, scalable, and open tabular foundation model.* ICML 2026. arXiv:2602.11139. https://doi.org/10.48550/arXiv.2602.11139

## Evaluation Status

### Established by the Upstream Work

The upstream evidence establishes general-purpose tabular regression, in-context learning, quantile-regression output, point and probabilistic prediction, optional downstream fine-tuning, strong evaluation on TabArena and TALENT, scalability beyond the direct pretraining context range, and open checkpoint availability.

### Application-Dependent or Not Generally Established

The upstream evidence does not establish universal RMSE, MAE, or R² on a particular downstream dataset, uncertainty calibration, subgroup parity, demographic fairness, robustness to arbitrary distribution shift, domain-specific safety, operational service levels, or suitability for high-impact autonomous decisions.

These properties must be evaluated for the downstream application.
