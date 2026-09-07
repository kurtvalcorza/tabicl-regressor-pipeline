# TabICLv2 Regressor standalone Colab tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tabicl-regressor-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabicl-regressor-pipeline/blob/main/tutorials/tabiclv2_regressor_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jingang%2FTabICL-ffcc4d?style=flat)](https://huggingface.co/jingang/TabICL)
[![Upstream](https://img.shields.io/badge/Upstream-soda--inria%2Ftabicl-181717?style=flat&logo=github&logoColor=white)](https://github.com/soda-inria/tabicl)
[![arXiv](https://img.shields.io/badge/arXiv-2602.11139-b31b1b.svg)](https://arxiv.org/abs/2602.11139)

These notebooks make the model usable outside DIMER Workbench while preserving the repository's pinned checkpoint identity and serving-artifact contract.

| Notebook | Purpose |
|---|---|
| [`tabiclv2_regressor_colab.ipynb`](tabiclv2_regressor_colab.ipynb) | End-to-end tutorial: checkpoint → data → evaluation → optional fine-tuning → inference → portable bundle |
| [`tabiclv2_regressor_artifact_inference_colab.ipynb`](tabiclv2_regressor_artifact_inference_colab.ipynb) | Load a trusted exported/DIMER-style bundle and run inference without gradient fine-tuning |

## Main tutorial

- exact pinned `tabicl-regressor-v2-20260212.ckpt` acquisition and SHA-256 verification
- DIMER ZIP/direct `.ckpt` of the same pinned base checkpoint, or pinned upstream source; fine-tuned serving bundles use the artifact-inference notebook
- Diabetes sample or BYOD CSV/pre-split data
- finite numeric-target validation and training-target variation check
- pretrained in-context evaluation with MAE, MSE, RMSE, R², and Pearson correlation
- optional `FinetunedTabICLRegressor` CUDA fine-tuning
- companion classical tree baselines (LightGBM and Random Forest) with holdout leaderboard and device latency
- in-memory post-hoc point-prediction blending with holdout RMSE minimization and generalization assessment
- holdout-only artifact selection; independent test is evaluation only
- point prediction with scalar `prediction` output
- DIMER-style bundle export: checkpoint + training context + manifest
- fresh reload prediction-equivalence smoke test

## Artifact contract

TabICL remains an in-context learner after downstream fine-tuning. A deployable bundle therefore requires at minimum:

```text
checkpoints/best.ckpt
training_context.parquet
artifact.json
```

The inference-only notebook verifies internal digests, reconstructs the support context, and calls the ordinary TabICL estimator with `allow_auto_download=False`.

## Trust boundary

ZIP path checks prevent traversal and symlink extraction, but they do not make model deserialization trustworthy. Load only artifacts you produced yourself or obtained from a trusted source. When a known ZIP SHA-256 is available, compare it before extraction/loading.

The portable artifact also embeds the labelled training context. Treat the exported ZIP with the same dataset licence, access-control, retention, and disclosure requirements as the source training data.

## Runtime evidence

Static/CI checks validate notebook structure, pinned checkpoint identity, task semantics, CSV-header safeguards, artifact contract, badges, and AI provenance. Live Colab execution evidence should be recorded separately on the PR; do not infer runtime success from static CI alone.

## AI provenance

These tutorials were developed with substantial AI assistance using **GPT-5.6 Sol High**, via **OpenAI / ChatGPT**, under Agent Relay role **Builder**, with maintainer direction and review. Attribution is provenance, not sign-off or independent verification.
