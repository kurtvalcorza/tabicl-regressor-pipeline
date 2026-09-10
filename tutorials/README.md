# TabICLv2 Regressor standalone Colab tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tabicl-regressor-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabicl-regressor-pipeline/blob/main/tutorials/tabiclv2_regressor_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jingang%2FTabICL-ffcc4d?style=flat)](https://huggingface.co/jingang/TabICL)
[![Upstream](https://img.shields.io/badge/Upstream-soda--inria%2Ftabicl-181717?style=flat&logo=github&logoColor=white)](https://github.com/soda-inria/tabicl)
[![arXiv](https://img.shields.io/badge/arXiv-2602.11139-b31b1b.svg)](https://arxiv.org/abs/2602.11139)

These notebooks make the model usable outside DIMER Workbench while preserving the repository's pinned checkpoint identity and serving-artifact contract.

| Notebook | Profile | Capability | Default runtime | Release status |
|---|---|---|---|---|
| [`tabiclv2_regressor_colab.ipynb`](tabiclv2_regressor_colab.ipynb) | `E2E` | Tabular regression: evaluation, optional adaptation, inference, export/reload | CPU (GPU optional for fine-tuning) | release-grade path; current-head CI enforces source + notebook-engine execution |
| [`tabiclv2_regressor_artifact_inference_colab.ipynb`](tabiclv2_regressor_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | External serving-bundle validation/reconstruction and new-data inference | CPU | release-grade path; current-head CI enforces source + notebook-engine execution |

## NOTEBOOK_SPEC v1.0 conformance

**Notebook specification:** `1.0`.

This repository remains the DIMER contract/docs umbrella and exposes an installable public reference API under `src/tabicl_regressor_pipeline/`. The tutorials install that API at immutable commit `41a4b2b3537da33b90a6d2562a351f3a3995a74e` and exercise it for their core model operations. The sibling validator/fine-tuner containers remain the DIMER Workbench production implementation; tutorial execution is reference-path evidence, not the final on-platform acceptance test.

[`requirements-release.txt`](requirements-release.txt) records the requested top-level environment and [`requirements-release.lock`](requirements-release.lock) records the resolved transitive graph used by the release notebooks. The lock contains `setuptools==78.1.0`, matching the exact PEP 517 build-backend pin in `pyproject.toml`. The repository adapter is installed with `--no-deps --no-build-isolation`, so the already-locked environment supplies the build backend instead of resolving a second isolated build graph. PyTorch is an explicit runtime-provided boundary and is verified as `2.11.0` rather than replaced after kernel startup.

Google Colab is the primary hosted runtime. Generic Jupyter/Python 3.13 use currently requires a writable `/content` workspace because tutorial artifact paths intentionally match the Colab/DIMER demonstration layout.

Notebook JSON is committed with cleared outputs/execution counts and stable nbformat cell IDs. Static validation rejects missing IDs, floating build-backend requirements, re-enabled build isolation, and the previously overbroad generic-Jupyter runtime claim.

### Durable SHOULD dispositions

- `MOD6` / artifact digests: implemented for the base checkpoint and exported payload.
- `SEC6`: implemented with 512 MiB expanded limit for base-checkpoint ZIP input and 1 GiB for serving artifacts.
- `SEC8` / `SEC9`: newly produced artifacts record payload sizes and an allowlist; the consumer verifies them. Legacy `tabicl-dimer-regressor-v1` bundles without those optional fields remain readable but emit explicit warnings.
- `DAT14`: hosted-runtime sensitive-data warnings are present before BYOD paths.

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

CI separates static checks from execution. Release verification executes the notebooks through real IPython/Jupyter kernels, preserving `%pip` and notebook semantics; the companion receives the producer artifact and a distinct fresh inference CSV through explicit external paths. Hosted-Colab execution may still be recorded as an additional surface check, but static checks alone are never runtime evidence.

## AI provenance

These tutorials were developed with substantial AI assistance using **GPT-5.6 Sol High**, via **OpenAI / ChatGPT**, under Agent Relay role **Builder**, with maintainer direction and review. Attribution is provenance, not sign-off or independent verification.
