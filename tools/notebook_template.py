"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment) — E2E.

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, checkpoint
staging) from repository files; this template holds the learner-facing prose, the guided layer and the learner
cells. Every learner cell calls ``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process
in an isolated, hash-locked environment, so nothing is installed into the notebook kernel. The ARTIFACT-INFERENCE
companion has its own template, ``tools/notebook_template_artifact_inference.py``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "tabicl-regressor-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabiclv2_regressor_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jingang%2FTabICL-ffcc4d?style=flat",
        "https://huggingface.co/jingang/TabICL",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-soda--inria%2Ftabicl-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/soda-inria/tabicl",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2602.11139-b31b1b.svg", "https://arxiv.org/abs/2602.11139"),
]


UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}
ENVIRONMENT = {
    "package": "tabicl_regressor_pipeline",
    "repo_name": REPO,
    "weights_key": "tabicl-regressor-v2",
    "modules": ["__init__.py", "api.py"],
    "entry_module": "api.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 0.2, "environment": 8.0},
    "runtime_modules": ["torch", "tabicl", "numpy", "pandas", "scikit-learn", "lightgbm"],
    "install_flags": ["--only-binary", ":all:"],
}

RUNTIME_PREREQ = (
    "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab with a **T4 GPU** is the documented runtime (the bounded fine-tuning stage needs CUDA); Kaggle or a Linux Jupyter kernel also work, and a CPU-only runtime runs everything except fine-tuning, which it skips with a printed note. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`tabicl` 2.1.1 with its fine-tuning extras, `torch` 2.11.0 with its CUDA libraries, `numpy` 2.5.3, `pandas` 2.3.3, `scikit-learn` 1.9.0, `lightgbm` 4.7.0). About 0.2 GB of disk is needed for the checkpoint and about 8 GB for the isolated environment."
)

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabiclv2_regressor",
    "notebook_name": "tabiclv2_regressor_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "carried_extra": {"sample-bundle/SAMPLE_BUNDLE.json": "examples/sample-bundle/SAMPLE_BUNDLE.json"},
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (a T4 GPU is recommended) builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, then runs each stage below in its own process: it stages and digest-verifies the pinned TabICLv2 regression checkpoint, loads scikit-learn's bundled diabetes table (no download), splits it 60/20/20 into support, holdout and independent test partitions, validates the tables into an input manifest and encodes them, evaluates the pretrained TabICLv2 regressor **adapted by in-context conditioning on the support rows**, then runs the pipeline's production adaptation — **gradient fine-tuning**, bounded to 3 epochs and 5 minutes, with the candidate selected on the holdout only if it beats the pretrained model — compares LightGBM, a random forest, a standardised linear regression and a training-mean baseline, writes the evaluation report, scores new rows, exports a DIMER-style serving bundle and reloads it in a fresh process. On a runtime without CUDA the fine-tuning stage is skipped with a printed note (TabICL fine-tuning needs a GPU) and everything else runs. No repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5). No hosted run of this revision has been recorded yet; the fine-tuning branch has never been executed in a recorded run."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` and `DATA_SOURCE` to `Upload CSV` or `Upload pre-split train/val/test` in Section 4 and run Sections 4–9 again. Give the files by path (`BYOD_CSV_PATH`, or `BYOD_TRAIN_PATH` / `BYOD_VAL_PATH` / optional `BYOD_TEST_PATH` — these work in Colab, Kaggle and Jupyter), or leave the paths empty in Colab to use upload dialogs. Name your numeric target column in `TARGET_COLUMN`. A missing target column is refused naming it; a target value that is present but not a number (text, a unit suffix, thousands separators such as `1,510.0`) is refused naming the column and the values; blank targets are dropped **and counted in the input manifest** (more than 20 % blank is refused); a feature column that is numeric except for a few stray strings is refused naming the column and the values (list it in `TEXT_COLUMNS` if it really is categorical). `RUN_NEW_DATA_INFERENCE` in Section 8 scores your own unlabelled rows. Files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "title": "TabICLv2 Regressor — DIMER E2E tabular regression tutorial (standalone)",
    "badges": BADGES,
    "capability": "end-to-end TabICLv2 tabular regression: immutable checkpoint acquisition, validated support data, in-context evaluation and bounded gradient fine-tuning against trivial and classical baselines, new-data point predictions, a DIMER-style serving bundle and its fresh reload",
    "intro": (
        "TabICLv2 (Qu et al.) is an **in-context tabular foundation model**: a transformer pretrained on synthetic tables "
        "to predict a query row's target by reading labelled example rows — the **support rows** or **context** — alongside "
        "it. `fit` registers the (encoded) support rows as context and every `predict` call feeds context plus query rows "
        "through the transformer and reads off a point prediction per query row; no gradient step happens there. "
        "`n_estimators=8` averages eight passes with different feature and row permutations. The DIMER pipeline's production "
        "workflow goes one step further: it **fine-tunes** the checkpoint by gradient descent on the support rows (AdamW, "
        "early stopping on the holdout) and keeps the fine-tuned model only if it beats the pretrained one on the holdout. "
        "This notebook runs both: conditioning by default, and a bounded fine-tune when a GPU is present.\n\n"
        "The upstream project supplies the model and the checkpoint; the carried package adds the pinned snapshot scheme, "
        "the table preparation and encoding rules, the metric set, the `validate_inputs` / `training_mean_baseline` / "
        "`evaluation_report` helpers, and the serving-bundle safety checks. Predictions are **continuous point estimates "
        "only**, with no per-prediction uncertainty interval and no shipped tolerance band. The default sample is "
        "scikit-learn's bundled diabetes table (442 patients, 10 standardised features, disease progression after one year); "
        "a plain linear regression is already a strong reference on it, so its metrics are tutorial sanity evidence, not a "
        "benchmark or production claim."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** the difference between in-context conditioning (registering support rows) and gradient fine-tuning, "
        "and why the candidate is selected on the holdout, never on the independent test (Section 6).\n"
        "2. **Diagnose** an invalid table from a validation refusal — including a target that is not a number — and say what a "
        "dropped-row count in the input manifest means (Sections 4, 5).\n"
        "3. **Compare** TabICLv2 with a training-mean baseline, a standardised linear regression, LightGBM and a random forest "
        "on the same partitions using MAE, RMSE and R² (Section 7).\n"
        "4. **Interpret** a paired standard error of an MSE difference and decide whether a gap between two models is "
        "within split noise; explain why a blend weight chosen on the holdout is selection-biased (Section 7).\n"
        "5. **Apply** the selected model to new rows and read point predictions in target units (Section 8).\n"
        "6. **Verify** that the exported bundle rebuilds an equivalent regressor in a fresh process (Section 9).\n"
        "7. **Predict**, run and **explain** how the ensemble size changes holdout and test error, in an optional activity "
        "(Section 10).\n"
        "8. **Write** an evidence-based conclusion that names the baselines, the selection basis and the limits of one public "
        "table (Conclusion)."
    ),
    "exclusions": (
        "classification, forecasting, calibrated per-prediction uncertainty intervals, or any deployment tolerance band. "
        "`prediction` is a continuous point estimate in target units; fine-tuning runs only on CUDA, and its selection is "
        "evidence on one holdout, not a guarantee of better generalisation."
    ),
    "prerequisites": [
        RUNTIME_PREREQ,
        "- **Knowledge:** basic pandas, what a train/holdout/test split is, and how to read a printed Python dictionary. MAE, RMSE, R², the paired standard error, in-context conditioning and fine-tuning are explained where they are first used, and the glossary collects them.",
        "- **Model file:** one checkpoint, `tabicl-regressor-v2-20260212.ckpt` (114 MB, released under BSD-3-Clause). It is a PyTorch/Lightning checkpoint: `tabicl` deserialises it, so only the digest-verified file is ever loaded.",
        "- **Data:** the default sample is scikit-learn's bundled diabetes table (442 rows, 10 numeric features, a numeric target), loaded from the installed package, so nothing is downloaded and no private data is needed; the optional `Sample: California Housing` downloads the California housing table from figshare through scikit-learn on first use and keeps a seeded 2,000-row subsample. BYOD (one CSV with a numeric target column, or pre-split `train`/`val`/`test` CSVs) is off by default so the sample path runs top to bottom without interaction. The tutorial's own floors are `MIN_TRAIN_ROWS` = 50 support rows and `MIN_EVAL_ROWS` = 2 holdout rows; the production DIMER validator described in the model card is stricter (10 evaluation rows). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook (Google Colab or Jupyter) and read "
                "short Python and pandas, and who want to see a tabular foundation model used the way a production pipeline uses "
                "it: conditioned, optionally fine-tuned with honest holdout selection, compared with strong classical models, and "
                "packaged as a reusable bundle. No experience with transformers is assumed; each term is explained where it is "
                "first needed, and the glossary below collects them.\n\n"
                "**Running it.** In Colab choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default "
                "path needs no edit, no upload, no account, no token and no runtime restart. Section 2 builds an isolated "
                "environment, which takes the longest (PyTorch and its CUDA libraries are a few GB); Section 6 then spends up to "
                "five minutes fine-tuning on a GPU. You can also run one cell at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files: the verified checkpoint, the partitions and fitted "
                "encoders, the fine-tuned checkpoint if one was selected, and JSON records.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–10) are the machine-learning workflow. *Infrastructure cells* "
                "(Sections 1–3: the runtime check, the carried code, the isolated environment and the checkpoint staging) are "
                "collapsed and titled **Infrastructure**. You may run them without studying their implementation: they exist for "
                "reproducibility and provenance, not as prerequisite machine-learning knowledge.\n\n"
                "**Form controls.** `DATA_SOURCE`, `USE_BYOD`, `TARGET_COLUMN`, the `BYOD_*_PATH` fields and `TEXT_COLUMNS` "
                "(Section 4); `RUN_FINE_TUNING`, `FINE_TUNE_EPOCHS` and `EVAL_METRIC` (Section 6); `RUN_NEW_DATA_INFERENCE` and "
                "`NEW_DATA_PATH` (Section 8); `RUN_ACTIVITY` and `ACTIVITY_N_ESTIMATORS` (Section 10). Leave them at their defaults "
                "for the first run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 6 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint. GPU kernels are not bitwise deterministic, so the last digits can vary between runs."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Validate and split** | a table with a numeric target | `validate_inputs`, a seeded random 60/20/20 split, support-fitted categorical encoders | an input manifest; 265 support, 88 holdout and 89 test rows |\n"
                "| **Condition** | the support rows | TabICLv2 reading them as context (`n_estimators=8`) | a point prediction for any query row |\n"
                "| **Fine-tune (GPU)** | support rows; holdout for early stopping | gradient fine-tuning of the checkpoint, ≤ 3 epochs / 5 min | a candidate checkpoint, kept only if it beats the pretrained model on the holdout |\n"
                "| **Evaluate** | holdout and test rows | `regression_metrics`, training-mean baseline, linear regression, LightGBM, random forest, an in-memory blend | MAE / RMSE / R², paired standard errors, an evaluation report |\n"
                "| **Package** | selected checkpoint + support rows + settings | ZIP bundle, `safe_extract_zip`, `verify_artifact_bundle` | a bundle that rebuilds an equivalent regressor |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux x86_64, GPU and disk; a run directory | the machine |\n"
                "| 2. Carry the code, build the environment | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify the model | [Engineering] | the 114 MB checkpoint downloaded at a fixed revision, digest-checked | identity and digest |\n"
                "| 4. Load the sample or your own data | [Concept] | the diabetes table (or your files), split 60/20/20 | partition sizes, digest |\n"
                "| 5. Validate and encode | [Evaluation practice] | input manifest; a refusal probe; encoders; training-mean baseline | the manifest |\n"
                "| 6. Condition and fine-tune | [Concept] | pretrained metrics; bounded fine-tuning and holdout selection | the selection |\n"
                "| 7. Baselines, blend and report | [Evaluation practice] | linear regression, LightGBM, random forest, blend; the evaluation report | the principal result |\n"
                "| 8. New-data inference | [Engineering] | eight rows (or your file) scored | the output contract |\n"
                "| 9. Export and fresh reload | [Engineering] | the bundle exported and rebuilt in a new process | the reload check and the digest |\n"
                "| 10. Optional activity | [Concept] | a different ensemble size on the same rows (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Run all, then read Sections 6, 7 and 9 and the conclusion."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **In-context learning / conditioning** | Predicting from labelled rows shown to the model at prediction time; `fit` only registers them. |\n"
                "| **Support rows / context** | The labelled rows the model reads; here the 265-row support partition. |\n"
                "| **`n_estimators`** | How many passes with different feature/row permutations are averaged into one prediction. |\n"
                "| **Fine-tuning** | Updating the checkpoint's weights by gradient descent on the support rows. |\n"
                "| **Early stopping** | Stopping fine-tuning when the holdout metric stops improving. |\n"
                "| **Holdout selection** | Keeping the fine-tuned candidate only if it beats the pretrained model on the holdout. |\n"
                "| **Independent test** | Rows used by no selection at all: the only unbiased estimate here. |\n"
                "| **Selection bias** | A score looks better than it will on new data because a choice was made by looking at it. |\n"
                "| **MAE / RMSE** | Mean absolute error and root mean squared error, both in target units; RMSE weighs large misses more. |\n"
                "| **R²** | 1 − MSE / variance of the target: 0 is the training-mean baseline, 1 is perfect, below 0 is worse than the mean. |\n"
                "| **Pearson r** | Linear correlation between predictions and targets; it ignores bias and scale. |\n"
                "| **Paired standard error** | The standard error of the per-row difference in squared error between two models on the same rows; a difference smaller than about twice it is within split noise. |\n"
                "| **Training-mean baseline** | Always predict the mean target of the support rows (R² ≈ 0). |\n"
                "| **Linear regression / LightGBM / random forest** | Classical models trained on the same support rows. |\n"
                "| **Blend** | A weighted average of TabICLv2's and LightGBM's predictions, weight chosen on the holdout. |\n"
                "| **Categorical encoder** | Ordinal codes fitted on the support rows only; unseen values get an 'unknown' code and are counted. |\n"
                "| **Serving bundle** | A ZIP of `artifact.json`, `checkpoints/best.ckpt` and `training_context.parquet`. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n"
                "| **BYOD** | Bring Your Own Data. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Load the sample or your own data · [Concept]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`. `Sample: Diabetes` (default) "
                "is the bundled numeric sanity check, split 60/20/20 at random (seed 42) into support, holdout and an independent "
                "test partition; `Sample: California Housing` downloads the California housing table from figshare through "
                "scikit-learn, keeps a seeded 2,000-row subsample and splits it the same way. With `USE_BYOD = True`: `Upload CSV` "
                "takes one CSV and carves an 80/20 holdout; `Upload pre-split train/val/test` takes your own partitions (`test` "
                "optional). Give files by path, or leave the path fields empty in Colab to choose them in upload dialogs.\n\n"
                "The stage refuses — naming the file and the rule — duplicate header names, a `TARGET_COLUMN` absent from the "
                "header (the message lists it), a **target value that is present but not a finite number** (the message names the "
                "values and their file lines; values written with thousands separators such as `1,510.0` are refused with that "
                "diagnosis rather than guessed), more than 20 % blank targets, and a feature column that is numeric except for a "
                "few stray strings (it would otherwise be encoded as categories without a word; list it in `TEXT_COLUMNS` if it "
                "really is categorical). Rows whose target is **blank** are **dropped and counted**: for a single CSV the count is "
                "carried into Section 5's input manifest as `dropped_missing_target_rows_before_split`. The stage first removes "
                "this notebook's earlier exports from `outputs/`, so a refused input never leaves an older result looking current.\n\n"
                "**BYOD privacy boundary.** Uploaded CSV bytes are read inside the current notebook runtime and are not sent "
                "by this notebook to an external inference or training service. After Section 3 the default path makes no network "
                "request (the diabetes table ships with scikit-learn); `Sample: California Housing` downloads from figshare."
            ),
            "code": (
                "DATA_SOURCE = 'Sample: Diabetes'  # @param [\"Sample: Diabetes\", \"Sample: California Housing\", \"Upload CSV\", \"Upload pre-split train/val/test\"]\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TARGET_COLUMN = 'target'  # @param {{type:\"string\"}}\n"
                "BYOD_CSV_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_TRAIN_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_VAL_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_TEST_PATH = ''  # @param {{type:\"string\"}}\n"
                "TEXT_COLUMNS = []  # @param {{type:\"raw\"}}\n\n"
                "def upload_files(what):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'No path was given for {{what}}, and the upload dialog exists only in Google Colab: set the BYOD path field(s).') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received for {{what}}. Run this cell again, or set the path field(s).')\n"
                "    saved = {{}}\n"
                "    for upload_name, payload in uploaded.items():\n"
                "        path = ROOT / 'inputs' / Path(upload_name).name\n"
                "        path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        path.write_bytes(payload)\n"
                "        saved[path.name.lower()] = str(path)\n"
                "    return saved\n\n"
                "paths = {{}}\n"
                "if USE_BYOD and DATA_SOURCE == 'Upload CSV':\n"
                "    if BYOD_CSV_PATH:\n"
                "        paths = {{'csv': BYOD_CSV_PATH}}\n"
                "    else:\n"
                "        saved = upload_files('one labelled CSV')\n"
                "        if len(saved) != 1:\n"
                "            raise ValueError(f'Upload exactly one CSV; got {{sorted(saved)}}.')\n"
                "        paths = {{'csv': next(iter(saved.values()))}}\n"
                "elif USE_BYOD and DATA_SOURCE == 'Upload pre-split train/val/test':\n"
                "    if BYOD_TRAIN_PATH or BYOD_VAL_PATH:\n"
                "        paths = {{'train': BYOD_TRAIN_PATH, 'val': BYOD_VAL_PATH, 'test': BYOD_TEST_PATH}}\n"
                "    else:\n"
                "        saved = upload_files('train.csv and val.csv (test.csv optional)')\n"
                "        if not {{'train.csv', 'val.csv'}} <= set(saved):\n"
                "            raise ValueError(f'Upload train.csv and val.csv (test.csv optional); got {{sorted(saved)}}.')\n"
                "        paths = {{'train': saved['train.csv'], 'val': saved['val.csv'], 'test': saved.get('test.csv', '')}}\n"
                "run_stage('data', data_source=DATA_SOURCE, use_byod=USE_BYOD, target_column=TARGET_COLUMN, paths=paths, text_columns=TEXT_COLUMNS)"
            ),
        },
        {
            "md": (
                "**What to notice:** `sample_kind: 'sample'`, 265 support, 88 holdout and 89 test rows, and the CSV digest. Every "
                "later number is measured on these exact partitions."
            ),
        },
        {
            "md": (
                "## 5. Validate the tables → input manifest, then encode · [Evaluation practice]\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks "
                "`prepare_regression_table` applies — unique column names, the target present and coerced to finite floats with "
                "the dropped rows counted, at least `MIN_TRAIN_ROWS` support rows (`MIN_EVAL_ROWS` for a holdout), at most "
                "`MAX_TRAIN_ROWS` rows and `MAX_FEATURES` columns, a target that varies — and returns an **input manifest** naming "
                "the schema, the observed structure (categorical columns, missing values, the target's minimum, maximum, mean and "
                "distinct count) and the verdict. It is written to `outputs/{stem}_input_manifest.json`. To show what rejection "
                "looks like, the stage also validates a probe with too few rows and records the package's own error message as a "
                "finding. The ceilings are printed before any model runs.\n\n"
                "The holdout and test partitions are then aligned to the support schema (`align_to_schema`), the categorical "
                "encoder is fitted on the support split only and applied everywhere (unseen values counted), and the trivial "
                "baseline is computed with `training_mean_baseline`: always predict the support rows' mean target."
            ),
            "code": "run_stage('validate')",
        },
        {
            "md": (
                "**What to notice:** the ceilings; the manifest entry for the support partition (10 numeric features, no categorical "
                "columns, the target from 25 to 346 with mean 151.0); the finding `too-few-rows-probe` rejected with the package's "
                "message; `dropped_non_finite_target_rows` all zero for the sample; and the training-mean baseline on the holdout — "
                "MAE 67.52, RMSE 76.16, R² −0.006.\n\n"
                "**Checkpoint:** a BYOD CSV has 5 rows with a blank target and 4 rows whose target reads `1,510.0`. What happens to "
                "each group, and why are they treated differently?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The blank rows carry no label, so they cannot be used: they are dropped and the count, 5, appears in the input "
                "manifest as `dropped_missing_target_rows_before_split`. The `1,510.0` rows do carry a label — written in a form "
                "pandas reads as text. Coercing them to 'missing' and dropping them would silently remove real, possibly "
                "systematically large, targets (the earlier version of this notebook did exactly that and reported 0 dropped), so "
                "Section 4 refuses the file, naming the column, the values and their lines. Fix the separators and the rows count.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Condition TabICLv2, then fine-tune it (GPU) · [Concept]\n\n"
                "`fit` does not train anything; it registers the support rows as the model's context (the checkpoint pinned in "
                "Section 3 is deserialised by `tabicl` here). `regression_metrics` scores the holdout and the independent test: "
                "**MAE** and **RMSE** are errors in target units (lower is better; RMSE weighs large misses more), **R²** compares "
                "the squared error with that of always predicting the mean (higher is better), and **Pearson r** is the linear "
                "correlation between predictions and targets.\n\n"
                "**Fine-tuning (on by default; needs CUDA).** With `RUN_FINE_TUNING` ticked on a GPU runtime, "
                "`create_finetuned_regressor` runs the upstream gradient fine-tuning for at most `FINE_TUNE_EPOCHS` (default 3) "
                "epochs and 300 seconds, with early stopping on the holdout and `EVAL_METRIC`; the best checkpoint is reloaded into "
                "an ordinary regressor with the same inference ensemble for a fair comparison, and the candidate replaces the "
                "pretrained model **only** if it beats it on the holdout (`compare_metric`) and the holdout has at least "
                "`MIN_SELECTION_HOLDOUT_ROWS` = 50 rows. The independent test is evidence only; a worse test result is printed as a "
                "warning and never changes the selection. TabICL writes one checkpoint per epoch; the stage keeps `best.ckpt` only. "
                "The fine-tuned `best.ckpt` may remain larger than the base checkpoint because upstream training state can be "
                "embedded; the format is kept for checkpoint compatibility. On a CPU runtime the stage skips fine-tuning and records "
                "why in the result.\n\n"
                "**Reproducibility boundary.** The split seed and the TabICL ensemble seed are fixed; that makes the partitioning and "
                "the estimator configuration repeatable, but not bitwise-identical floating-point results across devices, library "
                "builds or kernels.\n\n"
                "**Predict before running:** the training-mean baseline has holdout MAE 67.5 and a standardised linear regression "
                "38.2 (Section 7). Where will the pretrained model land — and will three epochs of fine-tuning on 265 rows move the "
                "holdout MAE by more than a point?"
            ),
            "code": (
                "RUN_FINE_TUNING = True  # @param {{type:\"boolean\"}}\n"
                "FINE_TUNE_EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "EVAL_METRIC = 'mae'  # @param [\"mae\", \"mse\", \"r2\"]\n"
                "run_stage('condition', run_fine_tuning=RUN_FINE_TUNING, fine_tune_epochs=FINE_TUNE_EPOCHS, eval_metric=EVAL_METRIC)"
            ),
        },
        {
            "md": (
                "**What to notice:** the pretrained holdout and test metrics; on a GPU the fine-tuning log, the candidate's holdout "
                "metrics, and `recommended_for_export` with its `selection_basis` (`holdout:mae` when the holdout has ≥ 50 rows); "
                "on a CPU `fine_tune.skipped` with the reason. No recorded run of this revision exists yet, so read your own numbers "
                "against the baselines of Section 7.\n\n"
                "**Checkpoint:** why is the candidate selected on the holdout and never on the independent test — and what would "
                "go wrong if the test were used?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Any partition used to make a choice stops being an unbiased estimate of performance on new data: picking the "
                "better of two models on a partition favours whichever happened to fit that partition's quirks. The holdout pays "
                "that price so the test does not. If the test chose the model, the reported test score would be optimistic and "
                "there would be no clean estimate left. That is why the stage only *warns* when the test disagrees with the "
                "holdout's choice: the disagreement is evidence about selection noise, not a reason to override it.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Classical baselines, blending and the evaluation report · [Evaluation practice]\n\n"
                "A standardised linear regression, LightGBM and a random forest are fitted on the exact same encoded support rows "
                "and scored on the exact same holdout and test partitions, so the comparison is fair. Because MAE and R² on 88 rows "
                "are noisy, the stage also reports the **paired standard error** of the MSE difference between TabICLv2 and the "
                "linear model on the same rows: a difference smaller than about twice its standard error is within split noise. A "
                "convex blend of TabICLv2 and LightGBM predictions is chosen by minimising holdout RMSE over a 101-point grid; it is "
                "evaluated in memory only, with its own paired standard error on the holdout and on the independent test, and the "
                "exported bundle in Section 9 is the unchanged selected model. Latencies are medians of five warmed runs on this "
                "runtime.\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. Here it carries the "
                "selected model's holdout metrics (`mae`, `mse`, `rmse`, `r2`, `pearsonr`), the independent-test metrics, the "
                "training-mean baseline, the classical baselines, the paired comparisons, the blend and the fine-tuning record, "
                "with the verdict `sample-sanity`: one seeded random split with no dispersion estimate. Without a labelled holdout "
                "the verdict would be `not-measurable`. The report is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** will the blend beat TabICLv2 alone on the **independent test**, and by more than "
                "twice its paired standard error?"
            ),
            "code": "run_stage('report')",
        },
        {
            "md": (
                "**What to notice:** on the holdout the standardised linear regression reaches MAE 38.22 / R² 0.581, the random "
                "forest 42.28 / 0.495 and LightGBM 44.94 / 0.426; on the independent test 46.72 / 0.439, 44.67 / 0.470 and "
                "47.77 / 0.369. Read TabICLv2's line from your run beside them, then `tabicl_vs_linear_holdout` (the MSE difference "
                "and its paired standard error), the blend's weight and its holdout and test paired comparisons, and the report's "
                "`verdict: sample-sanity`.\n\n"
                "**Checkpoint:** the blend lowers holdout RMSE, but on the independent test its MSE difference against TabICLv2 is "
                "smaller than one paired standard error. What do you report?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "That the blend shows no evidence of improvement. Its weight was chosen to minimise holdout RMSE, so its holdout "
                "score is selection-biased by construction. On the test, untouched by the choice, a difference inside one or two "
                "standard errors is indistinguishable from split noise. Report the test numbers with their paired standard error, "
                "note that a linear model is as good on this table, and do not export the blend.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. New-data inference with point predictions · [Engineering]\n\n"
                "By default the `predict` stage scores the first eight holdout rows — already scored in Section 7, so this "
                "demonstrates the output contract, not a new evaluation — and writes their unlabelled copy, with their original "
                "`row_id`, to `outputs/{stem}_new_rows.csv`, the input the companion artifact-inference notebook accepts. Tick "
                "`RUN_NEW_DATA_INFERENCE` and set `NEW_DATA_PATH` (or, in Colab, leave it empty to upload) to score your own "
                "unlabelled CSV: `read_inference_csv` and `validate_inputs(..., target_column=None, feature_columns=...)` require a "
                "unique header, no pre-existing `prediction` column and every feature present; extra columns are kept in the output "
                "and not passed to the model; the support-fitted encoder is applied (unseen values counted). The output adds one "
                "`prediction` column: a continuous point estimate in target units, with no uncertainty interval. "
                "`outputs/{stem}_result.json` records the predictions, metrics, baselines, the fine-tuning record, the evaluation "
                "report, the input manifest, the sample digest, the notebook's source, the model identity, revision and licence, "
                "and the runtime identity as a provenance record."
            ),
            "code": (
                "RUN_NEW_DATA_INFERENCE = False  # @param {{type:\"boolean\"}}\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "new_data_file = ''\n"
                "if RUN_NEW_DATA_INFERENCE:\n"
                "    new_data_file = NEW_DATA_PATH or next(iter(upload_files('one unlabelled CSV').values()))\n"
                "run_stage('predict', new_data_path=new_data_file)"
            ),
        },
        {
            "md": "**What to notice:** eight rows with their `row_id`, the ten features and one `prediction` in the target's units.",
        },
        {
            "md": (
                "## 9. Export a DIMER-style serving bundle, then prove a fresh reload · [Engineering]\n\n"
                "The ZIP carries the minimum serving contract the DIMER pipeline expects: `checkpoints/best.ckpt`, "
                "`training_context.parquet`, and `artifact.json`. The training context is *required*: TabICL is still an in-context "
                "learner at serve time, so whoever loads the bundle must hand the model the same rows you evaluated with, and the "
                "context inherits the source data's confidentiality, licensing, retention and disclosure obligations. "
                "`artifact.json` records the feature and target columns, the base-model identity and digest, the selected mode and "
                "its basis, the metrics, the inference settings including the fitted categorical encoders, per-file sizes and "
                "SHA-256 digests, the payload allowlist and the producer runtime. The `export` stage prints the ZIP's SHA-256 — the "
                "**trusted digest** to give the companion notebook (`EXPECTED_ZIP_SHA256`) — and, on the default path, whether the "
                "support table equals the companion's pinned sample bundle (`matches_pinned_sample_context`).\n\n"
                "The `reload` stage then runs in a **fresh process**: it extracts the ZIP with `safe_extract_zip` (member by member, "
                "after path, symlink and expanded-size checks — the same function the companion applies), verifies the allowlist, "
                "sizes and digests with `verify_artifact_bundle`, rebuilds the regressor from the bundle alone (checkpoint + "
                "context + recorded settings), and checks that its predictions agree with the exporting process's on eight holdout "
                "rows (`rtol=1e-5`, `atol=1e-7`).\n\n"
                "**Predict before running:** if Section 6 selected the fine-tuned candidate, which checkpoint is in the bundle, and "
                "what does the companion then check about it?"
            ),
            "code": "run_stage('export')\nrun_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** the ZIP path, its `zip_sha256`, the `mode`, the checkpoint size, "
                "`matches_pinned_sample_context: True` on the default path; then `max_abs_prediction_difference` and the `PASS` line.\n\n"
                "**Checkpoint:** the bundle's checkpoint is code-capable (a PyTorch/Lightning checkpoint). What protects the person "
                "who loads it?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "For a `pretrained` bundle, the companion requires the checkpoint's SHA-256 to equal the pinned base checkpoint's, so "
                "it is byte for byte the file the model's publisher released and Section 3 verified. For a `fine-tuned` bundle no "
                "such reference exists: the only protection is that the ZIP's digest matches a trusted digest received from the "
                "producer through a separate channel (`EXPECTED_ZIP_SHA256`). Internal digests inside `artifact.json` protect "
                "against corruption, not against someone who rewrites the manifest too. So: load fine-tuned bundles only from "
                "producers you trust, with their printed digest.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 10. Optional activity: does a larger ensemble help? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** if TabICLv2 averages 32 permuted passes instead of 8 "
                "on the same 265 support rows, will holdout and test MSE change by more than twice their paired standard error? "
                "**Change:** tick `RUN_ACTIVITY` (keep `ACTIVITY_N_ESTIMATORS = 32`, or try 1). **Run** this cell. **Observe** the "
                "`mse_delta_changed_minus_canonical`, its `paired_se` and `within_two_se` for both partitions. **Explain** what the "
                "ensemble size buys on a table this size. The activity conditions the pretrained model only, writes to "
                "`outputs/activity/`, and stops if any canonical output changed."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_N_ESTIMATORS = 32  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', n_estimators=ACTIVITY_N_ESTIMATORS)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** the two `n_estimators`, the holdout and test metrics for each, and whether "
                "`within_two_se` is true. `canonical_outputs_unchanged: True`.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "More passes average away some of the variance that comes from the feature and row order, so very small ensembles "
                "(1) usually do worse and the gain flattens quickly. With 88 holdout rows the paired standard error of an MSE "
                "difference is large compared with what the ensemble size can change, so a difference inside two standard errors "
                "says only that this table cannot tell the settings apart — not that they are equivalent in general.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel. |\n"
        "| `No CUDA GPU detected`; later `Fine-tuning skipped: no CUDA device` | a CPU runtime | Everything else runs. For the documented runtime choose *Runtime → Change runtime type → T4 GPU*. |\n"
        "| `Not enough free disk` | the isolated environment needs about 8 GB | Start a fresh runtime; an environment built from the same lock is reused. |\n"
        "| `Carried file integrity failure` | a carried file was edited in the notebook | Open a fresh copy from the repository. |\n"
        "| `uv … mismatch`, `URLError`, or `CalledProcessError` from `uv` | network or a transient PyPI error | Re-run the Section 2 install cell. Never remove a pin or a hash. |\n"
        "| `The run directory … has no carried files, or the isolated environment is gone` | Section 1 run with `NEW_RUN_DIRECTORY` ticked | Run Sections 1–3 again, or *Run all*. |\n"
        "| `RuntimeError: Stage '…' failed (exit 1): …` | the stage's own error follows the colon | Find it below; fix the cause and re-run from that cell. |\n"
        "| A Hub download error in Section 3, or `… sha256 … != manifest` | a transient failure or a corrupted download | Re-run Section 3; delete the partial file under `weights/` if the digest fails. Never edit the manifest. |\n"
        "| `URLError` or `HTTPError` with `Sample: California Housing` | figshare unreachable | Retry, or use `Sample: Diabetes`. |\n"
        "| `CUDA out of memory` in Section 6 | a large BYOD table | Untick `RUN_FINE_TUNING`, or restart on a fresh T4. |\n"
        "| `Fine-tuning did not produce best.ckpt` | the time limit ended before the first checkpoint | Re-run Section 6, or lower `FINE_TUNE_EPOCHS`; report it if it repeats. |\n"
        "| `the target column … is not in the header` | your label column has another name | Set `TARGET_COLUMN`. |\n"
        "| `… written with thousands separators` | targets such as `1,510.0` | Remove the separators and run again. |\n"
        "| `… value(s) that are not finite numbers` | text, units or notes in the target column | Fix the values, or leave them empty to drop the rows. |\n"
        "| `… rows have a blank target …, more than 20%` | most of the target is missing | Fill or remove the blank rows first. |\n"
        "| `column … is numeric except for N value(s)` | stray text in a numeric feature | Fix the values, or list the column in `TEXT_COLUMNS`. |\n"
        "| `need at least 50 labelled rows` / `Regression target must vary` | a table below the floors | Supply more rows, or check that the target column is the one you meant. |\n"
        "| `No path was given …` outside Colab, or `The upload was cancelled or empty` | no file supplied | Set the path field(s), or run the cell again and choose files. |\n\n"
        "## Interpretation and limits\n\n"
        "`prediction` is a continuous point estimate in target units, with no uncertainty interval and no shipped tolerance "
        "band; any deployment tolerance must be chosen on the caller's own labelled, domain-representative data. The "
        "evaluation report's `sample-sanity` verdict names what it is: one seeded random split of a public sample with no "
        "dispersion estimate — tutorial evidence that must not be generalised. On the diabetes sample a standardised linear "
        "regression already reaches holdout MAE 38.22 / R² 0.581 and test MAE 46.72 / R² 0.439 on these partitions, and with "
        "88 and 89 rows the paired standard error of an MSE difference is large, so a few points of MAE between models rank "
        "nothing. No recorded run of this notebook revision exists yet; read the numbers your run printed. The fine-tuning "
        "selection and the blend weight are chosen on the holdout and are therefore selection-biased; only the independent "
        "test is unbiased. Rows that are not independent, targets outside the support range, unseen categories and support "
        "sets near the ceilings all change results in ways these metrics do not measure.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire "
        "and digest-verify the pinned checkpoint, validate the demonstrated tables (refusing unparseable targets), condition "
        "on regression support rows, fine-tune with holdout-based selection on a GPU, compute sample metrics against trivial "
        "and classical baselines with paired standard errors, write the input manifest and the evaluation report, score rows "
        "with point predictions, export a DIMER-style serving bundle and rebuild an equivalent regressor from that bundle "
        "alone in a fresh process — without the repository being reachable. It does **not** establish benchmark "
        "superiority, domain generalisation, fairness, robustness, calibrated uncertainty, production safety, or deployment "
        "fitness.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three to five sentences, using the numbers your run printed:\n\n"
        "1. **Result:** the selected model's holdout and test MAE, RMSE and R² beside the training mean, the linear "
        "regression, LightGBM and the random forest.\n"
        "2. **Adaptation:** whether fine-tuning ran, what it changed on the holdout, and which model was selected and why.\n"
        "3. **Reading:** whether any difference exceeds about twice its paired standard error, and what the blend's test "
        "comparison tells you.\n"
        "4. **Reuse and limits:** what the Section 9 reload proved, and the one limitation you would fix first.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "On the 88-row holdout and 89-row independent test of the diabetes table, the selected TabICLv2 model scored the MAE "
        "and R² printed in Sections 6 and 7, against MAE 67.5 / R² ≈ 0 for the training mean and 38.2 / 0.581 (holdout) and "
        "46.7 / 0.439 (test) for a standardised linear regression on the same partitions. Fine-tuning (three epochs on the T4) "
        "was kept only if it beat the pretrained model on the holdout. The MSE difference between TabICLv2 and the linear "
        "model is within about two paired standard errors on this table, so the run shows that TabICLv2 matches a simple, "
        "strong reference here without hyperparameter search, not that it beats it; the blend's test comparison was inside "
        "its standard error, so it adds nothing demonstrable. The bundle rebuilt an equivalent regressor in a fresh process. "
        "Before deployment I would evaluate on repeated splits of a larger, domain-representative table and add a calibrated "
        "uncertainty estimate.\n\n"
        "</details>\n\n"
        "**Next experiments:** switch `DATA_SOURCE` to `Sample: California Housing` (2,000 rows: watch the paired standard "
        "errors shrink); untick `RUN_FINE_TUNING` and compare the selection record; upload your own pre-split partitions; feed "
        "`outputs/{stem}_artifact.zip`, its printed digest and `outputs/{stem}_new_rows.csv` to the companion "
        "artifact-inference notebook in a separate session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/soda-inria/tabicl\n"
        "- TabICLv2 paper: https://arxiv.org/abs/2602.11139\n"
        "- TabICL paper: https://arxiv.org/abs/2502.05564\n"
        "- Diabetes data: Efron, Hastie, Johnstone and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics 32(2)"
    ),
}
