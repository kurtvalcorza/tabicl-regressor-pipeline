"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier) — E2E.

Only the task-specific prose and stage cells live here. Runtime install, the embedded package module
(``src/tabicl_regressor_pipeline/api.py``), and the model pin/stage/verify cell are produced by the generator from
repository sources so they cannot drift from the package. The ARTIFACT-INFERENCE companion has its own template,
``tools/notebook_template_artifact_inference.py``.
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

TEMPLATE = {
    "package": "tabicl_regressor_pipeline",
    "repo_name": REPO,
    "stem": "tabiclv2_regressor",
    "notebook_name": "tabiclv2_regressor_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the pinned TabICLv2 checkpoint, loads scikit-learn's bundled diabetes table (no download), validates the tables into an input manifest and encodes them, evaluates the pretrained TabICLv2 regressor **adapted by in-context conditioning on the training split** (the adaptation stage that runs by default — no gradient update), fits classical tree baselines and a training-mean baseline for comparison (MAE/RMSE/R²) and writes the evaluation report, exports a DIMER-style serving bundle and reloads it from disk to prove the fresh boundary. Gradient fine-tuning (`FinetunedTabICLRegressor`) is an optional experiment (`RUN_FINE_TUNING`, off by default, Section 6) because it needs a GPU-sized time budget; a reviewer reading NOTEBOOK_SPEC 2.0 RUN7/FT2 as requiring gradient adaptation on the default path should treat that as an open decision. No repository clone, DIMER worker or service, credential, upload dialog or configuration edit is required (§5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one labelled CSV (or pre-split files); it enters the same validation, encoding, in-context conditioning, baseline, evaluation, export and fresh-reload cells as the sample (DAT14), and `RUN_NEW_DATA_INFERENCE` in Section 8 scores your own unlabelled rows as point predictions. Expected schema, ceilings and privacy guidance are stated in the Prerequisites and in Section 4; uploads stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "TabICLRegressionPipeline",
    "weights_key": "tabicl-regressor-v2",
    # generator /2: the package is one module, api.py (not pipeline.py); it holds the identity constants and the
    # only `__file__` use (DEFAULT_WEIGHTS_DIR, rewritten by the default rule).
    "modules": ["api.py"],
    "entry_module": "api.py",
    # `from_pretrained` stages + verifies the snapshot and constructs the upstream estimator on the verified
    # checkpoint (no auto-download); tabicl deserialises the checkpoint on the first fit.
    "model_load": "TabICLRegressionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, n_estimators=8, random_state=42)",
    "runtime_imports": ["torch", "numpy", "pandas", "sklearn"],
    "title": "TabICLv2 Regressor — DIMER E2E tabular regression tutorial (standalone)",
    "badges": BADGES,
    "capability": "end-to-end TabICLv2 tabular regression: immutable checkpoint acquisition, validated support data, in-context evaluation against trivial and classical baselines, optional gradient fine-tuning, new-data inference, a DIMER-style serving bundle and its fresh reload",
    "intro": (
        "TabICLv2 is an in-context tabular foundation model: `fit` registers the (encoded) training rows as the "
        "model's context and every `predict` call feeds context plus query rows through the transformer — no gradient "
        "step happens unless you opt into the fine-tuning gate. The upstream project supplies the model and the "
        "checkpoint; the carried package adds the pinned snapshot scheme, the table preparation and encoding rules, "
        "the metric set, the `validate_inputs` / `training_mean_baseline` / `evaluation_report` helpers, and the "
        "serving-bundle safety checks. The default sample is scikit-learn's bundled diabetes table; its metrics are "
        "tutorial sanity evidence, not a benchmark or production claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable "
        "upstream checkpoint, load a public sample or your own CSV(s) and validate them into an input manifest, "
        "evaluate the pretrained model on a holdout and an independent test partition against the training-mean "
        "baseline, optionally fine-tune on CUDA with holdout-based selection, benchmark LightGBM and Random Forest on "
        "the same partitions and blend in memory, write an evaluation report, optionally score new rows, export a "
        "DIMER-style serving bundle and prove it reloads from a fresh directory."
    ),
    "exclusions": (
        "classification, forecasting, calibrated per-prediction uncertainty intervals, or any deployment tolerance "
        "band. Predictions are **continuous point estimates only**; the fine-tuning path runs only on CUDA and only when "
        "`RUN_FINE_TUNING` is switched on."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.11+). The default path runs on CPU and uses CUDA automatically when available; the fine-tuning gate requires CUDA. The pinned `torch==2.11.0` install is the largest download of the run.",
        "- **Knowledge:** basic pandas; what a holdout, an independent test partition, MAE, RMSE and R² are.",
        "- **Data:** the default sample is scikit-learn's bundled diabetes table (442 rows, 10 numeric features), loaded from the installed package, so nothing is downloaded and no private data is needed; `Sample: California Housing` fetches a 2,000-row subsample through scikit-learn. BYOD upload (one CSV, or pre-split `train.csv`/`val.csv`/`test.csv`) is selected through `DATA_SOURCE` and is off by default so the sample path runs top-to-bottom without interaction. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Load the sample or your own data\n\n"
                "`Sample: Diabetes` (default) is the bundled numeric sanity check, split 60/20/20 into support, holdout and "
                "an independent test partition; `Sample: California Housing` is a deterministic 2,000-row numeric subsample "
                "fetched through scikit-learn. `Upload CSV` takes one CSV with a numeric `target` column and splits it "
                "80/20; `Upload pre-split train/val/test` takes your own partitions (`test.csv` optional). The target is "
                "coerced to finite floats and rows where that fails are **dropped and counted** (reported in the input "
                "manifest of Section 5, never hidden). Non-numeric feature columns are ordinal-encoded with maps fitted on "
                "the support split only; unseen or missing values map to an extra 'unknown' code and are counted.\n\n"
                "**BYOD privacy boundary.** Uploaded CSV bytes are read inside the current notebook runtime and are not sent "
                "by this notebook to an external inference or training service; the only network request on the default "
                "path is the pinned checkpoint download of Section 3."
            ),
            "code": (
                "import hashlib\n\n"
                "from sklearn.datasets import fetch_california_housing, load_diabetes\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "DATA_SOURCE = 'Sample: Diabetes'  # @param [\"Sample: Diabetes\", \"Sample: California Housing\", \"Upload CSV\", \"Upload pre-split train/val/test\"]\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TARGET_COLUMN = 'target'\n"
                "VALIDATION_SPLIT = 0.20\n"
                "RANDOM_SEED = 42\n"
                "if USE_BYOD and DATA_SOURCE.startswith('Sample'):\n"
                "    raise ValueError('USE_BYOD=True requires DATA_SOURCE = \"Upload CSV\" or \"Upload pre-split train/val/test\".')\n"
                "if DATA_SOURCE.startswith('Upload') and not USE_BYOD:\n"
                "    raise ValueError('Set USE_BYOD=True to use an upload DATA_SOURCE.')\n\n"
                "test_data = None\n"
                "if DATA_SOURCE == 'Sample: Diabetes':\n"
                "    dataset = load_diabetes(as_frame=True)\n"
                "    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})\n"
                "    train_data, remainder = train_test_split(frame, test_size=0.4, random_state=RANDOM_SEED)\n"
                "    holdout_data, test_data = train_test_split(remainder, test_size=0.5, random_state=RANDOM_SEED)\n"
                "    data_name, sample_kind = 'sklearn-diabetes', 'sample'\n"
                "elif DATA_SOURCE == 'Sample: California Housing':\n"
                "    TARGET_COLUMN = 'MedHouseVal'\n"
                "    dataset = fetch_california_housing(as_frame=True)\n"
                "    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})\n"
                "    if len(frame) > 2000:\n"
                "        frame = frame.sample(n=2000, random_state=RANDOM_SEED).reset_index(drop=True)\n"
                "    train_data, remainder = train_test_split(frame, test_size=0.4, random_state=RANDOM_SEED)\n"
                "    holdout_data, test_data = train_test_split(remainder, test_size=0.5, random_state=RANDOM_SEED)\n"
                "    data_name, sample_kind = 'sklearn-california-housing-2000', 'sample'\n"
                "elif DATA_SOURCE == 'Upload CSV':\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError('Upload exactly one CSV')\n"
                "    data_name, payload = next(iter(uploaded.items()))\n"
                "    frame, _dropped = prepare_regression_table(read_csv_payload(payload, data_name), TARGET_COLUMN)\n"
                "    train_data, holdout_data = train_test_split(frame, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    by_name = {{Path(key).name.lower(): (key, value) for key, value in uploaded.items()}}\n"
                "    if not {{'train.csv', 'val.csv'}} <= set(by_name):\n"
                "        raise ValueError('Upload train.csv and val.csv; test.csv optional')\n"
                "    train_data = read_csv_payload(by_name['train.csv'][1], 'train.csv')\n"
                "    holdout_data = read_csv_payload(by_name['val.csv'][1], 'val.csv')\n"
                "    if 'test.csv' in by_name:\n"
                "        test_data = read_csv_payload(by_name['test.csv'][1], 'test.csv')\n"
                "    data_name, sample_kind = 'pre-split upload', 'BYOD'\n\n"
                "sample_sha256 = hashlib.sha256(pd.concat([train_data, holdout_data] + ([test_data] if test_data is not None else [])).to_csv(index=False).encode('utf-8')).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': data_name, 'target': TARGET_COLUMN, 'train_rows': len(train_data), 'holdout_rows': len(holdout_data), 'test_rows': 0 if test_data is None else len(test_data), 'csv_sha256': sample_sha256}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the tables → input manifest, then encode\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks "
                "`prepare_regression_table` applies — unique column names, the target present, coerced to finite floats "
                "with the dropped-row count recorded, at least `MIN_TRAIN_ROWS` support rows (`MIN_EVAL_ROWS` for a "
                "holdout), at most `MAX_TRAIN_ROWS` rows and `MAX_FEATURES` columns, a target that varies — and returns "
                "an **input manifest** naming the schema, the observed structure (categorical columns, missing values, a "
                "target summary) and the verdict. It is written to `outputs/{stem}_input_manifest.json`. To show what "
                "rejection looks like, the cell also validates a probe with too few rows and records the package's own "
                "error message as a finding. The ceilings are printed before any model runs.\n\n"
                "The holdout and independent test partitions are then aligned to the support schema, the categorical "
                "encoder is fitted on the support split only and applied everywhere (unseen values counted), and the "
                "trivial training-mean baseline is computed with `training_mean_baseline`. Everything in Section 6 "
                "should be read against that baseline."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_TRAIN_ROWS': MIN_TRAIN_ROWS, 'MIN_EVAL_ROWS': MIN_EVAL_ROWS, 'MAX_TRAIN_ROWS': MAX_TRAIN_ROWS, 'MAX_FEATURES': MAX_FEATURES}}}})\n"
                "input_manifest = validate_inputs(train_data, target_column=TARGET_COLUMN, names=[data_name + ':train'])\n"
                "holdout_manifest = validate_inputs(holdout_data, target_column=TARGET_COLUMN, min_rows=MIN_EVAL_ROWS, names=[data_name + ':holdout'])\n"
                "input_manifest['inputs'].extend(holdout_manifest['inputs'])\n"
                "if test_data is not None:\n"
                "    input_manifest['inputs'].extend(validate_inputs(test_data, target_column=TARGET_COLUMN, min_rows=MIN_EVAL_ROWS, names=[data_name + ':test'])['inputs'])\n"
                "# Demonstrate rejection on a probe that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(train_data.head(MIN_TRAIN_ROWS - 1), target_column=TARGET_COLUMN)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'too-few-rows-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest['inputs'][0], indent=2))\n"
                "print('findings:', input_manifest['findings'])\n\n"
                "train_data, dropped_train = prepare_regression_table(train_data, TARGET_COLUMN)\n"
                "holdout_data, dropped_holdout = prepare_regression_table(holdout_data, TARGET_COLUMN, min_rows=MIN_EVAL_ROWS)\n"
                "dropped_test = 0\n"
                "if test_data is not None:\n"
                "    test_data, dropped_test = prepare_regression_table(test_data, TARGET_COLUMN, min_rows=MIN_EVAL_ROWS)\n"
                "FEATURE_COLUMNS = [column for column in train_data.columns if column != TARGET_COLUMN]\n"
                "train_data = train_data[FEATURE_COLUMNS + [TARGET_COLUMN]].reset_index(drop=True)\n"
                "holdout_data = align_to_schema(holdout_data, FEATURE_COLUMNS, TARGET_COLUMN)\n"
                "test_data = align_to_schema(test_data, FEATURE_COLUMNS, TARGET_COLUMN) if test_data is not None else None\n"
                "CATEGORICAL_ENCODERS = fit_categorical_encoder(train_data, FEATURE_COLUMNS)\n"
                "train_encoded, _ = apply_categorical_encoder(train_data, CATEGORICAL_ENCODERS)\n"
                "holdout_encoded, unseen_holdout = apply_categorical_encoder(holdout_data, CATEGORICAL_ENCODERS)\n"
                "test_encoded, unseen_test = (apply_categorical_encoder(test_data, CATEGORICAL_ENCODERS) if test_data is not None else (None, {{}}))\n"
                "baseline = training_mean_baseline(train_encoded[TARGET_COLUMN], holdout_encoded[TARGET_COLUMN])\n"
                "print({{'dropped_non_finite_target_rows': {{'train': dropped_train, 'holdout': dropped_holdout, 'test': dropped_test}}, 'unseen_categorical_values': {{'holdout': unseen_holdout, 'test': unseen_test}}, 'encoded_categoricals': sorted(CATEGORICAL_ENCODERS)}})\n"
                "print({{'train': len(train_encoded), 'holdout': len(holdout_encoded), 'test': 0 if test_encoded is None else len(test_encoded), 'features': len(FEATURE_COLUMNS)}})\n"
                "print('training-mean baseline on the holdout', baseline)"
            ),
        },
        {
            "md": (
                "## 6. Evaluate pretrained TabICLv2, then optionally fine-tune\n\n"
                "`pipe.fit(X_train, y_train)` does not train anything; it registers the support rows as the model's "
                "context (the checkpoint pinned in Section 3 is deserialised by `tabicl` here). `n_estimators=8` "
                "averages eight passes with different feature/row permutations. `regression_metrics` scores the holdout "
                "and, when present, the independent test partition: MAE and RMSE in target units, R² relative to a "
                "constant-mean reference, plus Pearson r. Reading one of them alone hides the others' failure modes.\n\n"
                "**Fine-tuning gate (off by default; CUDA only).** With `RUN_FINE_TUNING=True`, `create_finetuned_regressor` "
                "runs the upstream gradient fine-tuning with early stopping on the holdout and `EVAL_METRIC`, the best "
                "checkpoint is reloaded into an ordinary regressor with the same inference ensemble for a fair "
                "comparison, and the candidate replaces the pretrained model **only** if it beats it on the holdout "
                "(`compare_metric`) and the holdout has at least `MIN_SELECTION_HOLDOUT_ROWS` rows. The independent test "
                "is evidence only; a worse test result is surfaced as a warning and never changes the selection.\n\n"
                "**Fine-tuning disk usage.** TabICL writes an epoch checkpoint per epoch, so temporary disk use scales with "
                "`FINE_TUNE_EPOCHS`; after the best one is loaded and evaluated, the cell deletes the others and keeps "
                "`best.ckpt` only. The fine-tuned `best.ckpt` may remain larger than the base checkpoint because upstream "
                "training state can be embedded; the notebook preserves the upstream checkpoint format for checkpoint "
                "compatibility rather than rewriting serialised state.\n\n"
                "**Reproducibility boundary.** The tutorial fixes the data split seed and the TabICL ensemble seed. Those "
                "controls make the demonstrated partitioning and estimator configuration repeatable, but they do not "
                "promise bitwise-identical floating-point results across devices, library builds or kernel choices."
            ),
            "code": (
                "import shutil\n\n"
                "RUN_FINE_TUNING = False  # @param {{type:\"boolean\"}}\n"
                "EVAL_METRIC = 'mae'  # @param [\"mae\", \"mse\", \"r2\"]\n"
                "FINE_TUNE_EPOCHS, FINE_TUNE_TIME_LIMIT, FINE_TUNE_PATIENCE = 10, 600, 3\n"
                "MIN_SELECTION_HOLDOUT_ROWS = 50\n"
                "if EVAL_METRIC not in {{'mae', 'mse', 'r2'}}:\n"
                "    raise ValueError(f'Unsupported EVAL_METRIC: {{EVAL_METRIC}}')\n"
                "X_train, y_train = train_encoded[FEATURE_COLUMNS], train_encoded[TARGET_COLUMN]\n\n"
                "def score(model, frame):\n"
                "    return regression_metrics(frame[TARGET_COLUMN].to_numpy(dtype=float), model.predict(frame[FEATURE_COLUMNS]))\n\n"
                "pipe.fit(X_train, y_train)\n"
                "pretrained_metrics = score(pipe, holdout_encoded)\n"
                "pretrained_test_metrics = score(pipe, test_encoded) if test_encoded is not None else None\n"
                "if not math.isfinite(pretrained_metrics[EVAL_METRIC]):\n"
                "    raise ValueError(f'{{EVAL_METRIC}} is unavailable on the holdout; choose another selection metric or provide a holdout with sufficient target variation')\n"
                "print('pretrained holdout', pretrained_metrics)\n"
                "if pretrained_test_metrics:\n"
                "    print('pretrained independent test', pretrained_test_metrics)\n\n"
                "candidate = candidate_metrics = candidate_test_metrics = candidate_checkpoint = None\n"
                "if RUN_FINE_TUNING:\n"
                "    if not torch.cuda.is_available():\n"
                "        raise RuntimeError('TabICLv2 fine-tuning requires CUDA')\n"
                "    ft_dir = Path('outputs') / 'finetune'\n"
                "    if ft_dir.exists():\n"
                "        shutil.rmtree(ft_dir)\n"
                "    finetuner = create_finetuned_regressor(epochs=FINE_TUNE_EPOCHS, learning_rate=1e-5, weight_decay=0.01, n_estimators_finetune=1, n_estimators_validation=1, n_estimators_inference=4, early_stopping=True, patience=FINE_TUNE_PATIENCE, time_limit=FINE_TUNE_TIME_LIMIT, eval_metric=EVAL_METRIC, model_path=str(pipe.model_path), allow_auto_download=False, device='cuda', random_state=RANDOM_SEED, verbose=True)\n"
                "    fine_tune_regressor(finetuner, X_train, y_train, X_val=holdout_encoded[FEATURE_COLUMNS], y_val=holdout_encoded[TARGET_COLUMN], output_dir=str(ft_dir))\n"
                "    candidate_checkpoint = ft_dir / 'best.ckpt'\n"
                "    if not candidate_checkpoint.exists():\n"
                "        raise RuntimeError('Fine-tuning did not produce best.ckpt')\n"
                "    # Reload into the ordinary regressor with the pretrained model's inference ensemble for a fair comparison.\n"
                "    candidate = TabICLRegressionPipeline(create_regressor(model_path=candidate_checkpoint, allow_auto_download=False, n_estimators=pipe.n_estimators, random_state=RANDOM_SEED, device=pipe.device), model_path=candidate_checkpoint, n_estimators=pipe.n_estimators, random_state=RANDOM_SEED, device=pipe.device, source='fine-tuned')\n"
                "    candidate.fit(X_train, y_train)\n"
                "    candidate_metrics = score(candidate, holdout_encoded)\n"
                "    candidate_test_metrics = score(candidate, test_encoded) if test_encoded is not None else None\n"
                "    print('candidate holdout', candidate_metrics)\n"
                "    for checkpoint in [c for c in ft_dir.rglob('*.ckpt') if c.resolve() != candidate_checkpoint.resolve()]:\n"
                "        checkpoint.unlink()\n\n"
                "ACTIVE_MODEL, ACTIVE_CHECKPOINT_PATH, ACTIVE_MODE = pipe, pipe.model_path, 'pretrained'\n"
                "SELECTION_BASIS = 'default:pretrained'\n"
                "if candidate is not None:\n"
                "    if len(holdout_encoded) < MIN_SELECTION_HOLDOUT_ROWS:\n"
                "        SELECTION_BASIS = f'default:pretrained; holdout-too-small:{{len(holdout_encoded)}}<{{MIN_SELECTION_HOLDOUT_ROWS}}'\n"
                "    else:\n"
                "        SELECTION_BASIS = f'holdout:{{EVAL_METRIC}}'\n"
                "        if compare_metric(candidate_metrics, pretrained_metrics, EVAL_METRIC):\n"
                "            ACTIVE_MODEL, ACTIVE_CHECKPOINT_PATH, ACTIVE_MODE = candidate, candidate_checkpoint, 'fine-tuned'\n"
                "    if candidate_test_metrics and pretrained_test_metrics:\n"
                "        degraded = [name for name in ('mae', 'mse', 'rmse') if candidate_test_metrics[name] > pretrained_test_metrics[name]]\n"
                "        degraded += [name for name in ('r2', 'pearsonr') if math.isfinite(candidate_test_metrics[name]) and math.isfinite(pretrained_test_metrics[name]) and candidate_test_metrics[name] < pretrained_test_metrics[name]]\n"
                "        if degraded:\n"
                "            print('WARNING independent-test metrics worsened for the candidate:', degraded, '(evidence only; never used for selection)')\n"
                "active_metrics = candidate_metrics if ACTIVE_MODE == 'fine-tuned' else pretrained_metrics\n"
                "active_test_metrics = candidate_test_metrics if ACTIVE_MODE == 'fine-tuned' else pretrained_test_metrics\n"
                "print({{'recommended_for_export': ACTIVE_MODE, 'selection_basis': SELECTION_BASIS, 'source': ACTIVE_MODEL.source, 'device': ACTIVE_MODEL.device}})"
            ),
        },
        {
            "md": (
                "## 7. Classical tree baselines and in-memory blending, then the evaluation report\n\n"
                "LightGBM and Random Forest are fitted on the exact same encoded support rows and scored on the exact same "
                "holdout and test partitions, so the comparison is fair (EVAL15). A convex blend of TabICLv2 and "
                "LightGBM is then chosen by minimising holdout RMSE over a 101-point grid; it is evaluated in memory only, "
                "with a paired standard error on the squared-error difference, and the exported bundle in Section 9 is "
                "the unchanged active model. Latencies are medians of five warmed runs on this runtime.\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. Here it "
                "carries the active model's holdout metrics (`mae`, `mse`, `rmse`, `r2`, `pearsonr` — the repository's own "
                "metric ids), the independent-test metrics when a test partition exists, and the training-mean baseline, "
                "with the verdict `sample-sanity`: one seeded split with no dispersion estimate, tutorial evidence rather "
                "than a benchmark. Without a labelled holdout the verdict would be `not-measurable`. The report is written "
                "to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "import time\n\n"
                "from lightgbm import LGBMRegressor\n"
                "from sklearn.ensemble import RandomForestRegressor\n\n"
                "lgbm_model = LGBMRegressor(random_state=RANDOM_SEED, n_estimators=100, verbose=-1).fit(X_train, y_train)\n"
                "rf_model = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=100).fit(X_train, y_train)\n\n"
                "def timed_predict(model, frame, repeats=5):\n"
                "    X = frame[FEATURE_COLUMNS]\n"
                "    if repeats > 1:\n"
                "        model.predict(X)  # discarded warm-up call\n"
                "    latencies = []\n"
                "    for _ in range(repeats):\n"
                "        t0 = time.perf_counter()\n"
                "        preds = np.asarray(model.predict(X), dtype=float)\n"
                "        latencies.append((time.perf_counter() - t0) * 1000.0)\n"
                "    return preds, float(np.median(latencies))\n\n"
                "y_holdout = holdout_encoded[TARGET_COLUMN].to_numpy(dtype=float)\n"
                "n_holdout = len(holdout_encoded)\n"
                "pred_tabicl_holdout, lat_tabicl = timed_predict(ACTIVE_MODEL, holdout_encoded)\n"
                "pred_lgbm_holdout, lat_lgbm = timed_predict(lgbm_model, holdout_encoded)\n"
                "pred_rf_holdout, lat_rf = timed_predict(rf_model, holdout_encoded)\n"
                "classical = {{'tabicl_' + ACTIVE_MODE: regression_metrics(y_holdout, pred_tabicl_holdout), 'lightgbm': regression_metrics(y_holdout, pred_lgbm_holdout), 'random_forest': regression_metrics(y_holdout, pred_rf_holdout)}}\n"
                "for name, values in classical.items():\n"
                "    print(f\"{{name:<20}} rmse={{values['rmse']:.4f}} mae={{values['mae']:.4f}} r2={{values['r2']:.4f}}\")\n"
                "print({{'latency_ms_median_of_5': {{'tabicl': round(lat_tabicl, 2), 'lightgbm': round(lat_lgbm, 2), 'random_forest': round(lat_rf, 2)}}, 'rows': n_holdout, 'device': ACTIVE_MODEL.device}})\n\n"
                "best_w, best_rmse = 1.0, float('inf')\n"
                "for w in np.linspace(0.0, 1.0, 101):\n"
                "    blend_rmse = regression_metrics(y_holdout, w * pred_tabicl_holdout + (1.0 - w) * pred_lgbm_holdout)['rmse']\n"
                "    if blend_rmse < best_rmse:\n"
                "        best_rmse, best_w = blend_rmse, float(w)\n"
                "pred_blend_holdout = best_w * pred_tabicl_holdout + (1.0 - best_w) * pred_lgbm_holdout\n"
                "blend_holdout = regression_metrics(y_holdout, pred_blend_holdout)\n"
                "se_delta = (y_holdout - pred_blend_holdout) ** 2 - (y_holdout - pred_tabicl_holdout) ** 2\n"
                "paired_se = float(np.std(se_delta, ddof=1) / np.sqrt(n_holdout)) if n_holdout > 1 else 0.0\n"
                "print({{'blend_objective': 'minimise holdout RMSE', 'weight_tabicl': best_w, 'blend_holdout': blend_holdout, 'mse_delta_vs_tabicl': float(np.mean(se_delta)), 'paired_se': paired_se}})\n"
                "blend_test = None\n"
                "if test_encoded is not None:\n"
                "    y_test = test_encoded[TARGET_COLUMN].to_numpy(dtype=float)\n"
                "    pred_tabicl_test, _ = timed_predict(ACTIVE_MODEL, test_encoded, repeats=1)\n"
                "    pred_lgbm_test, _ = timed_predict(lgbm_model, test_encoded, repeats=1)\n"
                "    blend_test = regression_metrics(y_test, best_w * pred_tabicl_test + (1.0 - best_w) * pred_lgbm_test)\n"
                "    print({{'independent_test': {{'tabicl': regression_metrics(y_test, pred_tabicl_test), 'lightgbm': regression_metrics(y_test, pred_lgbm_test), 'blend': blend_test}}}})\n"
                "    if blend_holdout['rmse'] < classical['tabicl_' + ACTIVE_MODE]['rmse'] and blend_test['rmse'] >= regression_metrics(y_test, pred_tabicl_test)['rmse']:\n"
                "        print('WARNING mixed evidence: the blend improved holdout RMSE but not independent-test RMSE; treat the holdout gain as selection-biased.')\n"
                "else:\n"
                "    print('No independent test partition; the holdout blend score is selection-biased demonstration evidence.')\n\n"
                "report = evaluation_report(active_metrics, baseline=baseline, independent_test=active_test_metrics, n_holdout=n_holdout, n_test=None if test_encoded is None else len(test_encoded), target_column=TARGET_COLUMN, selection=SELECTION_BASIS, sample_kind=sample_kind, estimation='single seeded random split (support/holdout/independent test); no dispersion estimate')\n"
                "report['classical_baselines_holdout'] = {{name: values for name, values in classical.items() if not name.startswith('tabicl_')}}\n"
                "report['blend_holdout'] = {{'weight_tabicl': best_w, **blend_holdout}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{key: report[key] for key in ('verdict', 'reason', 'selection', 'n_holdout', 'n_test')}}, indent=2))"
            ),
        },
        {
            "md": (
                "## 8. Optional new-data point prediction\n\n"
                "Off by default so a top-to-bottom run needs no upload dialog. Switch `RUN_NEW_DATA_INFERENCE` on and upload "
                "one CSV with the support feature columns (order does not matter; extra columns are preserved in the output "
                "and not passed to the model), or set `NEW_DATA_PATH` for a non-interactive executor. `validate_inputs(..., "
                "target_column=None, feature_columns=...)` applies exactly the checks `read_inference_csv` applies — unique "
                "header, no pre-existing `prediction` column, every feature present — and the same fitted categorical "
                "encoder is applied (unseen values counted). Predictions are **continuous point estimates only**."
            ),
            "code": (
                "RUN_NEW_DATA_INFERENCE = False  # @param {{type:\"boolean\"}}\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "new_data_result = None\n"
                "if RUN_NEW_DATA_INFERENCE:\n"
                "    if NEW_DATA_PATH:\n"
                "        input_name, payload = os.path.basename(NEW_DATA_PATH), Path(NEW_DATA_PATH).read_bytes()\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        new_upload = files.upload()\n"
                "        if len(new_upload) != 1:\n"
                "            raise ValueError('Upload exactly one CSV')\n"
                "        input_name, payload = next(iter(new_upload.items()))\n"
                "    rows = read_inference_csv(payload, FEATURE_COLUMNS)\n"
                "    inference_manifest = validate_inputs(rows, None, feature_columns=FEATURE_COLUMNS, names=[input_name])\n"
                "    X_new, unseen_new = apply_categorical_encoder(rows[FEATURE_COLUMNS], CATEGORICAL_ENCODERS)\n"
                "    out = rows.copy()\n"
                "    out['prediction'] = ACTIVE_MODEL.predict(X_new)\n"
                "    out.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "    new_data_result = {{'input': input_name, 'rows': len(out), 'unseen_categorical_values': unseen_new, 'input_manifest': inference_manifest}}\n"
                "    print(out.head())\n"
                "else:\n"
                "    # Sample path: score eight held-out rows so a prediction CSV always exists.\n"
                "    smoke = holdout_encoded[FEATURE_COLUMNS].head(8)\n"
                "    out = holdout_data[FEATURE_COLUMNS].head(8).copy()\n"
                "    out['prediction'] = ACTIVE_MODEL.predict(smoke)\n"
                "    out.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "    print('Inference upload skipped; eight held-out rows scored instead.')\n"
                "    print(out)"
            ),
        },
        {
            "md": (
                "## 9. Export a DIMER-style serving bundle, then prove a fresh reload\n\n"
                "The ZIP carries the minimum serving contract the DIMER pipeline expects: `checkpoints/best.ckpt`, "
                "`training_context.parquet`, and `artifact.json`. The training context is *required*: TabICL is still an "
                "in-context learner at serve time, so whoever loads the bundle must hand the model the same rows you "
                "evaluated with (ART3), and the context inherits the source data's confidentiality, licensing, retention "
                "and disclosure obligations (ART7). `artifact.json` records the feature and target columns, the base-model "
                "identity and digest, the selected mode and its basis, the inference settings including the fitted "
                "categorical encoders, per-file sizes and SHA-256 digests, the payload allowlist and the producer runtime.\n\n"
                "Before calling the ZIP reusable, the cell extracts it into a fresh directory with `safe_extract_zip` "
                "(member-by-member, after path, symlink and expanded-size checks — the same function the companion "
                "notebook applies), verifies the allowlist, sizes and digests with `verify_artifact_bundle`, rebuilds the "
                "regressor from the bundle alone (checkpoint + context + recorded settings), and checks that its "
                "predictions agree with the in-memory model's on eight held-out rows (VER1–VER5). The result JSON then "
                "records everything: predictions, metrics, the evaluation report, the input manifest, the sample digest, "
                "the notebook's source, the model identity, revision and licence, and the runtime identity."
            ),
            "code": (
                "ARTIFACT_DIR = Path('outputs') / 'artifact'\n"
                "if ARTIFACT_DIR.exists():\n"
                "    shutil.rmtree(ARTIFACT_DIR)\n"
                "(ARTIFACT_DIR / 'checkpoints').mkdir(parents=True)\n"
                "export_ckpt = ARTIFACT_DIR / 'checkpoints' / 'best.ckpt'\n"
                "shutil.copy2(ACTIVE_CHECKPOINT_PATH, export_ckpt)\n"
                "context_path = ARTIFACT_DIR / 'training_context.parquet'\n"
                "train_encoded[FEATURE_COLUMNS + [TARGET_COLUMN]].to_parquet(context_path, index=False)\n"
                "manifest = {{\n"
                "    'artifactFormat': ARTIFACT_FORMAT,\n"
                "    'checkpoint': 'checkpoints/best.ckpt', 'trainingContext': 'training_context.parquet',\n"
                "    'targetColumn': TARGET_COLUMN, 'featureColumns': FEATURE_COLUMNS,\n"
                "    'baseCheckpoint': BASE_CHECKPOINT_NAME, 'baseModelRevision': MODEL_REVISION, 'baseModelSha256': BASE_MODEL_SHA256,\n"
                "    'tabiclVersion': importlib.metadata.version('tabicl'), 'mode': ACTIVE_MODE, 'selectionBasis': SELECTION_BASIS,\n"
                "    'metrics': {{'selectionMetric': EVAL_METRIC, 'pretrainedHoldout': pretrained_metrics, 'fineTunedHoldout': candidate_metrics, 'pretrainedIndependentTest': pretrained_test_metrics, 'fineTunedIndependentTest': candidate_test_metrics}},\n"
                "    'inference': {{'class': 'TabICLRegressor', 'modelPath': 'checkpoints/best.ckpt', 'nEstimators': ACTIVE_MODEL.n_estimators, 'randomState': RANDOM_SEED, 'allowAutoDownload': False, 'categoricalEncoders': CATEGORICAL_ENCODERS}},\n"
                "    'digests': {{'checkpointSha256': sha256_file(export_ckpt), 'trainingContextSha256': sha256_file(context_path)}},\n"
                "    'sizes': {{'checkpoint': export_ckpt.stat().st_size, 'trainingContext': context_path.stat().st_size}},\n"
                "    'payloadFiles': ['checkpoints/best.ckpt', 'training_context.parquet'],\n"
                "    'runtime': {{'pythonVersion': platform.python_version(), 'torchVersion': torch.__version__, 'pandasVersion': pandas.__version__, 'pyarrowVersion': importlib.metadata.version('pyarrow'), 'scikitLearnVersion': sklearn.__version__, 'device': ACTIVE_MODEL.device}},\n"
                "    'notebookSource': NOTEBOOK_SOURCE,\n"
                "}}\n"
                "(ARTIFACT_DIR / 'artifact.json').write_text(json.dumps(manifest, indent=2) + '\\n', encoding='utf-8')\n"
                "archive_path = Path(shutil.make_archive(str(Path('outputs') / '{stem}_artifact'), 'zip', root_dir=ARTIFACT_DIR))\n"
                "print({{'artifact_zip': str(archive_path), 'zip_sha256': sha256_file(archive_path)}})\n\n"
                "RELOAD_DIR = Path('outputs') / 'artifact-reload'\n"
                "if RELOAD_DIR.exists():\n"
                "    shutil.rmtree(RELOAD_DIR)\n"
                "reload_root = safe_extract_zip(archive_path, RELOAD_DIR)\n"
                "served = json.loads((reload_root / 'artifact.json').read_text(encoding='utf-8'))\n"
                "members = verify_artifact_bundle(reload_root, served)\n"
                "context = pd.read_parquet(members['training_context'])\n"
                "reloaded = TabICLRegressionPipeline(create_regressor(model_path=members['checkpoint'], allow_auto_download=False, n_estimators=served['inference']['nEstimators'], random_state=served['inference']['randomState'], device=ACTIVE_MODEL.device), model_path=members['checkpoint'], n_estimators=served['inference']['nEstimators'], random_state=served['inference']['randomState'], device=ACTIVE_MODEL.device, source='artifact')\n"
                "reloaded.fit(context[served['featureColumns']], context[served['targetColumn']])\n"
                "smoke_rows = holdout_encoded[FEATURE_COLUMNS].iloc[:min(8, len(holdout_encoded))]\n"
                "np.testing.assert_allclose(ACTIVE_MODEL.predict(smoke_rows), reloaded.predict(smoke_rows), rtol=1e-5, atol=1e-7)\n"
                "print('PASS: bundle extracted safely, digests verified, regressor rebuilt from the bundle alone; predictions equivalent (rtol=1e-5, atol=1e-7).')\n\n"
                "payload = {{\n"
                "    'predictions': out.to_dict(orient='records'),\n"
                "    'new_data': new_data_result,\n"
                "    'metrics': {{'active_mode': ACTIVE_MODE, 'holdout': active_metrics, 'independent_test': active_test_metrics, 'pretrained_holdout': pretrained_metrics, 'candidate_holdout': candidate_metrics}},\n"
                "    'training_mean_baseline': baseline,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': data_name, 'source': DATA_SOURCE, 'csv_sha256': sample_sha256, 'train_rows': len(train_encoded), 'holdout_rows': n_holdout, 'test_rows': 0 if test_encoded is None else len(test_encoded)}},\n"
                "    'artifact': {{'zip': archive_path.name, 'zip_sha256': sha256_file(archive_path), 'manifest': manifest}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'tabicl': importlib.metadata.version('tabicl'), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'device': ACTIVE_MODEL.device}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Predictions are continuous point estimates in the target's units with no uncertainty interval; any tolerance "
        "band must be chosen on the caller's own labelled, domain-representative data. The evaluation report's "
        "`sample-sanity` verdict names what it is: one seeded split of a public sample with no dispersion estimate — "
        "tutorial evidence that must not be generalised. On the diabetes sample a development run of the previous "
        "notebook revision gave pretrained holdout MAE ≈ 39 (test ≈ 45), RMSE ≈ 49 (test ≈ 55) and R² ≈ 0.58 (test ≈ 0.49) "
        "against a training-mean baseline MAE ≈ 67.5; with 88 and 89 rows the holdout/test gap is mostly sampling noise, "
        "which is why the independent test exists and never drives selection. The blend weight is chosen on the holdout "
        "and is therefore selection-biased; the classical baselines show when the foundation model adds value and what "
        "it costs in latency. Rows that are not independent, targets outside the support range, unseen categories and "
        "support sets near the ceilings all change results in ways these metrics do not measure.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, validate the demonstrated tables, condition on regression "
        "support rows, compute sample metrics against trivial and classical baselines, write the input manifest and the "
        "evaluation report, export a DIMER-style serving bundle and rebuild an equivalent regressor from that bundle "
        "alone — without the repository being reachable. It does **not** establish benchmark superiority, domain "
        "generalisation, fairness, robustness, calibration, production safety, or deployment fitness.\n\n"
        "**Next experiments:** switch `DATA_SOURCE` to `Upload pre-split train/val/test` with your own partitions; enable "
        "`RUN_FINE_TUNING` on a CUDA runtime and watch the holdout-based selection and the independent-test warning; "
        "raise `n_estimators` in Section 3's load expression and compare latency against LightGBM; feed the exported "
        "`outputs/{stem}_artifact.zip` to the companion artifact-inference notebook in a separate session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/soda-inria/tabicl\n"
        "- TabICLv2 paper: https://arxiv.org/abs/2602.11139\n"
        "- TabICL paper: https://arxiv.org/abs/2502.05564"
    ),
}
