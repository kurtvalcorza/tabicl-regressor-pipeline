"""Stage runner for the standalone TabICLv2 regressor E2E tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process and starts from files only: the
verified checkpoint under ``--weights``, the partitions, encoders and records of earlier stages under
``RUN_DIR/state``, and the learner-facing exports under ``--outputs``. In-context conditioning (``fit``) registers the
support rows and is repeated by every stage that needs the model; a fine-tuned candidate is kept as a checkpoint file.
On failure a stage writes ``RUN_DIR/state/<stage>.error.json``, which the notebook re-raises in the kernel.

Stages: weights → data → validate → condition → report → predict → export → reload, plus the optional ``activity``.
The ``data`` and ``validate`` stages import no model library, so CI exercises them directly.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

STEM = "tabiclv2_regressor"
PACKAGE = "tabicl_regressor_pipeline"
SEED = 42
VALIDATION_SPLIT = 0.20
NEW_ROWS = 8
RELOAD_RTOL, RELOAD_ATOL = 1e-5, 1e-7
SAMPLE_SOURCES = ("Sample: Diabetes", "Sample: California Housing")
UPLOAD_SOURCES = ("Upload CSV", "Upload pre-split train/val/test")
CALIFORNIA_ROWS = 2000
NUMERIC_LIKE_SHARE = 0.9  # an object column whose values are >= 90 % numbers is treated as a numeric column with typos
MAX_BLANK_TARGET_SHARE = 0.2  # more blank targets than this share of a file is refused instead of dropped
THOUSANDS = r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$"
FINE_TUNE = {"epochs": 3, "time_limit": 300, "patience": 2, "eval_metric": "mae"}
MIN_SELECTION_HOLDOUT_ROWS = 50
SAMPLE_BUNDLE_RECORD = "sample-bundle/SAMPLE_BUNDLE.json"  # carried: the pinned sample bundle's provenance


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_frame(self, name: str, frame) -> None:
        frame.to_parquet(self.state / name, index=True, engine="pyarrow")

    def read_frame(self, name: str, needed_by: str):
        import pandas as pd

        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return pd.read_parquet(path, engine="pyarrow")

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


# --------------------------------------------------------------------------------------------------
# data (TIR-M4, TIR-m1)
# --------------------------------------------------------------------------------------------------


def random_60_20_20(frame):
    from sklearn.model_selection import train_test_split

    train, remainder = train_test_split(frame, test_size=0.4, random_state=SEED)
    holdout, test = train_test_split(remainder, test_size=0.5, random_state=SEED)
    return train, holdout, test


def load_sample(P, source: str):
    """(train, holdout, test, name, target) for one of the two public samples."""
    if source == "Sample: Diabetes":
        from sklearn.datasets import load_diabetes

        dataset = load_diabetes(as_frame=True)
        frame = dataset.frame.rename(columns={dataset.target.name: "target"})
        return (*random_60_20_20(frame), "sklearn-diabetes", "target")
    if source == "Sample: California Housing":
        from sklearn.datasets import fetch_california_housing

        dataset = fetch_california_housing(as_frame=True)  # downloads from figshare through scikit-learn on first use
        frame = dataset.frame.rename(columns={dataset.target.name: "MedHouseVal"})
        if len(frame) > CALIFORNIA_ROWS:
            frame = frame.sample(n=CALIFORNIA_ROWS, random_state=SEED).reset_index(drop=True)
        return (*random_60_20_20(frame), "sklearn-california-housing-2000", "MedHouseVal")
    raise ValueError(f"DATA_SOURCE must be one of {list(SAMPLE_SOURCES + UPLOAD_SOURCES)}, got {source!r}")


def check_target(frame, name: str, target: str) -> int:
    """TIR-M4: blank targets are dropped and counted (refused above MAX_BLANK_TARGET_SHARE); a target that is present
    but not a finite number is refused naming the file, the column and example values. Thousands separators are
    refused with that diagnosis rather than parsed, so a value is never reinterpreted silently. Returns the blank count."""
    import numpy as np
    import pandas as pd

    raw = frame[target]
    text = raw.astype("string").str.strip()
    blank = (raw.isna() | (text == "")).fillna(True).astype(bool)
    values = pd.to_numeric(text.where(~blank), errors="coerce").astype(float)
    bad = (~blank) & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    if bad.any():
        examples = raw[bad].astype(str).unique().tolist()[:5]
        lines = (np.flatnonzero(bad.to_numpy()) + 2)[:5].tolist()
        if text[bad].str.match(THOUSANDS).all():
            raise ValueError(f"{name}: the target column {target!r} has {int(bad.sum())} value(s) written with thousands separators, e.g. {examples} (file lines {lines}). Remove the separators (1510.0, not 1,510.0) and run again; they are refused rather than guessed.")
        raise ValueError(f"{name}: the target column {target!r} has {int(bad.sum())} value(s) that are not finite numbers, e.g. {examples} (file lines {lines}). A regression target must be numeric; fix those values or leave the cells empty to drop the rows.")
    n_blank = int(blank.sum())
    if n_blank > MAX_BLANK_TARGET_SHARE * len(frame):
        raise ValueError(f"{name}: {n_blank} of {len(frame)} rows have a blank target {target!r}, more than {MAX_BLANK_TARGET_SHARE:.0%}; dropping them would change the data too much. Fill or remove them first.")
    return n_blank


def check_byod_table(frame, name: str, target: str, text_columns: list[str]) -> int:
    """Refuse, naming the file and the rule, a missing target column, an unparseable target (TIR-M4) and numeric
    columns spoiled by a few strings (they would otherwise be ordinal-encoded silently by sorted string value).
    Returns the number of blank-target rows that will be dropped and counted."""
    import pandas as pd

    if target not in frame.columns:
        raise ValueError(f"{name}: the target column {target!r} is not in the header {list(map(str, frame.columns))}. Set TARGET_COLUMN to the name of your label column.")
    blank = check_target(frame, name, target)
    absent = [c for c in text_columns if c not in frame.columns]
    if absent:
        raise ValueError(f"{name}: TEXT_COLUMNS {absent} are not in the header {list(map(str, frame.columns))}.")
    for column in frame.columns:
        if column == target or column in text_columns or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        numeric = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
        share = float(numeric.notna().mean())
        if share >= NUMERIC_LIKE_SHARE:
            bad = values[numeric.isna()].astype(str).unique().tolist()[:5]
            raise ValueError(
                f"{name}: column {column!r} is numeric except for {int(numeric.isna().sum())} value(s) {bad}; it would be "
                "encoded as categories and lose its numeric meaning. Fix those values (leave missing cells empty), or list "
                "the column in TEXT_COLUMNS if it really is categorical."
            )
    return blank


def clear_outputs(run: Run) -> list[str]:
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    for name in ("artifact", "artifact-reload"):
        if (run.out / name).is_dir():
            shutil.rmtree(run.out / name)
            removed.append(name + "/")
    return removed


def stage_weights(run: Run) -> None:
    P = package(run.root)
    snapshot = run.weights / P.MODEL_KEY
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / P.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / P.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified_files": len(verified["files"]), "revision": verified["revision"], "sha256": [f["sha256"] for f in verified["files"]]})
    run.write_state("weights.json", {"snapshot": str(snapshot), "checkpoint": str(snapshot / P.BASE_CHECKPOINT_NAME)})


def stage_data(run: Run) -> None:
    import pandas as pd

    P = package(run.root)
    opts = run.options
    source = opts.get("data_source", "Sample: Diabetes")
    use_byod = bool(opts.get("use_byod", False))
    if use_byod and source in SAMPLE_SOURCES:
        raise ValueError('USE_BYOD=True requires DATA_SOURCE = "Upload CSV" or "Upload pre-split train/val/test".')
    if source in UPLOAD_SOURCES and not use_byod:
        raise ValueError("Set USE_BYOD=True to use an upload DATA_SOURCE.")
    removed = clear_outputs(run)
    for name in ("train.parquet", "holdout.parquet", "test.parquet", "data.json", "encoded.json", "condition.json", "export_reference.json"):
        (run.state / name).unlink(missing_ok=True)
    if removed:
        print({"removed_previous_outputs": removed})
    text_columns = list(opts.get("text_columns") or [])
    dropped: dict[str, int] = {}
    if source in SAMPLE_SOURCES:
        train, holdout, test, name, target = load_sample(P, source)
        kind = "sample"
    else:
        target = opts.get("target_column") or "target"
        paths = opts.get("paths") or {}
        def read(key: str):
            path = Path(paths.get(key) or "")
            if not str(path) or not path.is_file():
                raise FileNotFoundError(f"BYOD {key} file {str(path)!r} does not exist in this runtime: upload it or correct the path field.")
            if path.suffix.lower() != ".csv":
                raise ValueError(f"{path.name}: BYOD files must be CSV.")
            frame = P.read_csv_payload(path.read_bytes(), path.name)
            blank = check_byod_table(frame, path.name, target, text_columns)
            if blank:
                dropped[key] = blank
            return frame, path.name
        if source == "Upload CSV":
            frame, name = read("csv")
            raw_manifest = P.validate_inputs(frame, target_column=target, names=[name])
            if raw_manifest["inputs"][0]["dropped_non_finite_target_rows"] != dropped.get("csv", 0):
                raise RuntimeError(f"{name}: dropped-row accounting disagrees ({raw_manifest['inputs'][0]['dropped_non_finite_target_rows']} vs {dropped.get('csv', 0)}); report this")
            frame, _ = P.prepare_regression_table(frame, target)
            from sklearn.model_selection import train_test_split

            train, holdout = train_test_split(frame, test_size=VALIDATION_SPLIT, random_state=SEED)
            test = None
        else:
            train, train_name = read("train")
            holdout, _ = read("val")
            test = read("test")[0] if paths.get("test") else None
            name = "pre-split upload (" + train_name + ")"
            dropped = {}  # the pre-split partitions keep their rows until Section 5, which counts the drops per partition
        kind = "BYOD"
    parts = {"train": train, "holdout": holdout, "test": test}
    for key, frame in parts.items():
        if frame is not None:
            run.write_frame(f"{key}.parquet", frame)
    digest = hashlib.sha256(pd.concat([f for f in parts.values() if f is not None]).to_csv(index=False).encode("utf-8")).hexdigest()
    record = {"sample_kind": kind, "name": name, "source": source, "target": target, "train_rows": len(train), "holdout_rows": len(holdout), "test_rows": 0 if test is None else len(test), "csv_sha256": digest, "dropped_missing_target_rows_before_split": dropped, "text_columns": text_columns}
    run.write_state("data.json", record)
    print(record)
    if dropped.get("csv"):
        print(f"{dropped['csv']} row(s) with a blank target were dropped before the split; Section 5's input manifest records them.")


def stage_validate(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "validate")
    target, name = data["target"], data["name"]
    train = run.read_frame("train.parquet", "validate")
    holdout = run.read_frame("holdout.parquet", "validate")
    test = run.read_frame("test.parquet", "validate") if (run.state / "test.parquet").is_file() else None
    print({"ceilings": {"MIN_TRAIN_ROWS": P.MIN_TRAIN_ROWS, "MIN_EVAL_ROWS": P.MIN_EVAL_ROWS, "MAX_TRAIN_ROWS": P.MAX_TRAIN_ROWS, "MAX_FEATURES": P.MAX_FEATURES}, "target": "finite numbers that vary; blank targets dropped and counted"})
    manifest = P.validate_inputs(train, target_column=target, names=[name + ":train"])
    if data["dropped_missing_target_rows_before_split"].get("csv"):
        manifest["inputs"][0]["dropped_missing_target_rows_before_split"] = data["dropped_missing_target_rows_before_split"]["csv"]
    manifest["inputs"].extend(P.validate_inputs(holdout, target_column=target, min_rows=P.MIN_EVAL_ROWS, names=[name + ":holdout"])["inputs"])
    if test is not None:
        manifest["inputs"].extend(P.validate_inputs(test, target_column=target, min_rows=P.MIN_EVAL_ROWS, names=[name + ":test"])["inputs"])
    # Demonstrate rejection on a probe that breaks a ceiling; the finding is recorded, not swallowed.
    try:
        P.validate_inputs(train.head(P.MIN_TRAIN_ROWS - 1), target_column=target)
    except ValueError as exc:
        manifest["findings"].append({"input": "too-few-rows-probe", "verdict": "rejected", "message": str(exc)})
    train_row_ids = list(map(int, train.index)) if train.index.dtype.kind in "iu" else list(range(len(train)))
    holdout_row_ids = list(map(int, holdout.index)) if holdout.index.dtype.kind in "iu" else list(range(len(holdout)))
    train, dropped_train = P.prepare_regression_table(train, target)
    holdout, dropped_holdout = P.prepare_regression_table(holdout, target, min_rows=P.MIN_EVAL_ROWS)
    dropped_test = 0
    if test is not None:
        test, dropped_test = P.prepare_regression_table(test, target, min_rows=P.MIN_EVAL_ROWS)
    features = [c for c in train.columns if c != target]
    train = train[features + [target]].reset_index(drop=True)
    holdout = P.align_to_schema(holdout, features, target)
    test = P.align_to_schema(test, features, target) if test is not None else None
    encoders = P.fit_categorical_encoder(train, features)
    train_enc, _ = P.apply_categorical_encoder(train, encoders)
    holdout_enc, unseen_holdout = P.apply_categorical_encoder(holdout, encoders)
    test_enc, unseen_test = P.apply_categorical_encoder(test, encoders) if test is not None else (None, {})
    baseline = P.training_mean_baseline(train_enc[target], holdout_enc[target])
    run.write_output(f"{STEM}_input_manifest.json", manifest)
    print(json.dumps(manifest["inputs"][0], indent=2))
    print("findings:", manifest["findings"])
    for key, frame in (("train_raw", train), ("holdout_raw", holdout), ("train_enc", train_enc), ("holdout_enc", holdout_enc)):
        run.write_frame(f"{key}.parquet", frame)
    if test is not None:
        run.write_frame("test_enc.parquet", test_enc)
    summary = {"min": float(train[target].min()), "max": float(train[target].max()), "mean": float(train[target].mean())}
    dropped = {"train": dropped_train, "holdout": dropped_holdout, "test": dropped_test, **({"before_split": data["dropped_missing_target_rows_before_split"]["csv"]} if data["dropped_missing_target_rows_before_split"].get("csv") else {})}
    run.write_state("encoded.json", {"features": features, "target": target, "encoders": encoders, "baseline": baseline, "dropped": dropped, "holdout_row_ids": holdout_row_ids if len(holdout_row_ids) == len(holdout) else list(range(len(holdout))), "train_row_ids": train_row_ids, "target_summary": summary})
    print({"dropped_non_finite_target_rows": dropped, "unseen_categorical_values": {"holdout": unseen_holdout, "test": unseen_test}, "encoded_categoricals": sorted(encoders)})
    print({"train": len(train_enc), "holdout": len(holdout_enc), "test": 0 if test_enc is None else len(test_enc), "features": len(features), "training_target": rounded(summary, 2)})
    print("training-mean baseline on the holdout", rounded(baseline))


# --------------------------------------------------------------------------------------------------
# model stages
# --------------------------------------------------------------------------------------------------


def encoded(run: Run, needed_by: str):
    enc = run.read_state("encoded.json", needed_by)
    train = run.read_frame("train_enc.parquet", needed_by)
    holdout = run.read_frame("holdout_enc.parquet", needed_by)
    test = run.read_frame("test_enc.parquet", needed_by) if (run.state / "test_enc.parquet").is_file() else None
    return enc, train, holdout, test


def pretrained(run: Run, n_estimators: int = 8):
    P = package(run.root)
    return P.TabICLRegressionPipeline.from_pretrained(weights_dir=run.weights / P.MODEL_KEY, n_estimators=n_estimators, random_state=SEED)


def from_checkpoint(run: Run, checkpoint: Path, like, source: str):
    P = package(run.root)
    return P.TabICLRegressionPipeline(P.create_regressor(model_path=checkpoint, allow_auto_download=False, n_estimators=like.n_estimators, random_state=SEED, device=like.device), model_path=checkpoint, n_estimators=like.n_estimators, random_state=SEED, device=like.device, source=source)


def active_model(run: Run, enc: dict, train):
    """The model Section 6 selected, conditioned on the support rows."""
    cond = run.read_state("condition.json", "active model")
    model = pretrained(run)
    if cond["active_mode"] == "fine-tuned":
        model = from_checkpoint(run, Path(cond["active_checkpoint"]), model, "fine-tuned")
    model.fit(train[enc["features"]], train[enc["target"]])
    return model, cond


def score(P, model, frame, enc):
    return P.regression_metrics(frame[enc["target"]].to_numpy(dtype=float), model.predict(frame[enc["features"]]))


def stage_condition(run: Run) -> None:
    import torch

    P = package(run.root)
    enc, train, holdout, test = encoded(run, "condition")
    opts = run.options
    run_ft = bool(opts.get("run_fine_tuning", True))
    epochs = int(opts.get("fine_tune_epochs", FINE_TUNE["epochs"]))
    metric = opts.get("eval_metric", FINE_TUNE["eval_metric"])
    if metric not in {"mae", "mse", "r2"}:
        raise ValueError(f"EVAL_METRIC must be mae, mse or r2, got {metric!r}")
    if not 1 <= epochs <= 20:
        raise ValueError(f"FINE_TUNE_EPOCHS must be from 1 to 20, got {epochs}")
    X_train, y_train = train[enc["features"]], train[enc["target"]]
    pipe = pretrained(run)
    pipe.fit(X_train, y_train)
    pretrained_holdout = score(P, pipe, holdout, enc)
    pretrained_test = score(P, pipe, test, enc) if test is not None else None
    print("pretrained holdout", rounded(pretrained_holdout))
    if pretrained_test:
        print("pretrained independent test", rounded(pretrained_test))
    candidate_holdout = candidate_test = None
    active_mode, active_checkpoint, selection = "pretrained", str(pipe.model_path), "default:pretrained"
    fine_tune = {"requested": run_ft, "ran": False}
    if run_ft and not torch.cuda.is_available():
        fine_tune["skipped"] = "no CUDA device: TabICLv2 fine-tuning needs a GPU runtime (RUN7 deviation for this run, recorded here)"
        print("Fine-tuning skipped:", fine_tune["skipped"], "- choose Runtime > Change runtime type > T4 GPU to run the adaptation.")
    elif run_ft and not math.isfinite(pretrained_holdout[metric]):
        fine_tune["skipped"] = f"{metric} is undefined on this holdout, so the candidate could not be selected on it; choose another EVAL_METRIC"
        print("Fine-tuning skipped:", fine_tune["skipped"])
    elif run_ft:
        ft_dir = run.state / "finetune"
        shutil.rmtree(ft_dir, ignore_errors=True)
        finetuner = P.create_finetuned_regressor(epochs=epochs, learning_rate=1e-5, weight_decay=0.01, n_estimators_finetune=1, n_estimators_validation=1, n_estimators_inference=4, early_stopping=True, patience=FINE_TUNE["patience"], time_limit=FINE_TUNE["time_limit"], eval_metric=metric, model_path=str(pipe.model_path), allow_auto_download=False, device="cuda", random_state=SEED, verbose=True)
        P.fine_tune_regressor(finetuner, X_train, y_train, X_val=holdout[enc["features"]], y_val=holdout[enc["target"]], output_dir=str(ft_dir))
        best = ft_dir / "best.ckpt"
        if not best.exists():
            raise RuntimeError("Fine-tuning did not produce best.ckpt")
        for checkpoint in [c for c in ft_dir.rglob("*.ckpt") if c.resolve() != best.resolve()]:
            checkpoint.unlink()  # keep best.ckpt only: TabICL writes one checkpoint per epoch
        candidate = from_checkpoint(run, best, pipe, "fine-tuned")
        candidate.fit(X_train, y_train)
        candidate_holdout = score(P, candidate, holdout, enc)
        candidate_test = score(P, candidate, test, enc) if test is not None else None
        fine_tune.update(ran=True, epochs_requested=epochs, time_limit_seconds=FINE_TUNE["time_limit"], checkpoint_bytes=best.stat().st_size)
        print("candidate holdout", rounded(candidate_holdout))
        if len(holdout) < MIN_SELECTION_HOLDOUT_ROWS:
            selection = f"default:pretrained; holdout-too-small:{len(holdout)}<{MIN_SELECTION_HOLDOUT_ROWS}"
        else:
            selection = f"holdout:{metric}"
            if P.compare_metric(candidate_holdout, pretrained_holdout, metric):
                active_mode, active_checkpoint = "fine-tuned", str(best)
        if candidate_test and pretrained_test:
            worse = [m for m in P.METRIC_IDS if math.isfinite(candidate_test[m]) and math.isfinite(pretrained_test[m]) and candidate_test[m] != pretrained_test[m] and not P.compare_metric(candidate_test, pretrained_test, m)]
            if worse:
                print("WARNING independent-test metrics worsened for the candidate:", worse, "(evidence only; never used for selection)")
    record = {"active_mode": active_mode, "active_checkpoint": active_checkpoint, "selection": selection, "eval_metric": metric, "fine_tune": fine_tune, "pretrained_holdout": pretrained_holdout, "pretrained_test": pretrained_test, "candidate_holdout": candidate_holdout, "candidate_test": candidate_test, "device": pipe.device}
    run.write_state("condition.json", record)
    print({"recommended_for_export": active_mode, "selection_basis": selection, "fine_tune": fine_tune, "device": pipe.device})


def paired_se(y, pred_a, pred_b) -> tuple[float, float]:
    """Mean and standard error of the per-row squared-error difference (a minus b): a negative mean favours a."""
    import numpy as np

    delta = (y - pred_a) ** 2 - (y - pred_b) ** 2
    se = float(np.std(delta, ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return float(np.mean(delta)), se


def stage_report(run: Run) -> None:
    import time

    import numpy as np
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    P = package(run.root)
    data = run.read_state("data.json", "report")
    enc, train, holdout, test = encoded(run, "report")
    model, cond = active_model(run, enc, train)
    features, target = enc["features"], enc["target"]
    X_train, y_train = train[features], train[target]
    lgbm = LGBMRegressor(random_state=SEED, n_estimators=100, verbose=-1).fit(X_train, y_train)
    rf = RandomForestRegressor(random_state=SEED, n_estimators=100).fit(X_train, y_train)
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)

    def timed(m, frame, repeats=5):
        X = frame[features]
        if repeats > 1:
            m.predict(X)  # discarded warm-up call
        latencies = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            preds = np.asarray(m.predict(X), dtype=float)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return preds, float(np.median(latencies))

    y = holdout[target].to_numpy(dtype=float)
    n = len(holdout)
    name = "tabicl_" + cond["active_mode"]
    pred_tab, lat_tab = timed(model, holdout)
    pred_lgbm, lat_lgbm = timed(lgbm, holdout)
    pred_rf, lat_rf = timed(rf, holdout)
    pred_lin, _ = timed(linear, holdout, repeats=1)
    classical = {name: P.regression_metrics(y, pred_tab), "lightgbm": P.regression_metrics(y, pred_lgbm), "random_forest": P.regression_metrics(y, pred_rf), "standardised_linear_regression": P.regression_metrics(y, pred_lin)}
    for label, values in classical.items():
        print(f"{label:<32} mae={values['mae']:.2f} rmse={values['rmse']:.2f} r2={values['r2']:.3f}")
    print({"latency_ms_median_of_5": {"tabicl": round(lat_tab, 2), "lightgbm": round(lat_lgbm, 2), "random_forest": round(lat_rf, 2)}, "rows": n, "device": model.device})
    lin_delta, lin_se = paired_se(y, pred_tab, pred_lin)
    print({"tabicl_vs_linear_holdout": {"mse_delta": round(lin_delta, 1), "paired_se": round(lin_se, 1), "reading": "negative favours TabICLv2; |delta| < 2 SE is within noise"}})
    best_w, best_rmse = 1.0, float("inf")
    for w in np.linspace(0.0, 1.0, 101):
        rmse = P.regression_metrics(y, w * pred_tab + (1.0 - w) * pred_lgbm)["rmse"]
        if rmse < best_rmse - 1e-12:
            best_rmse, best_w = rmse, float(w)
    pred_blend = best_w * pred_tab + (1.0 - best_w) * pred_lgbm
    blend = P.regression_metrics(y, pred_blend)
    blend_delta, blend_se = paired_se(y, pred_blend, pred_tab)
    print(rounded({"blend_objective": "minimise holdout RMSE", "weight_tabicl": best_w, "blend_holdout": blend, "mse_delta_vs_tabicl": blend_delta, "paired_se": blend_se}))
    blend_test_record = None
    if test is not None:
        y_test = test[target].to_numpy(dtype=float)
        pt, _ = timed(model, test, repeats=1)
        pl, _ = timed(lgbm, test, repeats=1)
        plin, _ = timed(linear, test, repeats=1)
        pb = best_w * pt + (1.0 - best_w) * pl
        tab_test = P.regression_metrics(y_test, pt)
        blend_test = P.regression_metrics(y_test, pb)
        test_delta, test_se = paired_se(y_test, pb, pt)
        blend_test_record = {"blend": blend_test, "mse_delta_vs_tabicl": test_delta, "paired_se": test_se}
        print({"independent_test": rounded({"tabicl": tab_test, "lightgbm": P.regression_metrics(y_test, pl), "standardised_linear_regression": P.regression_metrics(y_test, plin), "blend": blend_test, "blend_mse_delta_vs_tabicl": test_delta, "paired_se": test_se})})
        if blend["rmse"] < classical[name]["rmse"] and blend_test["rmse"] >= tab_test["rmse"]:
            print("WARNING mixed evidence: the blend improved holdout RMSE but not independent-test RMSE; treat the holdout gain as selection-biased.")
    else:
        print("No independent test partition; the holdout blend score is selection-biased demonstration evidence.")
    active_holdout = cond["candidate_holdout"] if cond["active_mode"] == "fine-tuned" else cond["pretrained_holdout"]
    active_test = cond["candidate_test"] if cond["active_mode"] == "fine-tuned" else cond["pretrained_test"]
    report = P.evaluation_report(active_holdout, baseline=enc["baseline"], independent_test=active_test, n_holdout=n, n_test=None if test is None else len(test), target_column=target, selection=cond["selection"], sample_kind=data["sample_kind"], estimation="single seeded random split (support/holdout/independent test); no dispersion estimate")
    report["classical_baselines_holdout"] = {k: v for k, v in classical.items() if k != name}
    report["tabicl_vs_linear_holdout"] = {"mse_delta": lin_delta, "paired_se": lin_se}
    report["blend_holdout"] = {"weight_tabicl": best_w, **blend, "mse_delta_vs_tabicl": blend_delta, "paired_se": blend_se}
    report["blend_independent_test"] = blend_test_record
    report["fine_tune"] = cond["fine_tune"]
    report["interpretation"] = f"{n} holdout rows; an MSE difference smaller than about twice its paired standard error is within split noise, and no dispersion over splits was measured. A standardised linear regression is the classical reference on this table."
    run.write_output(f"{STEM}_evaluation_report.json", report)
    print(json.dumps({k: report[k] for k in ("verdict", "reason", "selection", "n_holdout", "n_test", "interpretation")}, indent=2))


def runtime_versions(device: str) -> dict[str, Any]:
    import importlib.metadata
    import platform

    import numpy
    import pandas
    import sklearn
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__, "tabicl": importlib.metadata.version("tabicl"), "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "device": device}


def new_rows_frame(run: Run, enc: dict):
    """The first eight holdout rows (raw features, unlabelled) with their original row id: the default new-data input
    and the companion notebook's sample rows (TIR-S2)."""
    raw = run.read_frame("holdout_raw.parquet", "new rows").reset_index(drop=True)
    rows = raw[enc["features"]].head(NEW_ROWS).copy()
    rows.insert(0, "row_id", enc["holdout_row_ids"][: len(rows)])
    return rows


def with_predictions(P, model, rows, enc):
    X, unseen = P.apply_categorical_encoder(rows[enc["features"]], enc["encoders"])
    out = rows.copy()
    out["prediction"] = model.predict(X)
    return out, unseen


def stage_predict(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "predict")
    enc, train, holdout, test = encoded(run, "predict")
    model, cond = active_model(run, enc, train)
    new_data = None
    path = run.options.get("new_data_path") or ""
    if path:
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"NEW_DATA_PATH {path!r} does not exist in this runtime.")
        rows = P.read_inference_csv(file.read_bytes(), enc["features"])
        manifest = P.validate_inputs(rows, None, feature_columns=enc["features"], names=[file.name])
        out, unseen = with_predictions(P, model, rows, enc)
        new_data = {"input": file.name, "rows": len(out), "unseen_categorical_values": unseen, "input_manifest": manifest}
    else:
        rows = new_rows_frame(run, enc)
        out, unseen = with_predictions(P, model, rows, enc)
        rows.to_csv(run.out / f"{STEM}_new_rows.csv", index=False)
        print("No new-data file: the first eight holdout rows (already scored in Section 7) are scored instead; their unlabelled copy is written to", f"outputs/{STEM}_new_rows.csv")
    out.to_csv(run.out / f"{STEM}_predictions.csv", index=False)
    report = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    input_manifest = json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    active_holdout = cond["candidate_holdout"] if cond["active_mode"] == "fine-tuned" else cond["pretrained_holdout"]
    active_test = cond["candidate_test"] if cond["active_mode"] == "fine-tuned" else cond["pretrained_test"]
    payload = {
        "predictions": out.to_dict(orient="records"),
        "new_data": new_data,
        "metrics": {"active_mode": cond["active_mode"], "holdout": active_holdout, "independent_test": active_test, "pretrained_holdout": cond["pretrained_holdout"], "candidate_holdout": cond["candidate_holdout"]},
        "fine_tune": cond["fine_tune"],
        "training_mean_baseline": enc["baseline"],
        "evaluation_report": report,
        "input_manifest": input_manifest,
        "inference": {"prediction": "continuous point estimate in target units", "uncertainty": None},
        "sample": {"kind": data["sample_kind"], "name": data["name"], "source": data["source"], "csv_sha256": data["csv_sha256"], "train_rows": len(train), "holdout_rows": len(holdout), "test_rows": 0 if test is None else len(test), "training_target": enc["target_summary"]},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "runtime": runtime_versions(model.device),
    }
    run.write_output(f"{STEM}_result.json", payload)
    if unseen:
        print("unseen categorical values mapped to the unknown code:", unseen)
    print(out.round(4).to_string(index=False))


def build_manifest(P, *, enc: dict, mode: str, selection: str, metrics: dict | None, n_estimators: int, checkpoint: Path | None, context: Path, runtime: dict, source: dict, checkpoint_facts: tuple[str, int] | None = None) -> dict[str, Any]:
    """The bundle manifest: the E2E export and tools/build_sample_bundle.py share this function."""
    return {
        "artifactFormat": P.ARTIFACT_FORMAT,
        "checkpoint": "checkpoints/best.ckpt",
        "trainingContext": "training_context.parquet",
        "targetColumn": enc["target"],
        "featureColumns": enc["features"],
        "baseCheckpoint": P.BASE_CHECKPOINT_NAME,
        "baseModelRevision": P.MODEL_REVISION,
        "baseModelSha256": P.BASE_MODEL_SHA256,
        "tabiclVersion": "2.1.1",
        "mode": mode,
        "selectionBasis": selection,
        "metrics": metrics,
        "inference": {"class": "TabICLRegressor", "modelPath": "checkpoints/best.ckpt", "nEstimators": n_estimators, "randomState": SEED, "allowAutoDownload": False, "categoricalEncoders": enc["encoders"]},
        "digests": {"checkpointSha256": checkpoint_facts[0] if checkpoint_facts else sha256_file(checkpoint), "trainingContextSha256": sha256_file(context)},
        "sizes": {"checkpoint": checkpoint_facts[1] if checkpoint_facts else checkpoint.stat().st_size, "trainingContext": context.stat().st_size},
        "payloadFiles": ["checkpoints/best.ckpt", "training_context.parquet"],
        "runtime": runtime,
        "notebookSource": source,
    }


def stage_export(run: Run) -> None:
    import importlib.metadata
    import platform

    import numpy
    import pandas
    import sklearn
    import torch

    P = package(run.root)
    enc, train, holdout, _test = encoded(run, "export")
    model, cond = active_model(run, enc, train)
    art = run.out / "artifact"
    shutil.rmtree(art, ignore_errors=True)
    (art / "checkpoints").mkdir(parents=True)
    ckpt = art / "checkpoints" / "best.ckpt"
    shutil.copy2(cond["active_checkpoint"], ckpt)
    context = art / "training_context.parquet"
    train[enc["features"] + [enc["target"]]].to_parquet(context, index=False)
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    runtime = {"pythonVersion": platform.python_version(), "torchVersion": torch.__version__, "pandasVersion": pandas.__version__, "pyarrowVersion": importlib.metadata.version("pyarrow"), "scikitLearnVersion": sklearn.__version__, "device": model.device}
    metrics = {"selectionMetric": cond["eval_metric"], "pretrainedHoldout": cond["pretrained_holdout"], "fineTunedHoldout": cond["candidate_holdout"], "pretrainedIndependentTest": cond["pretrained_test"], "fineTunedIndependentTest": cond["candidate_test"]}
    manifest = build_manifest(P, enc=enc, mode=cond["active_mode"], selection=cond["selection"], metrics=metrics, n_estimators=model.n_estimators, checkpoint=ckpt, context=context, runtime=runtime, source=source)
    (art / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = Path(shutil.make_archive(str(run.out / f"{STEM}_artifact"), "zip", root_dir=art))
    smoke = holdout[enc["features"]].iloc[: min(NEW_ROWS, len(holdout))]
    run.write_state("export_reference.json", {"predictions": numpy.asarray(model.predict(smoke), dtype=float).tolist(), "device": model.device})
    digest = sha256_file(archive)
    result_path = run.out / f"{STEM}_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifact"] = {"zip": archive.name, "zip_sha256": digest, "manifest": manifest}
        run.write_output(result_path.name, result)
    record = {"artifact_zip": str(archive), "zip_sha256": digest, "mode": cond["active_mode"], "checkpoint_bytes": ckpt.stat().st_size, "training_context_sha256": manifest["digests"]["trainingContextSha256"]}
    sample = run.root / SAMPLE_BUNDLE_RECORD
    data = run.read_state("data.json", "export")
    if data["source"] == "Sample: Diabetes" and sample.is_file():
        pinned = json.loads(sample.read_text(encoding="utf-8"))
        record["matches_pinned_sample_context"] = manifest["digests"]["trainingContextSha256"] == pinned["files"]["training_context.parquet"]["sha256"]
    print(record)
    print(f"Trusted digest for the companion notebook: set EXPECTED_ZIP_SHA256 = '{digest}' when you hand it {archive.name}.")


def stage_reload(run: Run) -> None:
    import numpy as np
    import pandas as pd

    P = package(run.root)
    enc, _train, holdout, _test = encoded(run, "reload")
    reference = run.read_state("export_reference.json", "reload")
    archive = run.out / f"{STEM}_artifact.zip"
    reload_dir = run.out / "artifact-reload"
    shutil.rmtree(reload_dir, ignore_errors=True)
    root = P.safe_extract_zip(archive, reload_dir)
    served = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    members = P.verify_artifact_bundle(root, served)
    context = pd.read_parquet(members["training_context"])
    model = P.TabICLRegressionPipeline(P.create_regressor(model_path=members["checkpoint"], allow_auto_download=False, n_estimators=served["inference"]["nEstimators"], random_state=served["inference"]["randomState"], device=reference["device"]), model_path=members["checkpoint"], n_estimators=served["inference"]["nEstimators"], random_state=served["inference"]["randomState"], device=reference["device"], source="artifact")
    model.fit(context[served["featureColumns"]], context[served["targetColumn"]])
    smoke = holdout[enc["features"]].iloc[: min(NEW_ROWS, len(holdout))]
    preds = np.asarray(model.predict(smoke), dtype=float)
    np.testing.assert_allclose(np.asarray(reference["predictions"]), preds, rtol=RELOAD_RTOL, atol=RELOAD_ATOL)
    print({"reload_dir": str(reload_dir), "process": "fresh (separate from the exporting stage)", "max_abs_prediction_difference": float(np.max(np.abs(np.asarray(reference["predictions"]) - preds)))})
    print(f"PASS: bundle extracted safely, digests verified, regressor rebuilt from the bundle alone; predictions equivalent (rtol={RELOAD_RTOL}, atol={RELOAD_ATOL}).")


def stage_activity(run: Run) -> None:
    """Optional activity: condition the pretrained model with a different ensemble size on the same support rows and
    compare holdout and independent-test MSE with the canonical run through the paired standard error; writes only to
    outputs/activity/."""
    import numpy as np

    P = package(run.root)
    n = int(run.options.get("n_estimators", 32))
    if not 1 <= n <= 64:
        raise ValueError(f"ACTIVITY_N_ESTIMATORS must be from 1 to 64, got {n}")
    before = {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()}
    enc, train, holdout, test = encoded(run, "activity")
    X, y = train[enc["features"]], train[enc["target"]]
    canonical, changed = pretrained(run), pretrained(run, n_estimators=n)
    canonical.fit(X, y)
    changed.fit(X, y)
    record = {"changed": {"n_estimators": [canonical.n_estimators, n]}}
    for label, frame in (("holdout", holdout), ("independent_test", test)):
        if frame is None:
            continue
        yy = frame[enc["target"]].to_numpy(dtype=float)
        pa = np.asarray(changed.predict(frame[enc["features"]]), dtype=float)
        pb = np.asarray(canonical.predict(frame[enc["features"]]), dtype=float)
        delta, se = paired_se(yy, pa, pb)
        record[label] = {"rows": len(frame), "canonical": P.regression_metrics(yy, pb), "changed": P.regression_metrics(yy, pa), "mse_delta_changed_minus_canonical": delta, "paired_se": se, "within_two_se": abs(delta) < 2 * se}
    run.write_output(f"activity/{STEM}_activity_n_estimators_{n}.json", record)
    after = {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()}
    if before != after:
        raise RuntimeError("The activity changed a canonical output; it must write only to outputs/activity/.")
    print(rounded(record))
    print({"canonical_outputs_unchanged": True})


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "validate": stage_validate,
    "condition": stage_condition,
    "report": stage_report,
    "predict": stage_predict,
    "export": stage_export,
    "reload": stage_reload,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
