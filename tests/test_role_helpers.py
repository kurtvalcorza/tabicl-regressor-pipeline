"""Offline tests for the helpers extracted from the tutorials (DAT24 / EVAL21, encoding, metrics, ZIP safety).

No weights, no model: everything runs on small in-memory tables and temporary files.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabicl_regressor_pipeline import (
    INPUT_SCHEMA,
    MAX_FEATURES,
    MAX_TRAIN_ROWS,
    METRIC_IDS,
    MIN_EVAL_ROWS,
    MIN_TRAIN_ROWS,
    MODEL_ID,
    MODEL_REVISION,
    align_to_schema,
    apply_categorical_encoder,
    compare_metric,
    evaluation_report,
    fit_categorical_encoder,
    manifest_member_path,
    prepare_regression_table,
    read_inference_csv,
    regression_metrics,
    safe_extract_zip,
    training_mean_baseline,
    validate_inputs,
    verify_artifact_bundle,
)
from tabicl_regressor_pipeline.api import sha256_file


def _table(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.integers(0, 5, size=rows),
            "city": rng.choice(["a", "b", "c"], size=rows),
            "target": rng.normal(size=rows) * 10 + 100,
        }
    )


def test_validate_inputs_fit_mode_returns_manifest_with_schema_and_identity() -> None:
    frame = _table()
    frame.loc[0, "target"] = np.nan  # dropped, counted, not hidden
    manifest = validate_inputs(frame, target_column="target", names=["sample"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["train_rows"] == [MIN_TRAIN_ROWS, MAX_TRAIN_ROWS]
    assert manifest["schema"]["features"] == [1, MAX_FEATURES]
    assert manifest["min_rows"] == MIN_TRAIN_ROWS
    (entry,) = manifest["inputs"]
    assert entry["id"] == "sample"
    assert entry["mode"] == "fit"
    assert entry["rows"] == 59
    assert entry["dropped_non_finite_target_rows"] == 1
    assert entry["feature_columns"] == ["x1", "x2", "city"]
    assert entry["categorical_columns"] == ["city"]
    assert entry["missing_value_columns"] == {}
    assert set(entry["target_summary"]) == {"min", "max", "mean", "distinct"}
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_rejects_like_prepare_regression_table() -> None:
    cases = [
        (_table().rename(columns={"target": "y"}), KeyError, "missing target"),
        (_table(10), ValueError, "at least 50 labelled rows"),
        (_table()[["target"]], ValueError, "No feature columns"),
        (_table().assign(target=1.0), ValueError, "must vary"),
        (pd.concat([_table(), _table()[["x1"]]], axis=1), ValueError, "duplicate column names"),
    ]
    for bad, exc, message in cases:
        with pytest.raises(exc, match=message):
            validate_inputs(bad, target_column="target")
        with pytest.raises(exc, match=message):
            prepare_regression_table(bad, "target")
    wide = pd.DataFrame(np.zeros((60, MAX_FEATURES + 1))).assign(target=np.arange(60.0))
    wide.columns = [str(c) for c in wide.columns]
    with pytest.raises(ValueError, match="Operational row/feature ceiling exceeded"):
        prepare_regression_table(wide, "target")
    with pytest.raises(ValueError, match="Operational row/feature ceiling exceeded"):
        validate_inputs(wide, target_column="target")
    holdout = validate_inputs(_table(5), target_column="target", min_rows=MIN_EVAL_ROWS)
    assert holdout["inputs"][0]["rows"] == 5
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_table(), names=["a", "b"])


def test_validate_inputs_inference_mode_rejects_like_read_inference_csv() -> None:
    features = ["x1", "x2", "city"]
    rows = _table(3).drop(columns=["target"]).assign(extra="keep")
    manifest = validate_inputs(rows, None, feature_columns=features, names=["new"])
    (entry,) = manifest["inputs"]
    assert entry == {
        "id": "new",
        "mode": "inference",
        "rows": 3,
        "feature_columns": features,
        "extra_columns": ["extra"],
        "missing_value_columns": {},
    }
    assert manifest["target_column"] is None and manifest["min_rows"] is None
    for bad, message in (
        (rows.drop(columns=["city"]), r"missing features: \['city'\]"),
        (rows.assign(prediction=1.0), "already contains a 'prediction' column"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad, None, feature_columns=features)
        with pytest.raises(ValueError, match=message):
            read_inference_csv(bad.to_csv(index=False).encode("utf-8"), features)
    with pytest.raises(ValueError, match=r"duplicate column names: \['x1'\]"):
        read_inference_csv(b"x1,x1,x2,city\n1,2,3,a\n", features)
    with pytest.raises(ValueError, match="feature_columns is required"):
        validate_inputs(rows, None)


def test_categorical_encoder_and_schema_alignment() -> None:
    train = _table()
    encoders = fit_categorical_encoder(train, ["x1", "x2", "city"])
    assert encoders == {"city": ["a", "b", "c"]}
    new = pd.DataFrame({"x1": [0.0, 1.0], "x2": [1, 2], "city": ["b", "zzz"]})
    encoded, unseen = apply_categorical_encoder(new, encoders)
    assert encoded["city"].tolist() == [1, 3]  # 'zzz' -> unknown code = len(categories)
    assert unseen == {"city": 1}
    aligned = align_to_schema(train[["target", "city", "x2", "x1"]], ["x1", "x2", "city"], "target")
    assert list(aligned.columns) == ["x1", "x2", "city", "target"]
    with pytest.raises(ValueError, match="schema does not match train"):
        align_to_schema(train.drop(columns=["x2"]), ["x1", "x2", "city"], "target")


def test_regression_metrics_and_training_mean_baseline() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert set(metrics) == set(METRIC_IDS)
    assert math.isclose(metrics["mae"], 1 / 3)
    assert math.isclose(metrics["rmse"], math.sqrt(1 / 3))
    baseline = training_mean_baseline([1.0, 2.0, 3.0], [2.0, 4.0])
    assert set(baseline) == {"mae", "mse", "rmse", "r2"}
    assert baseline["mae"] == 1.0 and math.isclose(baseline["r2"], -1.0)
    assert compare_metric({"mae": 1.0}, {"mae": 2.0}, "mae") is True
    assert compare_metric({"r2": 0.1}, {"r2": 0.2}, "r2") is False
    with pytest.raises(ValueError, match="unavailable for selection"):
        compare_metric({"r2": float("nan")}, {"r2": 0.2}, "r2")
    with pytest.raises(ValueError, match="non-empty and finite"):
        training_mean_baseline([1.0, np.inf], [1.0])


def test_evaluation_report_not_measurable_without_metrics() -> None:
    report = evaluation_report(None, n_holdout=0, target_column="target", sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == [] and report["baselines"] == [] and report["independent_test"] == []
    assert "training_mean_baseline" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_baseline_and_independent_test() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 2.9, 4.2])
    baseline = training_mean_baseline([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    report = evaluation_report(
        metrics,
        baseline=baseline,
        independent_test=metrics,
        n_holdout=4,
        n_test=4,
        target_column="target",
        selection="default:pretrained",
        sample_kind="sample",
        estimation="e",
    )
    assert report["verdict"] == "sample-sanity"
    assert [m["id"] for m in report["metrics"]] == list(METRIC_IDS)
    assert all(m["estimation"] == "e" for m in report["metrics"])
    assert [m["id"] for m in report["independent_test"]] == list(METRIC_IDS)
    assert report["baselines"][0]["id"] == "training_mean"
    assert [m["id"] for m in report["baselines"][0]["metrics"]] == ["mae", "mse", "rmse", "r2"]
    assert report["selection"] == "default:pretrained"
    assert {m["id"]: m["higher_is_better"] for m in report["metrics"]}["r2"] is True
    nan_report = evaluation_report({"mae": 1.0, "pearsonr": float("nan")}, n_holdout=1)
    assert nan_report["metrics"][1] == {
        "id": "pearsonr",
        "value": None,
        "units": "unitless",
        "higher_is_better": True,
        "estimation": "single seeded random holdout; no dispersion estimate",
    }
    with pytest.raises(ValueError, match="unknown metric ids"):
        evaluation_report({"mape": 0.5})


def test_safe_extract_zip_and_bundle_verification(tmp_path: Path) -> None:
    src = tmp_path / "bundle"
    (src / "checkpoints").mkdir(parents=True)
    (src / "checkpoints" / "best.ckpt").write_bytes(b"ckpt-bytes")
    (src / "training_context.parquet").write_bytes(b"pq-bytes")
    manifest = {
        "checkpoint": "checkpoints/best.ckpt",
        "trainingContext": "training_context.parquet",
        "payloadFiles": ["checkpoints/best.ckpt", "training_context.parquet"],
        "sizes": {"checkpoint": 10, "trainingContext": 8},
        "digests": {
            "checkpointSha256": sha256_file(src / "checkpoints" / "best.ckpt"),
            "trainingContextSha256": sha256_file(src / "training_context.parquet"),
        },
    }
    (src / "artifact.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src).as_posix())
    root = safe_extract_zip(archive, tmp_path / "out")
    members = verify_artifact_bundle(root, manifest)
    assert members["checkpoint"] == (root / "checkpoints" / "best.ckpt").resolve()
    (root / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unexpected or missing artifact files"):
        verify_artifact_bundle(root, manifest)
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../escape.txt", "x")
        safe_extract_zip(evil, tmp_path / "evil-out")
    assert not (tmp_path / "escape.txt").exists()
    with pytest.raises(ValueError, match="expanded bytes"):
        safe_extract_zip(archive, tmp_path / "small", max_expanded_bytes=4)
    with pytest.raises(ValueError, match="Unsafe checkpoint path"):
        manifest_member_path(root, "../x", "checkpoint")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("a/b.txt", "x")
    nested = tmp_path / "nested.zip"
    nested.write_bytes(buffer.getvalue())
    assert (safe_extract_zip(nested, tmp_path / "nested-out") / "a" / "b.txt").read_text() == "x"
