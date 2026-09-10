from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_FORMAT = "tabicl-dimer-regressor-v1"
BASE_MODEL_REPO = "jingang/TabICL"
BASE_CHECKPOINT_NAME = "tabicl-regressor-v2-20260212.ckpt"
BASE_MODEL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
BASE_MODEL_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"


def runtime_identity() -> dict[str, str]:
    import torch

    return {
        "pythonVersion": sys.version.split()[0],
        "tabiclVersion": importlib.metadata.version("tabicl"),
        "torchVersion": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def create_regressor(
    *,
    model_path: str | Path,
    n_estimators: int,
    random_state: int,
    device: str,
    allow_auto_download: bool = False,
):
    """Construct the supported TabICLv2 regression serving estimator."""
    from tabicl import TabICLRegressor

    return TabICLRegressor(
        model_path=str(model_path),
        allow_auto_download=allow_auto_download,
        n_estimators=n_estimators,
        random_state=random_state,
        device=device,
    )


def create_finetuned_regressor(**kwargs: Any):
    """Construct the supported TabICLv2 regression fine-tuning estimator."""
    from tabicl import FinetunedTabICLRegressor

    kwargs.setdefault("allow_auto_download", False)
    return FinetunedTabICLRegressor(**kwargs)


def fine_tune_regressor(model: Any, X: Any, y: Any, **kwargs: Any) -> Any:
    """Run the upstream fine-tuning operation through the repository API."""
    return model.fit(X, y, **kwargs)


def condition_regressor(model: Any, X: Any, y: Any) -> Any:
    """Register the serving support context required by TabICL inference."""
    model.fit(X, y)
    return model


def predict_points(model: Any, X: Any) -> np.ndarray:
    """Return finite one-dimensional regression point predictions."""
    values = np.asarray(model.predict(X), dtype=float)
    if values.ndim != 1:
        values = values.reshape(-1)
    if not np.isfinite(values).all():
        raise RuntimeError("TabICL produced non-finite point predictions")
    return values


def read_single_input(*, env_var: str, label: str) -> tuple[str, bytes]:
    """Read one user-supplied file from an explicit path or Colab upload dialog."""
    explicit = os.environ.get(env_var, "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
        return path.name, path.read_bytes()

    try:
        from google.colab import files  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{label} requires either a Colab upload or environment variable {env_var}"
        ) from exc

    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError(f"Upload exactly one {label}")
    name, payload = next(iter(uploaded.items()))
    return str(name), bytes(payload)


def download_output(path: str | Path) -> Path:
    """Download in Colab; otherwise retain the file at its explicit Jupyter path."""
    resolved = Path(path).resolve()
    try:
        from google.colab import files  # type: ignore
    except ModuleNotFoundError:
        print(f"Output retained at {resolved}")
        return resolved
    files.download(str(resolved))
    return resolved


def _base_version(value: str) -> str:
    return value.split("+")[0]


def validate_artifact_runtime(
    manifest: dict[str, Any],
    *,
    expected_artifact_format: str = ARTIFACT_FORMAT,
    expected_tabicl_version: str = "2.1.1",
    expected_torch_version: str = "2.11.0",
) -> dict[str, Any]:
    """Validate and return artifact/runtime provenance before model deserialization."""
    if manifest.get("artifactFormat") != expected_artifact_format:
        raise ValueError(f"Unsupported artifactFormat: {manifest.get('artifactFormat')}")
    if manifest.get("tabiclVersion") != expected_tabicl_version:
        raise ValueError("Artifact TabICL version does not match the supported runtime")

    observed = runtime_identity()
    if _base_version(observed["torchVersion"]) != expected_torch_version:
        raise RuntimeError(
            f"Runtime torch {observed['torchVersion']} is incompatible with the "
            f"release-verified torch {expected_torch_version}"
        )

    producer = manifest.get("runtime")
    compatibility = {
        "artifactFormat": manifest.get("artifactFormat"),
        "baseCheckpoint": manifest.get("baseCheckpoint"),
        "baseModelRevision": manifest.get("baseModelRevision"),
        "baseModelSha256": manifest.get("baseModelSha256"),
        "tabiclVersion": manifest.get("tabiclVersion"),
        "producerRuntime": producer,
        "consumerRuntime": observed,
        "policy": "TabICL exact; consumer torch base version 2.11.0; device may differ",
    }

    if producer:
        producer_torch = str(producer.get("torchVersion", ""))
        if producer_torch and _base_version(producer_torch) != expected_torch_version:
            raise RuntimeError(
                f"Artifact was produced with torch {producer_torch}; expected release family "
                f"{expected_torch_version}"
            )
    else:
        compatibility["warning"] = (
            "Legacy artifact has no runtime block; TabICL/version and digest checks apply, "
            "but producer runtime compatibility cannot be proven."
        )
    return compatibility
