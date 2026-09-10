from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from sklearn.datasets import load_diabetes

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials/tabiclv2_regressor_colab.ipynb"
INFERENCE = ROOT / "tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb"
WORK = Path("/content")
MAIN_EXECUTED = WORK / "tabiclv2_regressor_colab.executed.ipynb"
INFERENCE_EXECUTED = WORK / "tabiclv2_regressor_artifact_inference_colab.executed.ipynb"
ARTIFACT = WORK / "tabiclv2-regressor-artifact.zip"
FRESH_ROWS = WORK / "tabiclv2_regressor_release_fresh_rows.csv"
PREDICTIONS = WORK / "tabiclv2_regressor_predictions.csv"


def execute_notebook(source: Path, executed: Path) -> None:
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(WORK)}},
    )
    client.execute()
    nbformat.write(notebook, executed)


def make_genuinely_new_rows(path: Path) -> pd.DataFrame:
    dataset = load_diabetes(as_frame=True)
    original = dataset.data.reset_index(drop=True).astype(float)
    means = original.mean(axis=0)
    scales = original.std(axis=0).replace(0.0, 1.0)
    offsets = np.linspace(-0.37, 0.37, 8)
    fresh = pd.DataFrame([means + offset * scales for offset in offsets], columns=original.columns)

    original_rows = {tuple(row) for row in original.to_numpy(dtype=float)}
    for row in fresh.to_numpy(dtype=float):
        if tuple(row) in original_rows:
            raise RuntimeError("Release verifier generated a row already present in the producer sample")
    fresh.to_csv(path, index=False)
    return fresh


WORK.mkdir(parents=True, exist_ok=True)
for path in (MAIN_EXECUTED, INFERENCE_EXECUTED, ARTIFACT, FRESH_ROWS, PREDICTIONS):
    if path.exists():
        path.unlink()

# This is an actual IPython/Jupyter kernel execution. Notebook magics such as
# %pip are executed by the kernel; they are not stripped or simulated.
execute_notebook(MAIN, MAIN_EXECUTED)
if not ARTIFACT.exists():
    raise RuntimeError("E2E notebook did not produce the serving artifact")

fresh = make_genuinely_new_rows(FRESH_ROWS)

# The companion runs in a second fresh kernel. Its artifact and inference CSV
# are supplied through explicit external paths, satisfying the same boundary a
# non-Colab Jupyter consumer uses; interactive Colab falls back to upload().
os.environ["DIMER_ARTIFACT_PATH"] = str(ARTIFACT)
os.environ["DIMER_INFERENCE_CSV_PATH"] = str(FRESH_ROWS)
execute_notebook(INFERENCE, INFERENCE_EXECUTED)

if not PREDICTIONS.exists():
    raise RuntimeError("Artifact-inference notebook did not write predictions.csv")
result = pd.read_csv(PREDICTIONS)
if len(result) != len(fresh) or "prediction" not in result.columns:
    raise RuntimeError("Artifact-inference prediction output contract failed")
if not np.isfinite(result["prediction"].to_numpy(dtype=float)).all():
    raise RuntimeError("Artifact-inference notebook produced non-finite predictions")

print("Release notebook-engine execution: PASS")
print("Commit:", os.environ.get("GITHUB_SHA", "local"))
print("Python:", sys.version.split()[0])
print("Execution engine: nbclient/IPython kernel")
print("Producer executed notebook:", MAIN_EXECUTED)
print("External artifact:", ARTIFACT)
print("Fresh inference rows:", FRESH_ROWS)
print("Rows scored:", len(result))
print("Consumer executed notebook:", INFERENCE_EXECUTED)
print("Predictions:", PREDICTIONS)
