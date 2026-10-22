from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials/tabiclv2_regressor_colab.ipynb"
WORK = Path("/content")
MAIN_EXECUTED = WORK / "tabiclv2_regressor_colab.executed.ipynb"
ARTIFACT = WORK / "tabiclv2-regressor-artifact.zip"


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


WORK.mkdir(parents=True, exist_ok=True)
for path in (MAIN_EXECUTED, ARTIFACT):
    if path.exists():
        path.unlink()

# This is an actual IPython/Jupyter kernel execution. Notebook magics such as
# %pip are executed by the kernel; they are not stripped or simulated.
execute_notebook(MAIN, MAIN_EXECUTED)
if not ARTIFACT.exists():
    raise RuntimeError("E2E notebook did not produce the serving artifact")

# NOTEBOOK_SPEC 3.0 (KND11): the artifact-inference companion was folded into the E2E notebook's
# closing external-artifact section, which this execution already ran in the same fresh kernel.
# That section consumes the pinned sample artifact of an earlier recorded run, never this
# execution's export; there is no second notebook.

print("Release notebook-engine execution: PASS")
print("Commit:", os.environ.get("GITHUB_SHA", "local"))
print("Python:", sys.version.split()[0])
print("Execution engine: nbclient/IPython kernel")
print("Producer executed notebook:", MAIN_EXECUTED)
print("This run's export:", ARTIFACT)
