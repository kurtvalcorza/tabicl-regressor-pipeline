from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials/tabiclv2_regressor_colab.ipynb"
INFERENCE = ROOT / "tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb"


class FilesStub:
    def __init__(self, uploads=None):
        self.uploads = list(uploads or [])
        self.downloads = []

    def upload(self):
        if not self.uploads:
            raise RuntimeError("Unexpected files.upload() in release execution")
        return self.uploads.pop(0)

    def download(self, path):
        self.downloads.append(str(path))


def install_colab_stub(stub: FilesStub) -> None:
    try:
        import google  # type: ignore
    except Exception:
        google = types.ModuleType("google")
        google.__path__ = []
        sys.modules["google"] = google
    colab = types.ModuleType("google.colab")
    colab.files = stub
    sys.modules["google.colab"] = colab
    setattr(sys.modules["google"], "colab", colab)


def execute_notebook(path: Path, stub: FilesStub) -> dict:
    install_colab_stub(stub)
    nb = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        code = "".join(src) if isinstance(src, list) else str(src)
        code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("%"))
        if not code.strip():
            continue
        exec(compile(code, f"{path.name}:cell{index}", "exec"), namespace)
    return namespace


main_stub = FilesStub()
main_ns = execute_notebook(MAIN, main_stub)
archive = Path(main_ns["archive_path"])
if not archive.exists():
    raise RuntimeError("Main notebook did not produce serving artifact")

features = list(main_ns["FEATURE_COLUMNS"])
holdout = main_ns["holdout_data"]
inference_frame = holdout[features].head(8).copy()
inference_csv = inference_frame.to_csv(index=False).encode("utf-8")
artifact_bytes = archive.read_bytes()

inf_stub = FilesStub([
    {archive.name: artifact_bytes},
    {"new_rows.csv": inference_csv},
])
inf_ns = execute_notebook(INFERENCE, inf_stub)
output = Path(inf_ns["output_path"])
if not output.exists():
    raise RuntimeError("Artifact notebook did not write predictions.csv")
result = pd.read_csv(output)
if len(result) != len(inference_frame) or "prediction" not in result.columns:
    raise RuntimeError("Artifact notebook prediction output contract failed")

print("Release notebook execution: PASS")
print("Commit:", os.environ.get("GITHUB_SHA", "local"))
print("Python:", sys.version.split()[0])
print("Rows scored:", len(result))
print("Main artifact:", archive)
print("Companion output:", output)
