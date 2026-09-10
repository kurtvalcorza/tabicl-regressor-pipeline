from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials/tabiclv2_regressor_colab.ipynb"
INFERENCE = ROOT / "tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb"
README = ROOT / "tutorials/README.md"
REQ = ROOT / "tutorials/requirements-release.txt"

MODEL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINT_NAME = "tabicl-regressor-v2-20260212.ckpt"
CHECKPOINT_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
TABICL_VERSION = "2.1.1"
ARTIFACT_FORMAT = "tabicl-dimer-regressor-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_nb(path: Path) -> dict:
    require(path.exists(), f"missing notebook: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def text(nb: dict) -> str:
    out = []
    for cell in nb["cells"]:
        src = cell.get("source", "")
        out.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(out)


def code_text(nb: dict) -> str:
    out = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        out.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(out)


def compile_cells(nb: dict, label: str) -> None:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        value = "".join(src) if isinstance(src, list) else str(src)
        value = "\n".join(line for line in value.splitlines() if not line.lstrip().startswith("%"))
        if value.strip():
            ast.parse(value, filename=f"{label}:cell{i}")


def clean_state(nb: dict, label: str) -> None:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            require(cell.get("execution_count") is None, f"{label}: cell {i} has execution_count")
            require(cell.get("outputs", []) == [], f"{label}: cell {i} has persisted outputs")


main_nb = load_nb(MAIN)
inf_nb = load_nb(INFERENCE)
main_all, inf_all = text(main_nb), text(inf_nb)
main_code, inf_code = code_text(main_nb), code_text(inf_nb)
compile_cells(main_nb, "main")
compile_cells(inf_nb, "inference")
clean_state(main_nb, "main")
clean_state(inf_nb, "inference")

require(main_nb.get("metadata", {}).get("dimer") == {"notebook_profile": "E2E", "notebook_spec": "1.0"}, "main profile metadata")
require(inf_nb.get("metadata", {}).get("dimer") == {"notebook_profile": "ARTIFACT-INFERENCE", "notebook_spec": "1.0"}, "artifact profile metadata")
require("**Profile:** `E2E`" in main_all, "main missing normative profile")
require("**Profile:** `ARTIFACT-INFERENCE`" in inf_all, "artifact missing normative profile")
require("**Notebook spec:** DIMER Notebook Specification `1.0`" in main_all, "main missing spec version")
require("**Notebook spec:** DIMER Notebook Specification `1.0`" in inf_all, "artifact missing spec version")
require("**API boundary.**" in main_all and "**API boundary.**" in inf_all, "API boundary not documented")

pins = {
    'tabicl[finetune]==2.1.1', 'lightgbm==4.7.0', 'pyarrow==25.0.1', 'pandas==2.3.3',
    'scikit-learn==1.9.0', 'huggingface_hub==1.30.0', 'transformers==5.6.2', 'wandb==0.27.2'
}
for pin in pins:
    require(pin in main_code, f"main missing pin {pin}")
for pin in {'tabicl==2.1.1', 'pyarrow==25.0.1', 'pandas==2.3.3', 'scikit-learn==1.9.0'}:
    require(pin in inf_code, f"artifact missing pin {pin}")
require('EXPECTED_TORCH_VERSION = "2.11.0"' in main_code and 'EXPECTED_TORCH_VERSION = "2.11.0"' in inf_code, "PyTorch verification pin missing")
for forbidden in ("pandas>=", "pyarrow>=", "scikit-learn>=", "huggingface_hub>=", "lightgbm>="):
    require(forbidden not in main_code and forbidden not in inf_code, f"floating explicit install remains: {forbidden}")

for marker in (CHECKPOINT_NAME, MODEL_REVISION, CHECKPOINT_SHA256, 'CHECKPOINT_SOURCE = "Pinned upstream"', '"DIMER ZIP"', "allow_auto_download=False", "TabICLRegressor", "FinetunedTabICLRegressor", "RUN_FINE_TUNING = False", "MIN_SELECTION_HOLDOUT_ROWS = 50", ARTIFACT_FORMAT, "training_context.parquet", "checkpoints/best.ckpt", "artifact.json", "Candidate checkpoint holdout"):
    require(marker in main_code, f"main code missing {marker!r}")
for marker in ("mean_absolute_error", "mean_squared_error", "r2_score", "Training regression target must vary", "trivial baseline"):
    require(marker in main_all, f"regression requirement missing {marker!r}")
require("no uncertainty band" in main_all or "no per-prediction uncertainty" in main_all, "main uncertainty boundary missing")
require("Reproducibility boundary" in main_all, "main variability statement missing")
require("BYOD privacy boundary" in main_all and "BYOD privacy boundary" in inf_all, "data locality/privacy statement missing")
require("Required inference feature columns:" in main_code, "main does not print inference schema")
require("Required feature columns:" in inf_code and "Expected inference schema:" in inf_code, "artifact does not surface schema before upload")
require("## What a successful run proves" in main_all and "## What a successful run proves" in inf_all, "interpretation/limits ending missing")

for marker in ("MAX_BASE_ZIP_EXPANDED_BYTES", "Ambiguous backslash ZIP member", "MAX_ARTIFACT_EXPANDED_BYTES", '"payloadFiles"', '"sizes"', '"runtime"', "Unexpected or missing artifact files"):
    require(marker in main_code, f"main hardening marker missing {marker!r}")
for marker in ("MAX_EXPANDED_BYTES", "Ambiguous backslash ZIP member", "payloadFiles", "Unexpected or missing artifact files", "Checkpoint size mismatch", "Training-context size mismatch", "EXPECTED_ZIP_SHA256", "manifest_member_path"):
    require(marker in inf_code, f"artifact hardening marker missing {marker!r}")

for marker in ('tabicl==2.1.1', "EXPECTED_ZIP_SHA256", "safe_extract_zip", "allow_auto_download=False", "trainingContext", "checkpointSha256", "trainingContextSha256", "read_inference_csv", "Inference CSV contains duplicate column names", "manifest_member_path", "rel.is_absolute()"):
    require(marker in inf_code, f"inference code missing {marker!r}")
require("FinetunedTabICLRegressor" not in inf_code, "artifact notebook must not fine-tune")

for label, content in (("main", main_all), ("artifact", inf_all)):
    for token in ("TODO", "TBD", "FIXME"):
        require(re.search(rf"\b{token}\b", content) is None, f"{label} contains {token}")
    require("C:\\Users\\" not in content and "/home/" not in content, f"{label} contains developer-local path")

registry = README.read_text(encoding="utf-8")
require("| `E2E` |" in registry and "| `ARTIFACT-INFERENCE` |" in registry, "tutorial registry missing profiles")
require("Notebook specification:** `1.0`" in registry, "tutorial registry missing spec version")
require("release-notebook-execution" in registry, "release execution status not documented")

req_lines = [line.strip() for line in REQ.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
require(all("==" in line for line in req_lines), "release requirements contain non-exact pins")

print("NOTEBOOK_SPEC 1.0 static conformance: OK")
