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
LOCK = ROOT / "tutorials/requirements-release.lock"
PYPROJECT = ROOT / "pyproject.toml"
EXECUTOR = ROOT / "scripts/execute_notebook_release.py"
API = ROOT / "src/tabicl_regressor_pipeline/api.py"

MODEL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINT_NAME = "tabicl-regressor-v2-20260212.ckpt"
CHECKPOINT_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
TABICL_VERSION = "2.1.1"
ARTIFACT_FORMAT = "tabicl-dimer-regressor-v1"
BUILD_BACKEND_PIN = "setuptools==78.1.0"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_nb(path: Path) -> dict:
    require(path.exists(), f"missing notebook: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def text(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list) else str(cell.get("source", ""))
        for cell in nb["cells"]
    )


def code_text(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list) else str(cell.get("source", ""))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


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

# Profiles / learning contract.
require(main_nb.get("metadata", {}).get("dimer") == {"notebook_profile": "E2E", "notebook_spec": "1.0"}, "main profile metadata")
require(inf_nb.get("metadata", {}).get("dimer") == {"notebook_profile": "ARTIFACT-INFERENCE", "notebook_spec": "1.0"}, "artifact profile metadata")
for marker in ("**Profile:** `E2E`", "**Notebook spec:** DIMER Notebook Specification `1.0`", "**API boundary.**", "## What a successful run proves"):
    require(marker in main_all, f"main missing {marker!r}")
for marker in ("**Profile:** `ARTIFACT-INFERENCE`", "**Notebook spec:** DIMER Notebook Specification `1.0`", "**API boundary.**", "## What a successful run proves"):
    require(marker in inf_all, f"artifact missing {marker!r}")

# Reproducible environment: direct intent plus a resolved transitive graph.
require(REQ.exists() and LOCK.exists(), "release requirements/lock missing")
req_lines = [line.strip() for line in REQ.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
lock_lines = [line.strip() for line in LOCK.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
require(all("==" in line for line in req_lines), "top-level release requirements contain non-exact pins")
require(all("==" in line for line in lock_lines), "transitive release lock contains non-exact pins")
locked_names = {line.split("==", 1)[0].lower() for line in lock_lines}
for package in ("tabicl", "numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "huggingface_hub", "transformers", "wandb", "lightgbm", "einops"):
    require(package.lower() in locked_names, f"transitive lock missing {package}")
require("torch" not in locked_names, "torch must remain the explicit runtime-provided boundary")
require(BUILD_BACKEND_PIN in lock_lines, "release lock must contain the exact PEP 517 build backend")
pyproject = PYPROJECT.read_text(encoding="utf-8")
require(f'requires = ["{BUILD_BACKEND_PIN}"]' in pyproject, "PEP 517 build backend must be exact-pinned to the release lock")
require('build-backend = "setuptools.build_meta"' in pyproject, "unexpected PEP 517 build backend")
require("setuptools>=" not in pyproject and "setuptools~=" not in pyproject, "floating setuptools build requirement is forbidden")
require('--no-deps "git+https://github.com/kurtvalcorza/tabicl-regressor-pipeline@' in main_code, "main adapter install must disable runtime dependency re-resolution")
require('--no-deps "git+https://github.com/kurtvalcorza/tabicl-regressor-pipeline@' in inf_code, "artifact adapter install must disable runtime dependency re-resolution")
require('EXPECTED_TORCH_VERSION = "2.11.0"' in main_code and 'EXPECTED_TORCH_VERSION = "2.11.0"' in inf_code, "PyTorch runtime verification missing")

# Both notebooks must install the same immutable repository API + lock anchor.
def api_revision(code: str, label: str) -> str:
    matches = set(re.findall(r"tabicl-regressor-pipeline@([0-9a-f]{40})", code))
    require(len(matches) == 1, f"{label} must pin exactly one repository API revision")
    revision = next(iter(matches))
    require(f"/{revision}/tutorials/requirements-release.lock" in code, f"{label} lock URL is not pinned to API revision")
    return revision

main_api_revision = api_revision(main_code, "main")
inf_api_revision = api_revision(inf_code, "artifact")
require(main_api_revision == inf_api_revision, "notebooks pin different repository API revisions")

# G2: core model operations must flow through this repository's public adapter.
api_source = API.read_text(encoding="utf-8")
for marker in ("def create_regressor", "def create_finetuned_regressor", "def fine_tune_regressor", "def condition_regressor", "def predict_points", "def validate_artifact_runtime"):
    require(marker in api_source, f"public API missing {marker}")
for code, label in ((main_code, "main"), (inf_code, "artifact")):
    require("from tabicl_regressor_pipeline import" in code, f"{label} does not import repository API")
    require("from tabicl import" not in code, f"{label} bypasses repository API with direct tabicl import")
for marker in ("create_regressor(", "condition_regressor(", "predict_points(", "fine_tune_regressor("):
    require(marker in main_code, f"main does not exercise repository API operation {marker}")
for marker in ("create_regressor(", "condition_regressor(", "predict_points("):
    require(marker in inf_code, f"artifact does not exercise repository API operation {marker}")

# Optional fine-tuning is gated off by default, so statically protect its callable path.
require("finetuner = create_finetuned_regressor(" in main_code, "optional fine-tune constructor does not use repository API")
require("Finetunedcreate_regressor" not in main_code, "mangled optional fine-tune constructor")
require(
    'raise RuntimeError("TabICLv2 fine-tuning requires CUDA")\n    ft_dir = Path("/content/tabiclv2-regressor-finetune")' in main_code,
    "fine-tune workspace must be initialized after the CUDA guard, not inside its failure branch",
)

# Model / task provenance and semantics.
for marker in (CHECKPOINT_NAME, MODEL_REVISION, CHECKPOINT_SHA256, 'CHECKPOINT_SOURCE = "Pinned upstream"', '"DIMER ZIP"', "RUN_FINE_TUNING = False", "MIN_SELECTION_HOLDOUT_ROWS = 50", ARTIFACT_FORMAT, "training_context.parquet", "checkpoints/best.ckpt", "artifact.json", "Candidate checkpoint holdout"):
    require(marker in main_code, f"main code missing {marker!r}")
for marker in ("mean_absolute_error", "mean_squared_error", "r2_score", "Training regression target must vary", "trivial baseline"):
    require(marker in main_all, f"regression requirement missing {marker!r}")
require("no uncertainty band" in main_all or "no per-prediction uncertainty" in main_all, "main uncertainty boundary missing")
require("Reproducibility boundary" in main_all, "main variability statement missing")
require("BYOD privacy boundary" in main_all and "BYOD privacy boundary" in inf_all, "data locality/privacy statement missing")
require("Required inference feature columns:" in main_code, "main does not print inference schema")
require("Required feature columns:" in inf_code and "Expected inference schema:" in inf_code, "artifact does not surface schema before upload")

# Artifact provenance must establish runtime compatibility before checkpoint reconstruction.
require("validate_artifact_runtime(" in inf_code, "artifact runtime compatibility is not validated")
require("Artifact/runtime provenance and compatibility:" in inf_code, "artifact runtime provenance is not displayed")
require(inf_code.index("validate_artifact_runtime(") < inf_code.index("create_regressor("), "runtime compatibility must be established before model reconstruction")

# Archive / artifact hardening.
for marker in ("MAX_BASE_ZIP_EXPANDED_BYTES", "Ambiguous backslash ZIP member", "MAX_ARTIFACT_EXPANDED_BYTES", '"payloadFiles"', '"sizes"', '"runtime"', "Unexpected or missing artifact files"):
    require(marker in main_code, f"main hardening marker missing {marker!r}")
for marker in ("MAX_EXPANDED_BYTES", "Ambiguous backslash ZIP member", "payloadFiles", "Unexpected or missing artifact files", "Checkpoint size mismatch", "Training-context size mismatch", "EXPECTED_ZIP_SHA256", "manifest_member_path"):
    require(marker in inf_code, f"artifact hardening marker missing {marker!r}")

# Source hygiene.
for label, content in (("main", main_all), ("artifact", inf_all)):
    for token in ("TODO", "TBD", "FIXME"):
        require(re.search(rf"\b{token}\b", content) is None, f"{label} contains {token}")
    require("C:\\Users\\" not in content and "/home/" not in content, f"{label} contains developer-local path")

# Registry and real notebook-engine verification contract.
registry = README.read_text(encoding="utf-8")
require("| `E2E` |" in registry and "| `ARTIFACT-INFERENCE` |" in registry, "tutorial registry missing profiles")
require("Notebook specification:** `1.0`" in registry, "tutorial registry missing spec version")
require("requirements-release.lock" in registry, "tutorial registry does not document transitive lock")
require(main_api_revision in registry, "tutorial registry does not record immutable repository API revision")

executor = EXECUTOR.read_text(encoding="utf-8")
for marker in ("NotebookClient", "client.execute()", "make_genuinely_new_rows", "DIMER_ARTIFACT_PATH", "DIMER_INFERENCE_CSV_PATH", "second fresh kernel"):
    require(marker in executor, f"release executor missing notebook-engine evidence marker {marker!r}")
require("exec(compile(" not in executor, "release verifier still executes notebook cells as plain Python")
require("startswith(\"%\")" not in executor, "release verifier still strips notebook magics")

print("NOTEBOOK_SPEC 1.0 static conformance: OK")
print("Repository API/lock revision:", main_api_revision)
