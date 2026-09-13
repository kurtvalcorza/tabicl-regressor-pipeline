#!/usr/bin/env python3
"""Repository-specific static checks for the standalone tutorials (NOTEBOOK_SPEC 1.1).

The carrier, parity, hygiene and profile checks live in ``tools/validate_release_assets.py`` (run first). This
script keeps the invariants that are specific to this repository's release contract: the exact PEP 517 build
backend, the release requirement/lock files and their agreement with the ``pyproject.toml`` runtime pins the
notebooks carry, the fine-tuning gate, the serving-bundle hardening markers, and the executor's evidence contract.
It claims no runtime execution evidence.
"""
# ruff: noqa: E501  -- rule messages name the requirement in full; they are kept on one line
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials" / "tabiclv2_regressor_colab.ipynb"
INFERENCE = ROOT / "tutorials" / "tabiclv2_regressor_artifact_inference_colab.ipynb"
README = ROOT / "tutorials" / "README.md"
REQ = ROOT / "tutorials" / "requirements-release.txt"
LOCK = ROOT / "tutorials" / "requirements-release.lock"
PYPROJECT = ROOT / "pyproject.toml"
API = ROOT / "src" / "tabicl_regressor_pipeline" / "api.py"
EXECUTOR = ROOT / "scripts" / "execute_notebook_release.py"
BUILD_BACKEND_PIN = "setuptools==78.1.0"
CHECKPOINT_NAME = "tabicl-regressor-v2-20260212.ckpt"
MODEL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINT_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
ARTIFACT_FORMAT = "tabicl-dimer-regressor-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_nb(path: Path) -> dict:
    require(path.exists(), f"missing notebook {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def text(nb: dict) -> str:
    return "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"])


def code_text(nb: dict, *, outside_modules: bool = False) -> str:
    parts = []
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        if outside_modules and c.get("metadata", {}).get("dimer", {}).get("embedded_module"):
            continue
        parts.append("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
    return "\n".join(parts)


def sys_path_src() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


main_nb, inf_nb = load_nb(MAIN), load_nb(INFERENCE)
main_all, inf_all = text(main_nb), text(inf_nb)
main_own, inf_own = code_text(main_nb, outside_modules=True), code_text(inf_nb, outside_modules=True)

# Profiles / learning contract (the carrier-level checks are the validator's).
for nb, profile, label in ((main_nb, "E2E", "main"), (inf_nb, "ARTIFACT-INFERENCE", "artifact")):
    dimer = nb.get("metadata", {}).get("dimer", {})
    require(dimer.get("notebook_profile") == profile, f"{label} profile metadata")
    require(dimer.get("notebook_spec") == "1.1" and dimer.get("standalone") is True, f"{label} must be standalone spec 1.1")
for marker in ("**Profile:** `E2E`", "**This notebook is standalone.**", "## Interpretation and limits"):
    require(marker in main_all, f"main missing {marker!r}")
for marker in ("**Profile:** `ARTIFACT-INFERENCE`", "**This notebook is standalone.**", "## Interpretation and limits"):
    require(marker in inf_all, f"artifact missing {marker!r}")

# Reproducible environment: the notebooks carry pyproject's pins; the release files stay consistent with them.
require(REQ.exists() and LOCK.exists(), "release requirements/lock missing")
req_lines = [line.strip() for line in REQ.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
lock_lines = [line.strip() for line in LOCK.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
require(all("==" in line for line in req_lines), "top-level release requirements contain non-exact pins")
require(all("==" in line for line in lock_lines), "transitive release lock contains non-exact pins")
locked = {line.split("==", 1)[0].lower().replace("_", "-"): line.split("==", 1)[1] for line in lock_lines}
for package in ("tabicl", "numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "huggingface-hub", "transformers", "wandb", "lightgbm", "einops"):
    require(package in locked, f"transitive lock missing {package}")
require(BUILD_BACKEND_PIN in lock_lines, "release lock must contain the exact PEP 517 build backend")
pyproject = PYPROJECT.read_text(encoding="utf-8")
require(f'requires = ["{BUILD_BACKEND_PIN}"]' in pyproject, "PEP 517 build backend must be exact-pinned to the release lock")
require('build-backend = "setuptools.build_meta"' in pyproject, "unexpected PEP 517 build backend")
require("setuptools>=" not in pyproject and "setuptools~=" not in pyproject, "floating setuptools build requirement is forbidden")
deps_block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", pyproject, re.M | re.S)
require(deps_block is not None, "pyproject.toml must declare [project].dependencies (the notebooks' PINS)")
pins = {p.split("==", 1)[0].lower().replace("_", "-"): p.split("==", 1)[1] for p in re.findall(r'"([^"]+)"', deps_block.group(1))}
require(all("==" in p for p in re.findall(r'"([^"]+)"', deps_block.group(1))), "pyproject runtime deps must be == pinned")
for name, version in pins.items():
    if name in locked:
        require(locked[name] == version, f"pyproject pin {name}=={version} disagrees with requirements-release.lock ({locked[name]})")
require(pins.get("torch") == "2.11.0", "torch must be pinned to the release-verified 2.11.0")
pins_cell = re.search(r"^PINS = \[(.*?)^\]", code_text(main_nb), re.M | re.S)
require(pins_cell is not None and "'torch==2.11.0'" in pins_cell.group(1), "main notebook must carry torch==2.11.0")

# G2: core model operations flow through the carried public API; no direct tabicl use in the notebooks' own cells.
api_source = API.read_text(encoding="utf-8")
for marker in ("def create_regressor", "def create_finetuned_regressor", "def fine_tune_regressor", "def condition_regressor", "def predict_points", "def validate_artifact_runtime", "class TabICLRegressionPipeline", "def safe_extract_zip", "def verify_artifact_bundle", "def manifest_member_path", "def validate_inputs", "def evaluation_report"):
    require(marker in api_source, f"public API missing {marker}")
for own, label in ((main_own, "main"), (inf_own, "artifact")):
    require("from tabicl import" not in own and "import tabicl" not in own, f"{label} bypasses the carried API with a direct tabicl import")
    require("from tabicl_regressor_pipeline" not in own, f"{label} imports the repository package (ST1)")
    require("TabICLRegressionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR" in own, f"{label} does not resolve the model through the carried API")
for marker in ("create_finetuned_regressor(", "fine_tune_regressor(", "pipe.fit(X_train, y_train)", "ACTIVE_MODEL.predict(", "compare_metric("):
    require(marker in main_own, f"main does not exercise repository API operation {marker}")
for marker in ("create_regressor(", "serving.fit(", "serving.predict(", "validate_artifact_runtime(", "safe_extract_zip(", "verify_artifact_bundle("):
    require(marker in inf_own, f"artifact does not exercise repository API operation {marker}")

# Optional fine-tuning is gated off by default, so statically protect its callable path.
require("RUN_FINE_TUNING = False  # @param" in main_own, "fine-tune gate must default to False")
require("finetuner = create_finetuned_regressor(" in main_own, "optional fine-tune constructor does not use repository API")
require(
    "raise RuntimeError('TabICLv2 fine-tuning requires CUDA')\n    ft_dir = Path('outputs') / 'finetune'" in main_own,
    "fine-tune workspace must be initialized after the CUDA guard, not inside its failure branch",
)
require("MIN_SELECTION_HOLDOUT_ROWS = 50" in main_own, "selection evidence guard missing")

# Model / task provenance and semantics.
for marker in (CHECKPOINT_NAME, MODEL_REVISION, CHECKPOINT_SHA256, ARTIFACT_FORMAT):
    require(marker in main_all, f"main missing provenance marker {marker!r}")
for marker in ("'artifactFormat': ARTIFACT_FORMAT", "training_context.parquet", "checkpoints/best.ckpt", "artifact.json", "RUN_NEW_DATA_INFERENCE = False  # @param"):
    require(marker in main_own, f"main code missing {marker!r}")
for marker in ("regression_metrics(", "training_mean_baseline(", "Regression target must vary", "trivial baseline"):
    require(marker in main_all, f"regression requirement missing {marker!r}")
require("no per-prediction uncertainty" in main_all, "main uncertainty boundary missing")
require("Reproducibility boundary" in main_all, "main variability statement missing")
require("BYOD privacy boundary" in main_all, "data locality/privacy statement missing")
require("Uploaded inputs remain in the notebook runtime" in inf_all, "artifact data locality statement missing")
require("'required_features': FEATURE_COLUMNS" in inf_own, "artifact does not surface schema before upload")

# Artifact provenance must establish runtime compatibility before checkpoint reconstruction.
require("validate_artifact_runtime(" in inf_own, "artifact runtime compatibility is not validated")
require(inf_own.index("validate_artifact_runtime(") < inf_own.index("create_regressor("), "runtime compatibility must be established before model reconstruction")
require(inf_own.index("verify_artifact_bundle(") < inf_own.index("create_regressor("), "bundle digests must be verified before model reconstruction")

# Archive / artifact hardening lives in the carried module; the notebooks must use it and never extract raw.
for marker in ("MAX_ARTIFACT_EXPANDED_BYTES", "Ambiguous backslash ZIP member", "Unexpected or missing artifact files", "Checkpoint size mismatch", "Training-context size mismatch", "def manifest_member_path"):
    require(marker in api_source, f"api hardening marker missing {marker!r}")
for own, label in ((main_own, "main"), (inf_own, "artifact")):
    require(".extractall(" not in own and "zipfile.ZipFile(" not in own, f"{label} must extract only through safe_extract_zip")
for marker in ("'payloadFiles'", "'sizes'", "'runtime'", "safe_extract_zip(", "verify_artifact_bundle("):
    require(marker in main_own, f"main hardening marker missing {marker!r}")
require("EXPECTED_ZIP_SHA256" in inf_own, "artifact hardening marker missing EXPECTED_ZIP_SHA256")

# Source hygiene.
for label, content in (("main", main_all), ("artifact", inf_all)):
    for token in ("TODO", "TBD", "FIXME"):
        require(re.search(rf"\b{token}\b", content) is None, f"{label} contains {token}")
    require("C:\\Users\\" not in content and "/home/" not in content, f"{label} contains developer-local path")

# Registry.
registry = README.read_text(encoding="utf-8")
require("| `E2E` |" in registry and "| `ARTIFACT-INFERENCE` |" in registry, "tutorial registry missing profiles")
require("DIMER Notebook Specification 1.1" in registry, "tutorial registry missing spec version")
require("requirements-release.lock" in registry, "tutorial registry does not document transitive lock")

# Executor evidence contract (the previous pair's kernel executor; must not fake execution).
executor = EXECUTOR.read_text(encoding="utf-8")
for marker in ("NotebookClient", "client.execute()", "make_genuinely_new_rows", "second fresh kernel"):
    require(marker in executor, f"release executor missing notebook-engine evidence marker {marker!r}")
require("exec(compile(" not in executor, "release verifier still executes notebook cells as plain Python")
require("startswith(\"%\")" not in executor, "release verifier still strips notebook magics")

print("NOTEBOOK_SPEC 1.1 repository-specific static conformance: OK")
print("NOTE: static validation is not clean-runtime execution evidence.")
