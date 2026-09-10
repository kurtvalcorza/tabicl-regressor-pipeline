from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "tutorials/tabiclv2_regressor_colab.ipynb"
ARTIFACT_PATH = ROOT / "tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb"
README_PATH = ROOT / "tutorials/README.md"
VALIDATOR_PATH = ROOT / "scripts/validate_colab_tutorial.py"
EXECUTOR_PATH = ROOT / "scripts/execute_notebook_release.py"
REQ_PATH = ROOT / "tutorials/requirements-release.txt"
CI_PATH = ROOT / ".github/workflows/ci.yml"


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value


def find_cell(nb: dict, needle: str, cell_type: str | None = None) -> dict:
    for cell in nb["cells"]:
        if cell_type and cell.get("cell_type") != cell_type:
            continue
        if needle in source(cell):
            return cell
    raise RuntimeError(f"Notebook cell not found: {needle!r}")


def append_once(text: str, marker: str, addition: str) -> str:
    if marker in text:
        return text
    return text.rstrip() + "\n\n" + addition.strip() + "\n"


# ---------------------------------------------------------------------------
# Main E2E notebook
# ---------------------------------------------------------------------------
main = json.loads(MAIN_PATH.read_text(encoding="utf-8"))
intro = find_cell(main, "# TabICLv2 Regressor — standalone Google Colab", "markdown")
intro_text = source(intro)
profile_block = """**Profile:** `E2E`  
**Notebook spec:** DIMER Notebook Specification `1.0`  
**Capability:** end-to-end TabICLv2 tabular regression: immutable checkpoint acquisition, validated support data, in-context evaluation, optional gradient fine-tuning, new-data point inference, deployable artifact export, and fresh-boundary reload verification.

**API boundary.** This repository is the DIMER contract/docs umbrella for TabICLv2 regression. Its supported notebook-facing API is the same public top-level `tabicl.TabICLRegressor` / `tabicl.FinetunedTabICLRegressor` estimator surface and `tabicl-dimer-regressor-v1` artifact contract documented by this repository and used by the serving guidance. The notebook does not reimplement TabICL model logic and does not launch the sibling DIMER container repositories.

**This notebook does not demonstrate:** classification, forecasting, causal inference, calibrated prediction intervals, or a production-readiness assessment. Point predictions are estimates only; downstream users own deployment-specific validation and risk controls."""
if "**Profile:** `E2E`" not in intro_text:
    first_break = intro_text.find("\n\n")
    intro_text = intro_text[: first_break + 2] + profile_block + "\n\n" + intro_text[first_break + 2 :]
set_source(intro, intro_text)

install = find_cell(main, "%pip -q install", "code")
set_source(
    install,
    '''%pip -q install "tabicl[finetune]==2.1.1" "lightgbm==4.7.0" "pyarrow==25.0.1" "pandas==2.3.3" "scikit-learn==1.9.0" "huggingface_hub==1.30.0" "transformers==5.6.2" "wandb==0.27.2"

import importlib.metadata
import sys

import torch

PINNED_PACKAGES = {
    "tabicl": "2.1.1",
    "lightgbm": "4.7.0",
    "pyarrow": "25.0.1",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.0",
    "huggingface-hub": "1.30.0",
    "transformers": "5.6.2",
    "wandb": "0.27.2",
}
for package, expected in PINNED_PACKAGES.items():
    observed = importlib.metadata.version(package)
    if observed != expected:
        raise RuntimeError(f"Unexpected {package} version: {observed}; expected {expected}")

# PyTorch is supplied by the supported Colab/runtime image rather than replaced
# in-process. Replacing an imported core framework can require a kernel restart,
# so release verification pins it by inspection instead.
EXPECTED_TORCH_VERSION = "2.11.0"
observed_torch = torch.__version__.split("+")[0]
if observed_torch != EXPECTED_TORCH_VERSION:
    raise RuntimeError(
        f"This tutorial is release-verified with torch=={EXPECTED_TORCH_VERSION}; "
        f"this runtime has {torch.__version__}. Start a supported runtime (or install "
        "the pinned framework before importing it and restart the kernel)."
    )

TABICL_VERSION = PINNED_PACKAGES["tabicl"]
print("Python:", sys.version.split()[0])
print("TabICL:", TABICL_VERSION)
print("PyTorch:", torch.__version__)
print("pandas:", importlib.metadata.version("pandas"))
print("scikit-learn:", importlib.metadata.version("scikit-learn"))
print("pyarrow:", importlib.metadata.version("pyarrow"))
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
''',
)

setup_md = find_cell(main, "## 1. Install and inspect the runtime", "markdown")
setup_text = source(setup_md)
setup_text = setup_text.replace(
    "`tabicl[finetune]` brings the in-context regressor and the fine-tuning trainer; the version is pinned and checked, because the checkpoint format and the estimator API changed between releases.",
    "The tutorial installs an exact release set for TabICL and its notebook-level dependencies. PyTorch is treated as a runtime-provided core framework: the notebook verifies the exact release-validated version instead of replacing it after the kernel has started, avoiding an ambiguous restart boundary."
)
set_source(setup_md, setup_text)

acquire_code = find_cell(main, "def safe_single_ckpt", "code")
acquire_text = source(acquire_code)
if "MAX_BASE_ZIP_EXPANDED_BYTES" not in acquire_text:
    acquire_text = acquire_text.replace(
        'WORK_DIR = Path("/content/tabiclv2-regressor")\nWORK_DIR.mkdir(parents=True, exist_ok=True)\n',
        'WORK_DIR = Path("/content/tabiclv2-regressor")\nWORK_DIR.mkdir(parents=True, exist_ok=True)\nMAX_BASE_ZIP_EXPANDED_BYTES = 512 * 1024 * 1024\n'
    )
    acquire_text = acquire_text.replace(
        '    with zipfile.ZipFile(zip_path) as archive:\n        checkpoint_members = []\n        for info in archive.infolist():\n            name = info.filename.replace("\\\\", "/")\n',
        '    with zipfile.ZipFile(zip_path) as archive:\n        checkpoint_members = []\n        expanded_bytes = 0\n        for info in archive.infolist():\n            if "\\\\" in info.filename:\n                raise ValueError(f"Ambiguous backslash ZIP member: {info.filename}")\n            expanded_bytes += info.file_size\n            if expanded_bytes > MAX_BASE_ZIP_EXPANDED_BYTES:\n                raise ValueError("DIMER checkpoint ZIP exceeds 512 MiB expanded-size limit")\n            name = info.filename\n'
    )
set_source(acquire_code, acquire_text)

data_md = find_cell(main, "## 3. Load sample or BYOD data", "markdown")
data_text = append_once(
    source(data_md),
    "BYOD privacy boundary",
    """**BYOD privacy boundary.** Uploaded CSV bytes are read inside the current notebook runtime and are not sent by this notebook to an external inference or training service. The upstream network request in Step 2 downloads model bytes; it does not upload your dataset. Google Colab is itself a hosted environment, so do not upload confidential, restricted, personal, or otherwise sensitive data unless your authorization and data-handling rules permit processing it there.""",
)
set_source(data_md, data_text)

eval_md = find_cell(main, "## 4. Evaluate pretrained TabICLv2", "markdown")
eval_text = append_once(
    source(eval_md),
    "Reproducibility boundary",
    """**Reproducibility boundary.** The tutorial fixes the data split seed and TabICL ensemble seed. Those controls make the demonstrated partitioning and estimator configuration repeatable, but they do not promise bitwise-identical floating-point results across different PyTorch/CUDA builds, GPU models, or fine-tuning kernels. Treat small metric differences across hardware as runtime variability unless they exceed the evaluation tolerance relevant to your use case.""",
)
set_source(eval_md, eval_text)

infer_md = find_cell(main, "## 5. Optional new-data point prediction", "markdown")
infer_text = append_once(
    source(infer_md),
    "Required schema and privacy",
    """**Required schema and privacy.** Immediately before an upload, the next cell prints the exact `FEATURE_COLUMNS` learned from the support data. Your CSV must contain those names; extra identifier columns are preserved. Uploaded rows remain in the current notebook runtime and are not sent to an external model service. Avoid confidential or restricted data in hosted notebook environments unless authorized.""",
)
set_source(infer_md, infer_text)

infer_code = find_cell(main, "RUN_NEW_DATA_INFERENCE = False", "code")
infer_code_text = source(infer_code)
if 'print("Required inference feature columns:"' not in infer_code_text:
    infer_code_text = infer_code_text.replace(
        'if RUN_NEW_DATA_INFERENCE:\n    uploaded = files.upload()\n',
        'if RUN_NEW_DATA_INFERENCE:\n    print("Required inference feature columns:", FEATURE_COLUMNS)\n    print("Upload stays in this notebook runtime; no external inference service is called.")\n    uploaded = files.upload()\n'
    )
set_source(infer_code, infer_code_text)

export_code = find_cell(main, "ARTIFACT_DIR = Path", "code")
export_text = source(export_code)
if '"payloadFiles"' not in export_text:
    export_text = export_text.replace(
        'train_encoded[FEATURE_COLUMNS + [TARGET_COLUMN]].to_parquet(context_path, index=False)\n\n# Key/value pairs',
        'train_encoded[FEATURE_COLUMNS + [TARGET_COLUMN]].to_parquet(context_path, index=False)\n\npayload_files = ["checkpoints/best.ckpt", "training_context.parquet"]\npayload_sizes = {\n    "checkpoint": export_ckpt.stat().st_size,\n    "trainingContext": context_path.stat().st_size,\n}\n\n# Key/value pairs'
    )
    export_text = export_text.replace(
        '    "digests":{"checkpointSha256":sha256_file(export_ckpt),"trainingContextSha256":sha256_file(context_path)},\n',
        '    "digests":{"checkpointSha256":sha256_file(export_ckpt),"trainingContextSha256":sha256_file(context_path)},\n    "sizes":payload_sizes,"payloadFiles":payload_files,\n    "runtime":{"pythonVersion":sys.version.split()[0],"torchVersion":torch.__version__,\n               "pandasVersion":importlib.metadata.version("pandas"),\n               "pyarrowVersion":importlib.metadata.version("pyarrow"),\n               "scikitLearnVersion":importlib.metadata.version("scikit-learn"),"device":DEVICE},\n'
    )
set_source(export_code, export_text)

reload_code = find_cell(main, "RELOAD_DIR = Path", "code")
reload_text = source(reload_code)
if "MAX_ARTIFACT_EXPANDED_BYTES" not in reload_text:
    reload_text = reload_text.replace(
        'RELOAD_DIR.mkdir()\n\nwith zipfile.ZipFile(archive_path) as archive:\n',
        'RELOAD_DIR.mkdir()\nMAX_ARTIFACT_EXPANDED_BYTES = 1024 * 1024 * 1024\n\nwith zipfile.ZipFile(archive_path) as archive:\n'
    )
    reload_text = reload_text.replace(
        '    root = RELOAD_DIR.resolve()\n    for info in archive.infolist():\n        name = info.filename.replace("\\\\", "/")\n',
        '    root = RELOAD_DIR.resolve()\n    expanded_bytes = 0\n    for info in archive.infolist():\n        if "\\\\" in info.filename:\n            raise ValueError(f"Ambiguous backslash artifact member: {info.filename}")\n        expanded_bytes += info.file_size\n        if expanded_bytes > MAX_ARTIFACT_EXPANDED_BYTES:\n            raise ValueError("Artifact exceeds 1 GiB expanded-size limit")\n        name = info.filename\n'
    )
    reload_text = reload_text.replace(
        'served_context = RELOAD_DIR / served["trainingContext"]\nif sha256_file(served_ckpt)',
        'served_context = RELOAD_DIR / served["trainingContext"]\nexpected_files = {"artifact.json", *served.get("payloadFiles", [served["checkpoint"], served["trainingContext"]])}\nactual_files = {p.relative_to(RELOAD_DIR).as_posix() for p in RELOAD_DIR.rglob("*") if p.is_file()}\nif actual_files != expected_files:\n    raise RuntimeError(f"Unexpected or missing artifact files: {sorted(actual_files ^ expected_files)}")\nif served.get("sizes"):\n    if served_ckpt.stat().st_size != served["sizes"]["checkpoint"] or served_context.stat().st_size != served["sizes"]["trainingContext"]:\n        raise RuntimeError("Artifact size mismatch")\nif sha256_file(served_ckpt)'
    )
set_source(reload_code, reload_text)

main.setdefault("metadata", {})["dimer"] = {"notebook_profile": "E2E", "notebook_spec": "1.0"}

# ---------------------------------------------------------------------------
# Artifact-inference notebook
# ---------------------------------------------------------------------------
artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
art_intro = find_cell(artifact, "# TabICLv2 Regressor artifact inference", "markdown")
art_intro_text = source(art_intro)
art_profile = """**Profile:** `ARTIFACT-INFERENCE`  
**Notebook spec:** DIMER Notebook Specification `1.0`  
**Capability:** validate an externally supplied `tabicl-dimer-regressor-v1` serving artifact, reconstruct its in-context serving state, score genuinely new tabular rows, and export point predictions.

**API boundary.** This umbrella repository defines the artifact and serving contract; reconstruction uses the same public top-level `tabicl.TabICLRegressor` API documented by the repository with `allow_auto_download=False`. No model logic is reimplemented in the notebook.

**This notebook does not train or fine-tune**, does not perform classification or forecasting, and does not provide calibrated per-prediction uncertainty. `fit(context_X, context_y)` restores the support context required by the in-context model; it is not a gradient update."""
if "**Profile:** `ARTIFACT-INFERENCE`" not in art_intro_text:
    first_break = art_intro_text.find("\n\n")
    art_intro_text = art_intro_text[: first_break + 2] + art_profile + "\n\n" + art_intro_text[first_break + 2 :]
set_source(art_intro, art_intro_text)

art_install = find_cell(artifact, "%pip -q install", "code")
set_source(
    art_install,
    '''%pip -q install "tabicl==2.1.1" "pyarrow==25.0.1" "pandas==2.3.3" "scikit-learn==1.9.0"

import importlib.metadata
import sys

import torch

PINNED_PACKAGES = {
    "tabicl": "2.1.1",
    "pyarrow": "25.0.1",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.0",
}
for package, expected in PINNED_PACKAGES.items():
    observed = importlib.metadata.version(package)
    if observed != expected:
        raise RuntimeError(f"Unexpected {package} version: {observed}; expected {expected}")

EXPECTED_TORCH_VERSION = "2.11.0"
observed_torch = torch.__version__.split("+")[0]
if observed_torch != EXPECTED_TORCH_VERSION:
    raise RuntimeError(
        f"This tutorial is release-verified with torch=={EXPECTED_TORCH_VERSION}; "
        f"this runtime has {torch.__version__}. Start a supported runtime."
    )

TABICL_VERSION = PINNED_PACKAGES["tabicl"]
print("Python:", sys.version.split()[0])
print("TabICL:", TABICL_VERSION)
print("PyTorch:", torch.__version__)
print("pandas:", importlib.metadata.version("pandas"))
print("scikit-learn:", importlib.metadata.version("scikit-learn"))
print("pyarrow:", importlib.metadata.version("pyarrow"))
print("CUDA:", torch.cuda.is_available())
''',
)

art_install_md = find_cell(artifact, "## 1. Install", "markdown")
art_install_text = source(art_install_md).replace(
    "`tabicl` is pinned to the version the bundle was produced with; the manifest's recorded version is checked against it in Step 2, because the checkpoint format is not stable across versions.",
    "The notebook installs exact notebook-level package versions and verifies the release-tested PyTorch framework version before any artifact is deserialized. The manifest's recorded TabICL version is checked against this runtime because checkpoint compatibility is version-sensitive."
)
set_source(art_install_md, art_install_text)

art_verify = find_cell(artifact, "def safe_extract_zip", "code")
art_verify_text = source(art_verify)
if "MAX_EXPANDED_BYTES" not in art_verify_text:
    art_verify_text = art_verify_text.replace(
        'EXPECTED_ZIP_SHA256 = ""  # @param {type:"string"}\n',
        'EXPECTED_ZIP_SHA256 = ""  # @param {type:"string"}\nMAX_EXPANDED_BYTES = 1024 * 1024 * 1024\n'
    )
    art_verify_text = art_verify_text.replace(
        '    with zipfile.ZipFile(zip_path) as archive:\n        for info in archive.infolist():\n            name = info.filename.replace("\\\\", "/")\n',
        '    with zipfile.ZipFile(zip_path) as archive:\n        expanded_bytes = 0\n        for info in archive.infolist():\n            if "\\\\" in info.filename:\n                raise ValueError(f"Ambiguous backslash ZIP member: {info.filename}")\n            expanded_bytes += info.file_size\n            if expanded_bytes > MAX_EXPANDED_BYTES:\n                raise ValueError("Artifact exceeds 1 GiB expanded-size limit")\n            name = info.filename\n'
    )
    art_verify_text = art_verify_text.replace(
        'context_path = manifest_member_path(root, manifest["trainingContext"], "trainingContext")\nif sha256_file(ckpt)',
        'context_path = manifest_member_path(root, manifest["trainingContext"], "trainingContext")\npayload_files = manifest.get("payloadFiles")\nif payload_files is not None:\n    expected_files = {"artifact.json", *payload_files}\n    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}\n    if actual_files != expected_files:\n        raise RuntimeError(f"Unexpected or missing artifact files: {sorted(actual_files ^ expected_files)}")\nelse:\n    print("⚠ Legacy manifest has no payloadFiles allowlist; unexpected-file rejection cannot be proven.")\nif manifest.get("sizes"):\n    if ckpt.stat().st_size != manifest["sizes"]["checkpoint"]:\n        raise RuntimeError("Checkpoint size mismatch")\n    if context_path.stat().st_size != manifest["sizes"]["trainingContext"]:\n        raise RuntimeError("Training-context size mismatch")\nelse:\n    print("⚠ Legacy manifest has no recorded payload sizes; digest checks still apply.")\nif sha256_file(ckpt)'
    )
set_source(art_verify, art_verify_text)

art_reconstruct = find_cell(artifact, "## 3. Reconstruct the in-context regressor", "markdown")
art_reconstruct_text = append_once(
    source(art_reconstruct),
    "schema is printed",
    """Before the notebook asks for inference data, the reconstruction cell prints the exact ordered feature schema carried by the artifact. This is the schema the producer used at serving time; it is not inferred from the uploaded inference CSV.""",
)
set_source(art_reconstruct, art_reconstruct_text)

art_reconstruct_code = find_cell(artifact, "FEATURE_COLUMNS = manifest", "code")
art_reconstruct_code_text = source(art_reconstruct_code)
if 'print("Required feature columns:"' not in art_reconstruct_code_text:
    art_reconstruct_code_text = art_reconstruct_code_text.replace(
        'print(f"✓ Loaded with {len(context)} context rows and {len(FEATURE_COLUMNS)} features")',
        'print(f"✓ Loaded with {len(context)} context rows and {len(FEATURE_COLUMNS)} features")\nprint("Required feature columns:", FEATURE_COLUMNS)\nprint("Target recorded by artifact:", TARGET_COLUMN)'
    )
set_source(art_reconstruct_code, art_reconstruct_code_text)

art_predict_md = find_cell(artifact, "## 4. Upload rows and predict", "markdown")
art_predict_text = append_once(
    source(art_predict_md),
    "BYOD privacy boundary",
    """**BYOD privacy boundary.** The uploaded CSV is read and scored inside the current notebook runtime; this notebook does not send rows to an external inference service. Google Colab is a hosted environment, so do not upload confidential, restricted, personal, or otherwise sensitive data unless your authorization and data-handling rules permit processing it there. The exact required feature names were printed in Step 3 before this upload prompt.""",
)
set_source(art_predict_md, art_predict_text)

art_predict_code = find_cell(artifact, "Upload exactly one inference CSV", "code")
art_predict_code_text = source(art_predict_code)
if 'print("Expected inference schema:"' not in art_predict_code_text:
    art_predict_code_text = art_predict_code_text.replace(
        'uploaded = files.upload()\nif len(uploaded) != 1:',
        'print("Expected inference schema:", FEATURE_COLUMNS)\nprint("Upload stays in this notebook runtime; no external inference service is called.")\nuploaded = files.upload()\nif len(uploaded) != 1:'
    )
set_source(art_predict_code, art_predict_code_text)

final_md = find_cell(artifact, "## When a check fails", "markdown")
final_text = source(final_md)
if "## What a successful run proves" not in final_text:
    interpretation = """## What a successful run proves — and what it does not

A successful run proves that the supplied archive passed this notebook's path/symlink/expanded-size checks, that its declared payload matches the manifest digests (and sizes/allowlist when provided), that the recorded TabICL runtime contract could be reconstructed without automatic model fallback, that the persisted support context was restored, and that the reconstructed model produced a machine-readable `prediction` column for new rows.

It does **not** prove who authored the archive unless you independently trust and verify the whole-archive digest; internal manifest consistency is not sender authenticity. It also does not establish predictive quality, fairness, calibration, robustness, or production fitness on your deployment population. Point predictions carry no per-row uncertainty interval. Validate the artifact on representative labelled data and apply domain-specific risk controls before consequential use.

"""
    final_text = final_text.replace("## When a check fails", interpretation + "## When a check fails")
set_source(final_md, final_text)
artifact.setdefault("metadata", {})["dimer"] = {"notebook_profile": "ARTIFACT-INFERENCE", "notebook_spec": "1.0"}

# Clear persisted state explicitly.
for nb in (main, artifact):
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

MAIN_PATH.write_text(json.dumps(main, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Tutorial registry and environment lock surface
# ---------------------------------------------------------------------------
readme = README_PATH.read_text(encoding="utf-8")
readme = readme.replace(
    "| Notebook | Badge | Purpose |\n|---|---|---|\n| [`tabiclv2_regressor_colab.ipynb`](tabiclv2_regressor_colab.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabicl-regressor-pipeline/blob/main/tutorials/tabiclv2_regressor_colab.ipynb) | End-to-end tutorial: checkpoint → data → evaluation → optional fine-tuning → inference → portable bundle |\n| [`tabiclv2_regressor_artifact_inference_colab.ipynb`](tabiclv2_regressor_artifact_inference_colab.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabicl-regressor-pipeline/blob/main/tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb) | Load a trusted exported/DIMER-style bundle and run inference without gradient fine-tuning |",
    "| Notebook | Profile | Capability | Default runtime | Release status |\n|---|---|---|---|---|\n| [`tabiclv2_regressor_colab.ipynb`](tabiclv2_regressor_colab.ipynb) | `E2E` | Tabular regression: evaluation, optional adaptation, inference, export/reload | CPU (GPU optional for fine-tuning) | candidate — NOTEBOOK_SPEC 1.0 source checks; clean-runtime execution required |\n| [`tabiclv2_regressor_artifact_inference_colab.ipynb`](tabiclv2_regressor_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | External serving-bundle validation/reconstruction and new-data inference | CPU | candidate — NOTEBOOK_SPEC 1.0 source checks; clean-runtime execution required |"
)
registry_note = """## NOTEBOOK_SPEC v1.0 conformance

**Notebook specification:** `1.0`.

This repository is the DIMER contract/docs umbrella. For these tutorials, the repository-defined notebook-facing API is the public top-level `tabicl.TabICLRegressor` / `tabicl.FinetunedTabICLRegressor` estimator surface configured to the immutable checkpoint and `tabicl-dimer-regressor-v1` artifact contract documented here. The notebooks do not reimplement model logic or launch the sibling validator/fine-tuner containers.

The exact notebook-level package pins used for release verification are recorded in [`requirements-release.txt`](requirements-release.txt). PyTorch is a runtime-provided core framework and is verified as `2.11.0` rather than replaced after kernel startup.

Release status remains **candidate** until the current revision's `release-notebook-execution` check passes. Static validation is not treated as execution evidence.

### Durable SHOULD dispositions

- `MOD6` / artifact digests: implemented for the base checkpoint and exported payload.
- `SEC6`: implemented with 512 MiB expanded limit for base-checkpoint ZIP input and 1 GiB for serving artifacts.
- `SEC8` / `SEC9`: newly produced artifacts record payload sizes and an allowlist; the consumer verifies them. Legacy `tabicl-dimer-regressor-v1` bundles without those optional fields remain readable but emit explicit warnings.
- `DAT14`: hosted-runtime sensitive-data warnings are present before BYOD paths.
"""
if "## NOTEBOOK_SPEC v1.0 conformance" not in readme:
    marker = "## Main tutorial"
    readme = readme.replace(marker, registry_note + "\n" + marker)
README_PATH.write_text(readme, encoding="utf-8")

REQ_PATH.write_text(
    """# Release-verified notebook-level dependencies for NOTEBOOK_SPEC 1.0.\n# PyTorch is runtime-provided and verified separately as 2.11.0.\ntabicl==2.1.1\nlightgbm==4.7.0\npyarrow==25.0.1\npandas==2.3.3\nscikit-learn==1.9.0\nhuggingface_hub==1.30.0\ntransformers==5.6.2\nwandb==0.27.2\n""",
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# Strong static conformance validator
# ---------------------------------------------------------------------------
VALIDATOR_PATH.write_text(textwrap.dedent(r'''
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
''').lstrip(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Clean-runtime execution harness: executes current notebook source top-to-bottom
# with Colab file upload/download surface stubbed, then feeds the external artifact
# into the companion notebook from a fresh namespace.
# ---------------------------------------------------------------------------
EXECUTOR_PATH.write_text(textwrap.dedent(r'''
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
''').lstrip(), encoding="utf-8")

CI_PATH.write_text(textwrap.dedent('''
name: CI

on:
  pull_request:
  workflow_dispatch:

jobs:
  tutorial-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install pandas numpy
      - name: NOTEBOOK_SPEC 1.0 static conformance
        run: python scripts/validate_colab_tutorial.py
      - name: Colab CSV header regressions
        run: python scripts/test_colab_csv_headers.py
      - name: Issue 10 tutorial hardening regressions
        run: python scripts/test_issue10_hardening.py
      - name: Compile tutorial checks
        run: python -m compileall -q scripts

  release-notebook-execution:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - name: Install release-verified notebook environment
        run: |
          python -m pip install --upgrade pip
          pip install torch==2.11.0
          pip install -r tutorials/requirements-release.txt
      - name: Execute E2E producer and external-artifact consumer
        env:
          HF_HUB_DISABLE_TELEMETRY: '1'
          WANDB_MODE: disabled
        run: python scripts/execute_notebook_release.py
''').lstrip(), encoding="utf-8")

print("Applied NOTEBOOK_SPEC v1.0 source migration")
