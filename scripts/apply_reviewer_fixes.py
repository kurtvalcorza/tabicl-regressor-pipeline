from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_REVISION = os.environ["PIPELINE_API_REVISION"]
LOCK_URL = f"https://raw.githubusercontent.com/kurtvalcorza/tabicl-regressor-pipeline/{API_REVISION}/tutorials/requirements-release.lock"
REPO_INSTALL = f"git+https://github.com/kurtvalcorza/tabicl-regressor-pipeline@{API_REVISION}"


def get_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value


def find_cell(nb: dict, marker: str, *, kind: str | None = None) -> dict:
    for cell in nb["cells"]:
        if kind and cell.get("cell_type") != kind:
            continue
        if marker in get_source(cell):
            return cell
    raise RuntimeError(f"Could not find notebook cell containing {marker!r}")


def patch_main(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    opening = find_cell(nb, "**Profile:** `E2E`", kind="markdown")
    text = get_source(opening)
    text = text.replace(
        "**API boundary.** This repository is the DIMER contract/docs umbrella for TabICLv2 regression. Its supported notebook-facing API is the same public top-level `tabicl.TabICLRegressor` / `tabicl.FinetunedTabICLRegressor` estimator surface and `tabicl-dimer-regressor-v1` artifact contract documented by this repository and used by the serving guidance. The notebook does not reimplement TabICL model logic and does not launch the sibling DIMER container repositories.",
        "**API boundary.** This repository remains the DIMER contract/docs umbrella, and now exposes an installable public reference API in `tabicl_regressor_pipeline`. The notebook installs that repository API at an immutable commit and routes model construction, support conditioning, optional fine-tuning, point prediction, runtime compatibility checks, and notebook I/O through it. The sibling DIMER validator/fine-tuner containers remain the production Workbench implementation; this notebook is the supported repository reference path, not an on-platform acceptance test.",
    )
    text = text.replace(
        "- **Runtime:** any Colab runtime for the default path (pretrained evaluation runs on CPU in seconds on the sample); a **GPU** only if you switch `RUN_FINE_TUNING` on.",
        "- **Runtime:** Google Colab or a clean Jupyter/Python 3.13 runtime with release-verified PyTorch 2.11.0; the default pretrained path is CPU-capable. A **GPU** is required only if you switch `RUN_FINE_TUNING` on.",
    )
    set_source(opening, text)

    install = find_cell(nb, "%pip -q install", kind="code")
    set_source(
        install,
        f'''%pip -q install -r "{LOCK_URL}"\n%pip -q install --no-deps "{REPO_INSTALL}"\n\nimport importlib.metadata\nimport sys\n\nimport torch\nfrom tabicl_regressor_pipeline import (\n    condition_regressor,\n    create_finetuned_regressor,\n    create_regressor,\n    download_output,\n    fine_tune_regressor,\n    predict_points,\n    read_single_input,\n    runtime_identity,\n)\n\nEXPECTED_TORCH_VERSION = "2.11.0"\nobserved_torch = torch.__version__.split("+")[0]\nif observed_torch != EXPECTED_TORCH_VERSION:\n    raise RuntimeError(\n        f"Supported runtime requires torch=={{EXPECTED_TORCH_VERSION}}; this runtime has {{torch.__version__}}. "\n        "Use the documented Colab/Jupyter release runtime before continuing."\n    )\nTABICL_VERSION = "2.1.1"\nidentity = runtime_identity()\nprint("Pipeline API revision:", "{API_REVISION}")\nprint("Python:", identity["pythonVersion"])\nprint("TabICL:", identity["tabiclVersion"])\nprint("PyTorch:", identity["torchVersion"])\nprint("pandas:", importlib.metadata.version("pandas"))\nprint("scikit-learn:", importlib.metadata.version("scikit-learn"))\nprint("pyarrow:", importlib.metadata.version("pyarrow"))\nprint("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")\n''',
    )

    acquire = find_cell(nb, "MODEL_REPO =", kind="code")
    s = get_source(acquire)
    s = s.replace("from google.colab import files\n", "try:\n    from google.colab import files  # type: ignore\nexcept ModuleNotFoundError:\n    files = None\n")
    old = '''    uploaded = files.upload()\n    if len(uploaded) != 1:\n        raise ValueError("Upload exactly one .ckpt or .zip")\n    name, payload = next(iter(uploaded.items()))\n'''
    new = '''    name, payload = read_single_input(\n        env_var="DIMER_CHECKPOINT_PATH", label="checkpoint .ckpt or .zip"\n    )\n'''
    if old not in s:
        raise RuntimeError("main checkpoint upload block changed")
    s = s.replace(old, new)
    set_source(acquire, s)

    core = find_cell(nb, "RUN_FINE_TUNING = False", kind="code")
    s = get_source(core)
    s = s.replace("from tabicl import TabICLRegressor\n", "")
    s = s.replace("from tabicl import FinetunedTabICLRegressor\n", "")
    s = s.replace("TabICLRegressor(", "create_regressor(")
    s = s.replace("FinetunedTabICLRegressor(", "create_finetuned_regressor(")
    s = s.replace("baseline_model.fit(X_train, y_train)", "condition_regressor(baseline_model, X_train, y_train)")
    s = s.replace("candidate_model.fit(X_train, y_train)", "condition_regressor(candidate_model, X_train, y_train)")
    s = s.replace(
        "finetuner.fit(X_train, y_train, X_val=holdout_encoded[FEATURE_COLUMNS], y_val=holdout_encoded[TARGET_COLUMN], output_dir=str(ft_dir))",
        "fine_tune_regressor(finetuner, X_train, y_train, X_val=holdout_encoded[FEATURE_COLUMNS], y_val=holdout_encoded[TARGET_COLUMN], output_dir=str(ft_dir))",
    )
    s = s.replace("np.asarray(model.predict(frame[FEATURE_COLUMNS]), dtype=float)", "predict_points(model, frame[FEATURE_COLUMNS])")
    set_source(core, s)

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = get_source(cell)
        s = s.replace("files.download(str(archive_path))", "download_output(archive_path)")
        s = s.replace("files.download(str(out_path))", "download_output(out_path)")
        s = s.replace("np.asarray(ACTIVE_MODEL.predict(X), dtype=float)", "predict_points(ACTIVE_MODEL, X)")
        s = s.replace("_ = model.predict(X)", "_ = predict_points(model, X)")
        s = s.replace("np.asarray(model.predict(X), dtype=float)", "predict_points(model, X)")
        if "RELOAD_DIR =" in s:
            s = s.replace("TabICLRegressor(", "create_regressor(")
            s = s.replace(
                "reloaded.fit(context[served[\"featureColumns\"]], context[served[\"targetColumn\"]])",
                "condition_regressor(reloaded, context[served[\"featureColumns\"]], context[served[\"targetColumn\"]])",
            )
            s = s.replace("np.asarray(ACTIVE_MODEL.predict(smoke_rows), dtype=float)", "predict_points(ACTIVE_MODEL, smoke_rows)")
            s = s.replace("np.asarray(reloaded.predict(smoke_rows), dtype=float)", "predict_points(reloaded, smoke_rows)")
        if "RUN_NEW_DATA_INFERENCE = False" in s:
            old = '''    uploaded = files.upload()\n    if len(uploaded) != 1:\n        raise ValueError("Upload exactly one CSV")\n    _, payload = next(iter(uploaded.items()))\n'''
            new = '''    _, payload = read_single_input(env_var="DIMER_INFERENCE_CSV_PATH", label="inference CSV")\n'''
            s = s.replace(old, new)
        set_source(cell, s)

    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_inference(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    opening = find_cell(nb, "**Profile:** `ARTIFACT-INFERENCE`", kind="markdown")
    text = get_source(opening)
    text = text.replace(
        "**API boundary.** This umbrella repository defines the artifact and serving contract; reconstruction uses the same public top-level `tabicl.TabICLRegressor` API documented by the repository with `allow_auto_download=False`. No model logic is reimplemented in the notebook.",
        "**API boundary.** This umbrella repository defines the artifact/serving contract and exposes the installable `tabicl_regressor_pipeline` public reference API. This notebook installs that API at an immutable commit and uses it for artifact/runtime compatibility, serving-object construction, support-context restoration, point prediction, and notebook I/O. `allow_auto_download=False` remains enforced.",
    )
    text = text.replace(
        "- Any Colab runtime; inference runs on CPU. About two minutes end to end.",
        "- Google Colab or a clean Jupyter/Python 3.13 runtime with release-verified PyTorch 2.11.0; inference is CPU-capable. About two minutes end to end on the measured release runner.",
    )
    set_source(opening, text)

    install = find_cell(nb, "%pip -q install", kind="code")
    set_source(
        install,
        f'''%pip -q install -r "{LOCK_URL}"\n%pip -q install --no-deps "{REPO_INSTALL}"\n\nimport importlib.metadata\nimport sys\n\nimport torch\nfrom tabicl_regressor_pipeline import (\n    condition_regressor,\n    create_regressor,\n    download_output,\n    predict_points,\n    read_single_input,\n    runtime_identity,\n    validate_artifact_runtime,\n)\n\nEXPECTED_TORCH_VERSION = "2.11.0"\nobserved_torch = torch.__version__.split("+")[0]\nif observed_torch != EXPECTED_TORCH_VERSION:\n    raise RuntimeError(\n        f"Supported runtime requires torch=={{EXPECTED_TORCH_VERSION}}; this runtime has {{torch.__version__}}."\n    )\nTABICL_VERSION = "2.1.1"\nidentity = runtime_identity()\nprint("Pipeline API revision:", "{API_REVISION}")\nprint("Python:", identity["pythonVersion"])\nprint("TabICL:", identity["tabiclVersion"])\nprint("PyTorch:", identity["torchVersion"])\nprint("pandas:", importlib.metadata.version("pandas"))\nprint("scikit-learn:", importlib.metadata.version("scikit-learn"))\nprint("pyarrow:", importlib.metadata.version("pyarrow"))\nprint("CUDA:", torch.cuda.is_available())\n''',
    )

    verify = find_cell(nb, "EXPECTED_ZIP_SHA256", kind="code")
    s = get_source(verify)
    s = s.replace("from google.colab import files\n", "")
    old = '''uploaded = files.upload()\nif len(uploaded) != 1:\n    raise ValueError("Upload exactly one artifact ZIP")\nname, payload = next(iter(uploaded.items()))\n'''
    new = '''name, payload = read_single_input(env_var="DIMER_ARTIFACT_PATH", label="artifact ZIP")\n'''
    if old not in s:
        raise RuntimeError("artifact upload block changed")
    s = s.replace(old, new)
    marker = '''if manifest.get("tabiclVersion") != TABICL_VERSION:\n    raise ValueError("Artifact TabICL version does not match notebook pin")\n\n'''
    addition = marker + '''compatibility = validate_artifact_runtime(\n    manifest, expected_tabicl_version=TABICL_VERSION, expected_torch_version=EXPECTED_TORCH_VERSION\n)\nprint("Artifact/runtime provenance and compatibility:")\nprint(json.dumps(compatibility, indent=2, sort_keys=True))\n\n'''
    if marker not in s:
        raise RuntimeError("artifact version validation marker changed")
    s = s.replace(marker, addition)
    set_source(verify, s)

    reconstruct = find_cell(nb, "from tabicl import TabICLRegressor", kind="code")
    s = get_source(reconstruct)
    s = s.replace("from tabicl import TabICLRegressor\n\n", "")
    s = s.replace("TabICLRegressor(", "create_regressor(")
    s = s.replace("model.fit(context[FEATURE_COLUMNS], context[TARGET_COLUMN])", "condition_regressor(model, context[FEATURE_COLUMNS], context[TARGET_COLUMN])")
    set_source(reconstruct, s)

    predict = find_cell(nb, "Expected inference schema:", kind="code")
    s = get_source(predict)
    old = '''uploaded = files.upload()\nif len(uploaded) != 1:\n    raise ValueError("Upload exactly one inference CSV")\n_, payload = next(iter(uploaded.items()))\n'''
    new = '''_, payload = read_single_input(env_var="DIMER_INFERENCE_CSV_PATH", label="inference CSV")\n'''
    if old not in s:
        raise RuntimeError("inference upload block changed")
    s = s.replace(old, new)
    s = s.replace("np.asarray(model.predict(X), dtype=float)", "predict_points(model, X)")
    s = s.replace("files.download(str(output_path))", "download_output(output_path)")
    set_source(predict, s)

    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "This repository is the DIMER contract/docs umbrella. For these tutorials, the repository-defined notebook-facing API is the public top-level `tabicl.TabICLRegressor` / `tabicl.FinetunedTabICLRegressor` estimator surface configured to the immutable checkpoint and `tabicl-dimer-regressor-v1` artifact contract documented here. The notebooks do not reimplement model logic or launch the sibling validator/fine-tuner containers.",
        f"This repository remains the DIMER contract/docs umbrella and now exposes an installable public reference API under `src/tabicl_regressor_pipeline/`. The tutorials install that API at immutable commit `{API_REVISION}` and exercise it for their core model operations. The sibling validator/fine-tuner containers remain the DIMER Workbench production implementation; tutorial execution is reference-path evidence, not the final on-platform acceptance test.",
    )
    text = text.replace(
        "The exact notebook-level package pins used for release verification are recorded in [`requirements-release.txt`](requirements-release.txt). PyTorch is a runtime-provided core framework and is verified as `2.11.0` rather than replaced after kernel startup.",
        "[`requirements-release.txt`](requirements-release.txt) records the requested top-level environment and [`requirements-release.lock`](requirements-release.lock) records the resolved transitive graph used by the release notebooks. PyTorch is an explicit runtime-provided boundary and is verified as `2.11.0` rather than replaced after kernel startup.",
    )
    text = text.replace(
        "Release status remains **candidate** until the current revision's `release-notebook-execution` check passes. Static validation is not treated as execution evidence.",
        "Release status remains **candidate** until the current revision's notebook-engine execution check passes. Static validation and the legacy plain-Python integration harness are not treated as notebook execution evidence.",
    )
    text = text.replace(
        "Static/CI checks validate notebook structure, pinned checkpoint identity, task semantics, CSV-header safeguards, artifact contract, badges, and AI provenance. Live Colab execution evidence should be recorded separately on the PR; do not infer runtime success from static CI alone.",
        "CI separates static checks from execution. Release verification executes the notebooks through a real IPython/Jupyter kernel, preserving `%pip` and notebook semantics; the companion receives the producer artifact and a distinct fresh inference CSV through explicit external paths. Hosted-Colab execution may still be recorded as an additional surface check, but static checks alone are never runtime evidence.",
    )
    path.write_text(text, encoding="utf-8")


patch_main(ROOT / "tutorials/tabiclv2_regressor_colab.ipynb")
patch_inference(ROOT / "tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb")
patch_readme(ROOT / "tutorials/README.md")
print("Reviewer-fix notebook migration complete for API revision", API_REVISION)
