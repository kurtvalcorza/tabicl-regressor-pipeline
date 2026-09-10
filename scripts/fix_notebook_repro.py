from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    (ROOT / "tutorials/tabiclv2_regressor_colab.ipynb", "e2e"),
    (ROOT / "tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb", "artifact"),
]

OLD_INSTALL = '%pip -q install --no-deps "git+https://github.com/kurtvalcorza/tabicl-regressor-pipeline@'
NEW_INSTALL = '%pip -q install --no-deps --no-build-isolation "git+https://github.com/kurtvalcorza/tabicl-regressor-pipeline@'

WORDING = {
    "Google Colab or a clean Jupyter/Python 3.13 runtime with release-verified PyTorch 2.11.0;":
        "Google Colab, or a clean Jupyter/Python 3.13 runtime with release-verified PyTorch 2.11.0 and a writable `/content` workspace;",
}


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


for path, prefix in NOTEBOOKS:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed_install = False
    for index, cell in enumerate(notebook.get("cells", [])):
        if not cell.get("id"):
            cell["id"] = f"{prefix}-{index:03d}"
        source = source_text(cell)
        if OLD_INSTALL in source:
            source = source.replace(OLD_INSTALL, NEW_INSTALL)
            changed_install = True
        for old, new in WORDING.items():
            source = source.replace(old, new)
        cell["source"] = source

    if not changed_install and NEW_INSTALL not in "\n".join(source_text(c) for c in notebook.get("cells", [])):
        raise RuntimeError(f"{path.name}: adapter install command was not found")

    notebook["nbformat_minor"] = max(int(notebook.get("nbformat_minor", 0)), 5)
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {path}")
