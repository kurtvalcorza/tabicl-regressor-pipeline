from __future__ import annotations

import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tutorials/tabiclv2_regressor_colab.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = cell.get("source", "")
    text = "".join(src) if isinstance(src, list) else str(src)
    if "RUN_FINE_TUNING = False" not in text:
        continue
    text = text.replace("Finetunedcreate_regressor(", "create_finetuned_regressor(")
    text = text.replace(
        "        raise RuntimeError(\"TabICLv2 fine-tuning requires CUDA\")\n        ft_dir = Path(\"/content/tabiclv2-regressor-finetune\")",
        "        raise RuntimeError(\"TabICLv2 fine-tuning requires CUDA\")\n    ft_dir = Path(\"/content/tabiclv2-regressor-finetune\")",
    )
    if "Finetunedcreate_regressor" in text:
        raise RuntimeError("mangled fine-tuner constructor remains")
    if "finetuner = create_finetuned_regressor(" not in text:
        raise RuntimeError("repository fine-tuner constructor missing")
    cell["source"] = text
    break
else:
    raise RuntimeError("fine-tuning cell not found")

path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Optional fine-tuning branch repaired")
