from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/tabiclv2_regressor_colab.ipynb")
README = Path("tutorials/README.md")
SOURCE_SAMPLE_REVISION = "040bae8359f279defdbfbced96e6af96dcf8de8e"

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cell = next(
    c
    for c in nb["cells"]
    if c.get("cell_type") == "code"
    and 'DATA_SOURCE = "Sample: Diabetes"' in "".join(c.get("source", []))
)
src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

src = src.replace(
    "import math\n\nimport numpy as np",
    "import math\nimport urllib.request\nimport zipfile\n\nimport numpy as np",
    1,
)
src = src.replace(
    "from sklearn.datasets import load_diabetes",
    "from sklearn.datasets import fetch_california_housing, load_diabetes",
    1,
)
old_selector = (
    'DATA_SOURCE = "Sample: Diabetes"  # @param '
    '["Sample: Diabetes", "Upload CSV", "Upload pre-split train/val/test"]'
)
new_selector = (
    'DATA_SOURCE = "Sample: Diabetes"  # @param '
    '["Sample: Diabetes", "Sample: California Housing", '
    '"Sample: Insurance Charges (medical cost)", "Upload CSV", '
    '"Upload pre-split train/val/test"]'
)
if old_selector not in src:
    raise SystemExit("regressor DATA_SOURCE selector anchor not found")
src = src.replace(old_selector, new_selector, 1)

start_marker = 'test_data = None\nif DATA_SOURCE == "Sample: Diabetes":\n'
end_marker = 'elif DATA_SOURCE == "Upload CSV":\n'
start = src.find(start_marker)
end = src.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("regressor sample branch anchors not found")

sample_block = f'''test_data = None
if DATA_SOURCE == "Sample: Diabetes":
    dataset = load_diabetes(as_frame=True)
    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})
    train_data, remainder = train_test_split(frame, test_size=0.4, random_state=RANDOM_SEED)
    holdout_data, test_data = train_test_split(
        remainder, test_size=0.5, random_state=RANDOM_SEED
    )
    print("✓ Using Diabetes sample (continuous sanity case).")
elif DATA_SOURCE == "Sample: California Housing":
    TARGET_COLUMN = "MedHouseVal"
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.sample(n=2000, random_state=RANDOM_SEED).reset_index(drop=True)
    train_data, remainder = train_test_split(frame, test_size=0.4, random_state=RANDOM_SEED)
    holdout_data, test_data = train_test_split(
        remainder, test_size=0.5, random_state=RANDOM_SEED
    )
    print("✓ Using a seeded 2,000-row California Housing sample.")
elif DATA_SOURCE == "Sample: Insurance Charges (medical cost)":
    TARGET_COLUMN = "charges"
    sample_url = (
        "https://raw.githubusercontent.com/kurtvalcorza/tabicl-regressor-pipeline/"
        "{SOURCE_SAMPLE_REVISION}/examples/sample-data/insurance-medical-charges.zip"
    )
    with urllib.request.urlopen(sample_url, timeout=30) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        by_name = {{Path(info.filename).name.lower(): info for info in archive.infolist() if not info.is_dir()}}
        required = {{"train.csv", "val.csv", "test.csv"}}
        missing = required - set(by_name)
        if missing:
            raise RuntimeError(f"Insurance Charges sample ZIP missing: {{sorted(missing)}}")
        train_data = read_csv_payload(archive.read(by_name["train.csv"]), "train.csv")
        holdout_data = read_csv_payload(archive.read(by_name["val.csv"]), "val.csv")
        test_data = read_csv_payload(archive.read(by_name["test.csv"]), "test.csv")
    print("✓ Using Insurance Medical Charges sample (mixed-feature regression).")
'''
src = src[:start] + sample_block + src[end:]
cell["source"] = src

markdown = next(
    c
    for c in nb["cells"]
    if c.get("cell_type") == "markdown"
    and "## 3. Load sample or BYOD data" in "".join(c.get("source", []))
)
md = "".join(markdown["source"]) if isinstance(markdown["source"], list) else markdown["source"]
old_intro = (
    "**Sample: Diabetes** is scikit-learn's diabetes progression set (442 rows, 10 numeric "
    "features already standardized, a continuous score). It is split 60 / 20 / 20 at random "
    "into train / holdout / independent test. It is a small, noisy *sanity* dataset: expect an "
    "R² around 0.5, which is what published models reach on it too."
)
new_intro = (
    "The default **Diabetes** sample remains the small continuous sanity path. The selector also "
    "exposes a seeded 2,000-row **California Housing** sample for broader numeric regression and "
    "the bundled **Insurance Medical Charges** archive for mixed numeric/categorical regression. "
    "These are tutorial/sanity datasets, not benchmark evidence; provenance and licensing are "
    "recorded in `examples/sample-data/DATASET_CARD.md`."
)
if old_intro not in md:
    raise SystemExit("regressor Step 3 markdown anchor not found")
markdown["source"] = md.replace(old_intro, new_intro, 1)
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

text = README.read_text(encoding="utf-8")
if "## Sample portfolio\n" not in text:
    text = text.rstrip() + """

## Sample portfolio

The E2E notebook retains Diabetes as the default continuous sanity case and adds the sample portfolio from PR #15: a seeded California Housing subset for numeric regression and Insurance Medical Charges for mixed numeric/categorical regression. `examples/sample-data/DATASET_CARD.md` records provenance and licensing; sample results remain tutorial evidence rather than benchmark claims.
"""
    README.write_text(text, encoding="utf-8")
