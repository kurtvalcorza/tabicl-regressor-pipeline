#!/usr/bin/env python3
"""Build and package sample datasets for TabICLv2 Regressor tutorials."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "examples" / "sample-data"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_INSURANCE_SHA256 = "7e16a06e13a7cb58fabb27ee53c1e20af83575c12ac97ea874fea45f013690a2"


def build_insurance_charges() -> Path:
    """Build Insurance Medical Charges (continuous regression, 6 mixed features)."""
    out_zip = SAMPLE_DIR / "insurance-medical-charges.zip"

    # Check local sibling repo first, asserting expected content hash
    sibling_zip = ROOT.parent / "mitra-regressor-pipeline" / "examples" / "sample-data" / "insurance-medical-charges.zip"
    if sibling_zip.exists():
        sibling_bytes = sibling_zip.read_bytes()
        sibling_digest = hashlib.sha256(sibling_bytes).hexdigest()
        if sibling_digest == EXPECTED_INSURANCE_SHA256:
            print(f"Verified digest on sibling archive {sibling_zip.name}. Copying...")
            out_zip.write_bytes(sibling_bytes)
            print(f"[OK] Wrote {out_zip.name} ({out_zip.stat().st_size:,} bytes)")
            return out_zip
        print(f"Sibling archive hash mismatch ({sibling_digest} != {EXPECTED_INSURANCE_SHA256}); rebuilding from source...")

    print("Fetching Insurance Medical Charges from public mirror...")
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    with urllib.request.urlopen(url) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))

    # Split 1,338 rows: 802 train (60%), 268 val (20%), 268 test (20%)
    train_df, temp_df = train_test_split(df, train_size=802, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, train_size=268, random_state=SEED)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, part in [("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)]:
            info = zipfile.ZipInfo(name, (2026, 9, 7, 23, 2, 46))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, part.to_csv(index=False))

    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == EXPECTED_INSURANCE_SHA256, f"Expected {EXPECTED_INSURANCE_SHA256}, got {digest}"
    out_zip.write_bytes(payload)

    print(f"[OK] Wrote {out_zip.name} ({out_zip.stat().st_size:,} bytes, SHA-256: {digest})")
    return out_zip


def main() -> None:
    build_insurance_charges()
    print("All TabICL Regressor sample datasets built successfully.")


if __name__ == "__main__":
    main()
