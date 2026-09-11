# TabICLv2 Regressor Sample Datasets

This directory provides and documents sample datasets for the [TabICLv2 Regressor Colab Tutorial](../../tutorials/tabiclv2_regressor_colab.ipynb).

TabICLv2 is an in-context learning tabular foundation model that ingests training rows as in-context conditioning context (`training_context.parquet`). The tutorial supports three complementary sample datasets:

| Dataset | Modality / Task | Rows (Train / Val / Test) | Features | Source & License |
|---|---|---|---|---|
| **Diabetes Progression** | Continuous sanity check (disease progression score) | 265 / 88 / 89 (442 total) | 10 numeric | scikit-learn / Efron et al. (Public Domain Dedication) |
| **California Housing** (Sampled) | Canonical spatial regression benchmark | 1,200 / 400 / 400 (2,000 sampled) | 8 numeric | scikit-learn / Pace & Barry (Public Domain Dedication) |
| **Insurance Medical Charges** | Healthcare expense regression | 802 / 268 / 268 (1,338 total) | 6 mixed (3 categorical strings, 3 numeric) | Brett Lantz / OpenML (CC0 1.0 Universal) |

---

## 1. Diabetes Progression

- **Purpose:** Quick sanity benchmark. Small, noisy continuous progression score ($R^2 \approx 0.45\text{--}0.50$), ideal for verifying GPU execution, in-context conditioning, and export mechanics.
- **Source:** Built into `sklearn.datasets.load_diabetes` (zero network downloads).
- **Target:** `target` — quantitative measure of diabetes disease progression one year after baseline.
- **Features:** 10 numeric attributes centered and standardized: age, sex, body mass index, average blood pressure, and six blood serum measurements (s1–s6).
- **Split:** 60% train / 20% holdout / 20% test (random split, seed 42).

---

## 2. California Housing (Sampled)

- **Purpose:** Demonstrates TabICLv2 on a canonical, non-linear spatial regression problem with complex latitude/longitude and socio-economic interactions. Pretrained TabICLv2 typically achieves strong holdout baselines on this sample (typical $R^2 \approx 0.78\text{--}0.81$ depending on context conditioning).
- **Source:** Subsampled from `sklearn.datasets.fetch_california_housing` to 2,000 rows (seed 42) for ultra-fast in-context inference in Colab (~2 seconds).
- **Target:** `MedHouseVal` — median house value for California districts in hundreds of thousands of dollars ($100,000).
- **Features:** 8 continuous numeric attributes: `MedInc` (median income), `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.
- **Split:** 60% train (1,200 rows) / 20% holdout (400 rows) / 20% test (400 rows) (random split, seed 42).

---

## 3. Insurance Medical Charges

- **Purpose:** Demonstrates TabICLv2 on mixed tabular data with string/categorical columns, verifying the `fit_encoder()` and `apply_encoder()` ordinal mapping pipeline on continuous regression ($R^2 \approx 0.85$). Enables direct cross-model benchmarking with the Mitra Regressor pipeline.
- **Archive:** `insurance-medical-charges.zip` (17 KB). Contains `train.csv`, `val.csv`, and `test.csv` (SHA-256: `7e16a06e13a7cb58fabb27ee53c1e20af83575c12ac97ea874fea45f013690a2`).
- **Target:** `charges` — individual medical expenses billed by health insurance in dollars (mean ~$13,298, range $1,122–$62,593).
- **Features:** 6 mixed features:
  - Categoricals (strings): `sex` (`female`, `male`), `smoker` (`yes`, `no`), `region` (`southwest`, `southeast`, `northwest`, `northeast`).
  - Numerics: `age`, `bmi`, `children`.
- **Split:** 60% train (802 rows) / 20% val (268 rows) / 20% test (268 rows) (random split, seed 42).
- **Provenance & License:** Brett Lantz (2013), *Machine Learning with R*. Source bytes are pinned to `stedy/Machine-Learning-with-R-datasets@d20658ec6d336af2d4ddb5fd72b6f677dd46136e` and verified by SHA-256 before packaging. Distributed under **CC0 1.0 Universal (Public Domain Dedication)**.

### Immutable tutorial asset

The notebook retrieves `insurance-medical-charges.zip` from immutable repository revision `ec4d9e88846aac5dc478975c698703d38737575a` and verifies the archive SHA-256 above before extraction. That revision is retained by the durable branch `anchors/sample-data-20260911`, so deleting the feature branch after merge does not break the published tutorial asset.

---

## How Bundled Datasets Were Built

The bundled archive was generated deterministically using [`../build_sample_datasets.py`](../build_sample_datasets.py):

```bash
python examples/build_sample_datasets.py
```
