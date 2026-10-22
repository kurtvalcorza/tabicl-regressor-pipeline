# Release verification

`tutorials/tabiclv2_regressor_colab.ipynb` (`E2E`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 3.0. This file is the durable release-gate record.

**Notebook Spec 3.0 fold (2026-10-22).** The notebook now declares Notebook Spec 3.0 (kind `TUTORIAL`) and closes with the
external-artifact section (Section 10, §19.1). The artifact-inference companion
`tutorials/tabiclv2_regressor_artifact_inference_colab.ipynb`, its template, its stage runner, `tests/test_companion_parity.py`,
`tools/build_sample_bundle.py` and the sample bundle `examples/sample-bundle/` were deleted (KND11; the bundle was assembled
by a script that ran no model, so it could not meet SART9). The section's default artifact is the release asset
`tabiclv2_regressor_artifact.zip` of tag `sample-artifact-diabetes-v1` (SHA-256 `3cd5abc8d364398b6ff2b2716b48ca3d45781f6893accc097c93eec768dbe828`): the export of the recorded hosted
T4 run of `6b95e73` (the default diabetes path, pretrained mode), written by that revision's `shutil.make_archive`; its digest
is pinned as published (SART8). From this change on, the export stage writes a deterministic ZIP (sorted members, fixed
1980-01-01 timestamps, mode 0644, atomic rename), so an export's digest depends only on its contents. The section's own hosted
evidence with that pin (the second pass of §32.2) is the `1e1a1ab` record below (*First hosted run of the migrated
notebook*); the notebook stays Candidate for the reasons under *Current status*. Text below that describes the companion, its sample bundle or both notebooks describes
revisions before the fold and is kept as history (REL14); the companion's execution evidence is not evidence for the migrated
notebook.

**Producing run of the sample artifact (hosted Colab T4, 2026-10-22, `6b95e73` / blob `c74dcd4c08fc`).** The default path of
`tutorials/tabiclv2_regressor_colab.ipynb` at `main` `6b95e73` (blob `c74dcd4c08fc`, the pre-fold notebook), run by the Colab CLI
(`colab exec -f`, 0.7.4) on a fresh hosted T4 session, started 2026-10-22T01:18:36Z, wall 156.8 s, exit 0, input unchanged,
11/11 code cells, verdict PASS. Evidence: [`execution-evidence/2026-10-22-6b95e73/`](execution-evidence/2026-10-22-6b95e73/).
Its export `outputs/tabiclv2_regressor_artifact.zip` (printed `zip_sha256` `3cd5abc8d364398b6ff2b2716b48ca3d45781f6893accc097c93eec768dbe828`,
mode `pretrained`) is published unchanged as the release asset of tag `sample-artifact-diabetes-v1` (target `6b95e73`), whose
notes name this run (SART8). It is the pinned default artifact of the external-artifact section (Section 10); it is not evidence
for the migrated notebook, which needs its own hosted run.

**Local-path redaction of the producing-run evidence.** Local paths were replaced by `<local-run-dir>` before the first commit.
The artifact ZIP is not committed here; it is the release asset.

| File (under `execution-evidence/2026-10-22-6b95e73/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_6b95e73_colab-cli-t4.ipynb` | `f5298ad6eebe8437e7e253339f0dfa7760bbcbff794b88797d3faa67978fca59` | unchanged (no local path) |
| `tabiclv2_regressor_colab_6b95e73_colab-cli-t4_exec.log` | `7d7eb14bcfadd0c683366c13ef2186e18cff386d28fad6fa52df8af12be7822b` | `93ddf0cdfed6bd7c2f35690a6d3a1310d976ff3bd37fbcd0ef61231604f0d3dd` |
| `tabiclv2_regressor_colab_6b95e73_colab-cli-t4_run_summary.json` | `c404d2f936c432d6ed6469dfb297c39d06414e77f713ef72bd85fdf7d1cf4f54` | unchanged (no local path) |
| `tabiclv2_regressor_colab_6b95e73_colab-cli-t4_suite.log` | `0ac330d193b0b964094aad21dc8954ea1a8aa7f562ebcd326bdba4f4f4db6c7f` | `277e3ac78da430cde69c13dc29b9c820665e549cf22fc7d4fd99155b803d0e79` |

**First hosted run of the migrated notebook (hosted Colab T4, 2026-10-22, `1e1a1ab` / blob `8cb91a8e`, Notebook Spec 3.0).**
The default path of the folded E2E notebook on PR #29, run by the Colab CLI (`colab exec -f`) on a fresh hosted T4 session,
started 2026-10-22T02:09:26Z, wall 157.4 s, exit 0, input unchanged; executed cell sources equal blob `8cb91a8e` (44 cells),
verdict PASS. Evidence: [`execution-evidence/2026-10-22-1e1a1ab/`](execution-evidence/2026-10-22-1e1a1ab/). This is the second
pass of §32.2. **Section 10 ran end to end on the default path:** it downloaded the release asset of tag
`sample-artifact-diabetes-v1`, verified SHA-256 `3cd5abc8d364398b6ff2b2716b48ca3d45781f6893accc097c93eec768dbe828` before
extraction, rebuilt the regressor from the files alone (the `external_reconstruct` stage completed on the hosted runtime) and
wrote `outputs/external_artifact/tabiclv2_regressor_external_predictions.csv` and `..._external_result.json` (printed in the
executed notebook; the files stayed on the released VM). The run's own deterministic export printed `zip_sha256`
`fc2e628a776732cecaf91c8f670d566bd83df661367e6376dee0104b412ed059`.

| File (under `execution-evidence/2026-10-22-1e1a1ab/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_1e1a1ab_colab-cli-t4.ipynb` | `c1902e839b04591783fb85e9b42a4fefd5972936037a69c7c85392d710b72188` | unchanged (no local path) |
| `tabiclv2_regressor_colab_1e1a1ab_colab-cli-t4_run_summary.json` | `82935a004266d33a7daf056cefd1d84bdc219baa6d479f8d53ec97c34f8a76ba` | unchanged (no local path) |
| `tabiclv2_regressor_colab_1e1a1ab_colab-cli-t4_exec.log` | `6ff20606618f0e9fb5855baefcc9e7da0de1b3ce147fd49d09ff87d722d60d42` | `4992c2a9dd9b413db67d55da3ebb9142c6b74a41195fdebc12f22bba548b228c` (2 replacements) |
| `tabiclv2_regressor_colab_1e1a1ab_colab-cli-t4_suite.log` | `b32d73b34aec174b6a6c8d74305005ab23654328d8e353b8c912c5bfc84cdcc6` | `a276bc9b7b44abbcbd38b5293814305a0af70b68d9fadd41f6ff297e162464f9` (4 replacements) |

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks the notebook (the companion-specific items below applied before the Notebook Spec 3.0 fold; the same checks now bind the external-artifact section, plus the exact pinned URL and digest, the digest check before extraction, the SART10 contract check, the `outputs/external_artifact/` exports and the deterministic export):

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, mode `GUIDED`, `standalone: true` and
  `generated_from` (repository, generating commit, package module paths and SHA-256, carried-file digests, generator
  `build_notebook.py/3.0-tabular`);
- the standalone carrier and isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): no clone, repository install
  or repository import on the primary path and nothing installed into the notebook kernel; one carrier cell
  (`metadata.dimer.embedded_sources`) whose `CARRIED_FILES` / `CARRIED_BINARY` equal the repository files they come from
  (`src/tabicl_regressor_pipeline/{__init__,api}.py`, the stage runner `tools/tutorial_stages.py` or
  `tools/tutorial_stages_artifact_inference.py`, `tutorials/requirements-colab.lock.txt`, the committed
  `weights/tabicl-regressor-v2/dimer-base-manifest.json`, the licence, and the companion's pinned sample bundle from
  `examples/sample-bundle/`) with matching `CARRIED_HASHES`; the lock pins every `pyproject.toml` runtime pin with hashes;
  the install cell builds a managed-CPython virtual environment with a pinned `uv` (size + SHA-256 checked) and
  `--require-hashes`, keyed on the lock digest, drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP` and sets `MPLBACKEND=Agg`
  for the stages; the four Infrastructure cells are titled and collapsed; every learner cell runs a stage (`run_stage`);
  no "restart the runtime" instruction anywhere; the notebook byte-identical to `tools/build_notebook.py` output;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried package (and repeated in the carried manifest, which the
  `weights` stage checks against the package before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific calls in the stage runners — E2E: `stage_missing_files`, `verify_snapshot`,
  `TabICLRegressionPipeline.from_pretrained(weights_dir=..., n_estimators=..., random_state=42)`, `validate_inputs`
  (with the too-few-rows rejection probe), the BYOD checks (missing target column named with the header, present but
  unparseable targets refused naming the values — thousands separators diagnosed —, more than 20 % blank targets
  refused, numeric columns with stray strings refused unless listed in `TEXT_COLUMNS`, the blank-target count carried
  into the input manifest), `prepare_regression_table`, `fit_categorical_encoder`, `training_mean_baseline`, `fit`, the
  bounded `create_finetuned_regressor` / `fine_tune_regressor` run (3 epochs, 300 s, patience 2) with holdout-only selection
  (`compare_metric`, `MIN_SELECTION_HOLDOUT_ROWS`) and a recorded skip on CPU, a standardised linear regression, LightGBM and a random forest on the
  same partitions with paired standard errors of the MSE differences, `evaluation_report` with the independent test,
  `read_inference_csv` + inference-mode `validate_inputs`, the bundle export with digests, the payload allowlist and the
  printed ZIP SHA-256, `safe_extract_zip` + `verify_artifact_bundle` on the fresh reload and the `rtol=1e-5, atol=1e-7`
  equivalence check; companion: the trusted-digest check (`EXPECTED_ZIP_SHA256`) before `safe_extract_zip`,
  `validate_artifact_runtime`, the base-model identity check including the pretrained checkpoint binding
  (`digests.checkpointSha256 == BASE_MODEL_SHA256`), `verify_artifact_bundle`, reconstruction on the bundled checkpoint with
  `allow_auto_download=False`, inference-mode `validate_inputs` with a rejection probe, the numeric-feature check,
  `apply_categorical_encoder`, `predict`, a `not-measurable` `evaluation_report` — the ceiling prints, the four exports per
  notebook, the learner-facing statements, the form-parameter defaults (`USE_BYOD`, the BYOD path fields,
  `RUN_NEW_DATA_INFERENCE`, `RUN_ACTIVITY` off; the companion's `ARTIFACT_ZIP_PATH`, `UPLOAD_ARTIFACT`,
  `EXPECTED_ZIP_SHA256`, `NEW_DATA_PATH`, `UPLOAD_NEW_DATA` empty/off), and no quality `assert` in the stage runners;
  forbidden patterns (credential-in-URL, own-repository clone/install, a `git+https://` dependency without a 40-hex SHA, a
  mutable `revision='main'`, direct `from tabicl import` / `TabICLRegressor(` / `FinetunedTabICLRegressor(` /
  `hf_hub_download(` / `sklearn` use **in the notebook's own cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`,
  `extractall(`; in the companion's runner also `load_diabetes(`, `fetch_california_housing(`,
  `training_mean_baseline(`, `shutil.make_archive(`, any fine-tuning call);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the `## Checkpoint Provenance` section.

CI also runs `ruff`, `tools/build_notebook.py --check`, `scripts/validate_colab_tutorial.py` (the
repository's own checks: exact build backend, release lock files and their agreement with the pins, the carried hash lock,
the fine-tune gate and its bounds, hardening markers), the two `scripts/test_*.py` suites and the offline
unit suite (`tests/`: snapshot, role and weight-fact helpers, `test_notebook_parity.py`, `test_external_artifact_section.py`, and
`test_notebook_review_fixes.py`, which execs the notebooks' own cells with stand-ins and runs the model-free stages; no
weights, no model). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab T4 GPU runtime (CPU runs everything except fine-tuning); any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorials are written for; a clean one-pass **Run all** here is promotion evidence |
| Kaggle CLI kernel | Kaggle GPU kernel, any image Python | Reproducible clean-room executor of the same class; the notebook is pushed verbatim (no repository checkout is needed — the notebooks are standalone, and neither notebook's default path opens an upload dialog). For the companion's executor path, set `ARTIFACT_ZIP_PATH` to the E2E run's `outputs/tabiclv2_regressor_artifact.zip`, `EXPECTED_ZIP_SHA256` to the digest that run printed, and `NEW_DATA_PATH` to its `outputs/tabiclv2_regressor_new_rows.csv` |
| GitHub Actions `release-notebook-execution` (previous pair) | ubuntu, Python 3.13, `nbclient` | Executed the previous repository-installing pair through real kernels (`scripts/execute_notebook_release_paths.py`); it must be re-pointed at the standalone pair before it counts again |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a new **T4 GPU** runtime with **no repository checkout** and a clean model cache;
3. run it with one **Run all** and no restart, without editing implementation cells (form parameters at their defaults:
   `DATA_SOURCE = 'Sample: Diabetes'`, `USE_BYOD = False`, `RUN_FINE_TUNING = True`, `FINE_TUNE_EPOCHS = 3`,
   `EVAL_METRIC = 'mae'`, `RUN_NEW_DATA_INFERENCE = False`, `RUN_ACTIVITY = False`), then re-run the export cell
   once to confirm a second pass reuses the environment and reproduces the bundle digest;
4. verify that Section 2 reports the carried-file verification and the isolated environment's versions (CPython 3.12.12,
   `torch` 2.11.0, `tabicl` 2.1.1) with CUDA visible, and that the exported result records `NOTEBOOK_SOURCE` equal to
   `metadata.dimer.generated_from`;
5. verify every default-path stage completes:
   - pinned `jingang/TabICL` acquisition at the immutable revision: the carried manifest is checked against the package
     identity, `stage_missing_files` reports the one manifest entry (`tabicl-regressor-v2-20260212.ckpt`) on a clean
     runtime, and `verify_snapshot` returns its digest;
   - the diabetes sample split 60/20/20 at random (`train=265, holdout=88, test=89`); the ceilings surfaced; `outputs/tabiclv2_regressor_input_manifest.json` with one recorded rejection finding;
   - in-context conditioning and the pretrained holdout/test metrics; the **fine-tuning stage ran** (epochs, the
     candidate's holdout metrics, the selection basis, `best.ckpt` kept only); on the holdout the linear regression at MAE 38.22 /
     R² 0.581, the random forest 42.28 / 0.495 and LightGBM 44.94 / 0.426; the paired standard errors;
   - `outputs/tabiclv2_regressor_evaluation_report.json` with verdict `sample-sanity`;
   - eight held-out rows scored into `outputs/tabiclv2_regressor_predictions.csv`, and their unlabelled copy
     `outputs/tabiclv2_regressor_new_rows.csv`;
   - the bundle `outputs/tabiclv2_regressor_artifact.zip` written with its SHA-256 printed, reloaded in a fresh process
     with equivalent predictions;
6. in the same run, confirm the closing external-artifact section (Section 10): the pinned URL and SHA-256 printed before the
   download, `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`, `zip_sha256` `3cd5abc8…`, `mode: pretrained` with the
   checkpoint bound to the base, `data_contract` matching the diabetes default path, 265 context rows, eight diabetes
   holdout rows scored, and `outputs/external_artifact/tabiclv2_regressor_external_predictions.csv` and
   `..._external_result.json` written. A tampered or wrong pin must stop the run before extraction;
7. verify the exports exist and the interpretation sections match the observed paths;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, tabicl, device), model identifier and
   immutable revision, whether the model cache was clean, `restarted: false`, outcome, produced outputs and metrics, and any
   warning or applicable `SHOULD` deviation in the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with `git rev-parse <commit>:tutorials/<notebook>`). Wall
times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `78312ed` / `9bbb5a682aad` | Kaggle T4 (`kurtvalcorza/dimer-nb2-tabiclv2-regressor` v1) | Default sample path | 206.2 s | **PASSED** — 9/9 ok code cells executed cleanly, 5 files, 114 MB staged (previous E2E blob, superseded; outcome only — no metrics, package versions or `restarted` flag were recorded) |
| 2026-10-10 | `fe7596f` / `c74dcd4c08fc` of `tabiclv2_regressor_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabiclv2_regressor_colab_fe7596f_colab-cli-t4.ipynb`, sha256 `096c878fdbfbb5bbe00ce4ce82772287e0df820075aea82d83a6e28a9eb5ea73` (`…_run_summary.json` `06478e033412…`, `…_exec.log` `4cf8c26a63f5…` (as first committed; redacted 2026-10-22, now `76eba841dd1f…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (15,360 MiB, reported by cell 1), via the workspace `colab-cli-serial-test-suite`; isolated uv CPython 3.12.12 environment from the carried hash lock (85 packages; torch 2.11.0, tabicl 2.1.1, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.0, lightgbm 4.7.0). Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/11–11/11) | E2E default path incl. the bounded CUDA fine-tune: Diabetes sample 265 / 88 / 89, device cuda; checkpoint fetched from the Hub at revision `4dcd344ece2c` and digest-verified (`0db9cb53…`) | 145.5 s session wall | **PASSED — one pass, no restart, 0 errors, 11/11 code cells.** Too-few-rows probe rejected; holdout: TabICL pretrained MAE 38.83, RMSE 49.07, R² 0.582 (equal to the CPU lock-only drive); standardised linear regression 38.22 / 0.581, random forest 42.28 / 0.495, LightGBM 44.94 / 0.426, training mean 67.5 (equal to the notebook's worked answers); independent test TabICL 44.78 / 0.486, linear 46.72 / 0.439, LightGBM 47.77 / 0.369. Fine-tune requested (3 epochs / 300 s) and ran on cuda, stopping early after about 3.5 s; the pretrained model won on holdout MAE (`recommended_for_export` `pretrained`), so the export carries the 114,324,594-byte base checkpoint and the 342,875,297-byte fine-tuned checkpoint was neither exported nor reloaded. Fresh-process reload PASSED, max abs prediction difference 0.0. The fine-tuner prints its validation MAE negated (`Baseline val mae: -38.6600`). Optional activity not run. Carried provenance names `453066a` (generation-time HEAD, not `fe7596f`). Status stays **Candidate** |
| 2026-10-10 | `fe7596f` / `a295fc45b289` of `tabiclv2_regressor_artifact_inference_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabiclv2_regressor_artifact_inference_colab_fe7596f_colab-cli-t4.ipynb`, sha256 `a94c1fed608b7ddd77f0bf976e98a8d2f90ca829b36f8926c5c979ff1b79bb5f` (`…_run_summary.json` `a98e27e315ef…`, `…_exec.log` `83a165c2d68e…` (as first committed; redacted 2026-10-22, now `87f1cb6b701b…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`; same 85-package isolated uv environment. Not a browser Run all; order from the CLI log (1/9–9/9) | Companion default path: the carried, offline-built sample (`artifact.json` + 265-row context parquet + `new_rows.csv`, no release, no upload dialog), checkpoint fetched from the Hub and bound by digest | 107.7 s session wall | **PASSED — one pass, no restart, 0 errors, 9/9 code cells.** No upload dialog; `artifact.json` verified against the pinned sample record; checkpoint `0db9cb538f11…` bound (identical to the pinned base, SHA-256 checked); reconstructed on cuda from 265 context rows; missing-column probe rejected (`Inference CSV missing features: ['age']`); the 8 predictions match the E2E run's within 1e-4 — 5/8 exact, rows 285 / 349 / 331 differ by 0.0001 — consistent with the notebook's stated device floating-point contract. Optional activity not run. Carried provenance `453066a`. Status stays **Candidate** |

### Local lock-only drives (not clean-runtime evidence)

These ran the notebooks' own code cells in order (nbclient, a separate ipykernel venv) in a Linux x86_64 WSL2 container,
CPU only (`CUDA_VISIBLE_DEVICES=-1`), with every stage in the environment the notebook itself built from
`tutorials/requirements-colab.lock.txt` (85 packages, `torch` 2.11.0+cu130, `tabicl` 2.1.1). Two local deviations: the
pinned checkpoint was pre-seeded into `weights/tabicl-regressor-v2/` after a SHA-256 check (this container's IPv6 route to
the Hub stalls; the stage then reported `fetched: []` and re-verified it), and `LD_LIBRARY_PATH` pointed at the locked
`torch/lib` because the container lacks the system `libgomp1` that LightGBM loads (Colab and Kaggle images ship it).
They are not hosted runs and do not count toward promotion.

| Date (UTC) | Commit / notebook blob | Path exercised | Wall | Outcome |
|---|---|---|---|---|
| 2026-10-11 | `2dfd602` / `c74dcd4c08fc` (E2E) | Default sample path; fine-tuning requested and skipped (no CUDA, recorded) | 262 s (environment reused) | 11/11 code cells, no error. Pretrained TabICLv2 holdout MAE 38.83 / RMSE 49.07 / R² 0.582, test MAE 44.78 / R² 0.486; linear 38.22 / 0.581 (test 46.72 / 0.439), random forest 42.28 / 0.495, LightGBM 44.94 / 0.426 (test 47.77 / 0.369), training mean 67.52; TabICLv2 vs linear holdout MSE delta −7.6, paired SE 92.6; blend weight 1.0. Bundle exported (`mode: pretrained`, `matches_pinned_sample_context: True`), fresh-process reload max abs difference 3.1e-05, PASS |
| 2026-10-11 | `2dfd602` / `a295fc45b289` (companion) | Defaults: carried sample bundle and rows, no upload | 17 s | 9/9 code cells, no error. `trusted_digest` verified, `mode: pretrained`, checkpoint bound to the base digest, 265 context rows; the eight predictions equal the E2E run's to four decimals (182.7016 … 136.1901); report `not-measurable`, `sample_kind: sample` |
| 2026-10-11 | `2dfd602` / `a295fc45b289` (companion) | The E2E run's ZIP by `ARTIFACT_ZIP_PATH` with its printed `EXPECTED_ZIP_SHA256`, its `..._new_rows.csv` by `NEW_DATA_PATH` | 20 s | 9/9 code cells, no error; `trusted_digest: verified (EXPECTED_ZIP_SHA256)`, checkpoint binding holds |
| 2026-10-11 | `2dfd602` | The default-path fine-tuning call of `stage_condition` with its exact arguments except `device="cpu"`, `epochs=1` (GPU path not exercised) | 176 s | The locked `tabicl` accepts every argument; `fit(..., X_val, y_val, output_dir)` wrote `best.ckpt` (342,865,633 bytes) and `epoch1.ckpt`; a regressor rebuilt from `best.ckpt` predicts (holdout MAE 38.99) |

### Local-path redaction of earlier evidence (2026-10-22)

The local run directory in these earlier committed evidence files was replaced by `<local-run-dir>/`, the
convention the later evidence in this repository already uses. Only those bytes changed: line endings and every
other byte are as committed. Git history is not rewritten, so earlier commits still hold the original files.
SHA-256 before (committed on `main`) → after:

| File | Replacements | SHA-256 before | SHA-256 after |
| :--- | ---: | :--- | :--- |
| `docs/execution-evidence/2026-10-10/tabiclv2_regressor_artifact_inference_colab_fe7596f_colab-cli-t4_exec.log` | 2 | `83a165c2d68e9a51e1630334c1552fc0cc3f1f23f569e40a5f5f02ee9afef12b` | `87f1cb6b701bb8eb8426eacafad6eed155f157b6e12e2343c573ddb0f5d95371` |
| `docs/execution-evidence/2026-10-10/tabiclv2_regressor_colab_fe7596f_colab-cli-t4_exec.log` | 2 | `4cf8c26a63f5483eec1e269564d319003b77c2c0386dfefed22c847f3ce2d225` | `76eba841dd1f405b356ace1a8ca0314824d23ed18d0401dfa864d2b6f61eef25` |

Where this document quotes one of these files' earlier hash as its committed hash, the new hash is now
given beside it.


## SART7 deviation (2026-10-22)

- **Tag:** `sample-artifact-diabetes-v1` (asset `tabiclv2_regressor_artifact.zip`).
- **Mutable release:** the GitHub release is not immutable (`gh api repos/kurtvalcorza/tabicl-regressor-pipeline/releases/tags/sample-artifact-diabetes-v1 --jq .immutable` returns `false`), so SART7 (SHOULD) is not met.
- **Pinned digest:** the notebook pins the asset by `SAMPLE_ARTIFACT_SHA256 = "3cd5abc8d364398b6ff2b2716b48ca3d45781f6893accc097c93eec768dbe828"` (`tools/tutorial_stages.py`, carried into the notebook); the GitHub asset digest is the same value.
- **Integrity:** integrity rests on that pinned SHA-256 check. `fetch_pinned_artifact` hashes the downloaded file with `check_trusted_digest` and refuses it before extraction if the digest differs, so a replaced asset is refused, not used.
- **Owner acceptance:** the owner accepted this deviation on 2026-10-22. The release is not re-tagged and the pin is not changed.

## Current status

**Current status (2026-10-22): Candidate.** The current E2E notebook blob is `9595d23b` (`tutorials/tabiclv2_regressor_colab.ipynb`, regenerated from template commit `b7320bc` on `docs/rel14-status`; a text-only change that drops the notebook's own run-status claims). It has no recorded hosted run until the orchestrator records one (REL14). The one recorded hosted run covers the previous blob `8cb91a8e` (`1e1a1ab`, unchanged on `main` at `28d00c5`): the `1e1a1ab` record (*First hosted run of the migrated notebook*, near the top of this file), evidence in `execution-evidence/2026-10-22-1e1a1ab/`. It ran by Colab CLI `colab exec -f` (scripted, not a browser Run all) on a fresh Colab Tesla T4: exit 0, 14/14 code cells, 0 error outputs, 157.4 s, input unchanged, verdict PASS. In that run the bounded CUDA fine-tune ran and the pretrained model was recommended for export (selection on holdout MAE), so the fine-tuned-checkpoint export path was again not exercised; the export reloaded in a fresh process (PASS, max abs prediction difference 1.5e-05); and Section 10 verified the pinned `sample-artifact-diabetes-v1` asset (`3cd5abc8…`) before extraction. The notebook stays **Candidate** until a hosted default-path run of the current blob and these are recorded for it:

- **REL12:** a BYOD run in which one representative input is accepted through the downstream stages and one incompatible input is refused;
- **REL2:** a browser run of the interactive journeys (BYOD upload dialog, optional-activity toggle, own-artifact ZIP); the scripted CLI run counts for the default path only;
- **Optional activity:** a run of the Section 11 ensemble-size activity (`RUN_ACTIVITY`); the `1e1a1ab` run printed "Optional activity skipped". The owner ruled on 2026-10-22 that it must be run before promotion.

Promotion is an integrator decision; the builder does not change the status.

**History, 2026-10-10 (blobs `c74dcd4c08fc` and `a295fc45b289`, both superseded; not evidence for the current blob, REL14) — one-pass hosted runs of both then-current blobs were recorded** (the 2026-10-10 rows of the clean-runtime table): `tabiclv2_regressor_colab.ipynb` (`c74dcd4c08fc`) and `tabiclv2_regressor_artifact_inference_colab.ipynb` (`a295fc45b289`) at `fe7596f`, each by Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (not a browser Run all), no restart, 0 error outputs. Notes for the reviewer: (a) the CUDA fine-tune ran (3 epochs / 300 s requested), stopped early in about 3.5 s, and the pretrained model won on holdout MAE — the export and reload path of the 342,875,297-byte fine-tuned checkpoint was **not** exercised on GPU (follow-up); (b) the companion's predictions match the E2E run's within 1e-4 (5/8 exact; rows 285, 349 and 331 differ by 0.0001), consistent with the stated device floating-point contract; (c) the fine-tuner prints its validation MAE negated (`Baseline val mae: -38.6600`) — cosmetic, follow-up; (d) both carried revision labels read `453066a` (generation-time HEAD, not `fe7596f`) — follow-up; (e) the optional activity was not run; (f) the maintainer accepted the carried, offline-built sample (no release; BSD-3, no weights; CPU cross-check recorded) and keeping the bounded CUDA fine-tune on by default. Both notebooks stayed **Candidate**; the gates then remaining were reviewer confirmation of these runs, the fine-tuned-export path, the BYOD journeys and the optional activities.

Before 2026-10-10: no clean-runtime execution of the current standalone notebook blobs had been recorded (the 2026-09-14 Kaggle row is
the previous E2E blob). Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is necessary but not
sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook blobs
under review and an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real `hf_hub_download` fetch
into a fresh `weights/tabicl-regressor-v2/` has not been executed); `from_pretrained` constructs the estimator but `tabicl`
deserialises the checkpoint only on the first `fit`; the previous CI kernel executions covered the old repository-installing
notebooks, not this carrier; and the standalone carrier itself — executing the carried module cell in a runtime that has no
repository checkout — has been validated statically only (parity PASS, carrier probe with the package import blocked),
never run. The clean runs will be the first execution of the standalone path, of the staging path, of the extracted helpers
(`validate_inputs`, `prepare_regression_table`, `regression_metrics`, `training_mean_baseline`, `evaluation_report`,
`safe_extract_zip`, `verify_artifact_bundle`) and of `TabICLRegressionPipeline` against the real weights.