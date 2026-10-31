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

**Sample artifact v2 (2026-10-24).** The section's default artifact is now the asset `tabiclv2_regressor_artifact.zip` of tag `sample-artifact-diabetes-v2` (SHA-256 `693cc996249e3b8364745244d0bcfe8e7ab9cec1591e6f98f014ec54cd39dbce`, 106,347,998 bytes; GitHub asset digest equal): the bundle-format v2 re-issue of the v1 asset above (its `reissue` block names `fromReleaseTag` `sample-artifact-diabetes-v1`, `fromZipSha256` `3cd5abc8…`, `convertedCheckpointSha256` `ad4b0ec0…` and code revision `86a767b`). It is not the export of a new hosted run; the producer record stays the `6b95e73` run (SART8). The v1 tag is unchanged and v1 bundles stay readable (deprecated). Every hosted record below that names `3cd5abc8…` ran the v1 pin and is history for the v2-pinned blob (REL14); its hosted default-path and optional-activity runs at `1f69142` are recorded below, both PASS: Section 10 verified the v2 asset `693cc996…` and loaded it with no v1 `DeprecationWarning`. A local CPU check (not clean-runtime evidence; `CUDA_VISIBLE_DEVICES=-1`, torch 2.11.0+cpu, 2026-10-24) ran the section's own stages (`weights`, then `external_artifact` with `source='pinned'`, `external_reconstruct` and `external_predict`, the last three with `torch.load` patched to raise): the stage downloaded the v2 asset fresh, verified `693cc996…`, reported `checkpoint_binding: identical to the pinned safetensors conversion of the base`, rebuilt serving state from the bundle and scored the eight diabetes rows (within 0.0001 of the `e80d976` hosted record of the v1 pin).

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

### Hosted Colab T4 runs of blob `5e7d4c4e` (2026-10-29)

These runs executed the PR head `25a787458745ed45acbc1027d381dace13717940`, notebook `tutorials/tabiclv2_regressor_colab.ipynb` blob `5e7d4c4eabe910d1f2c8d95ed7691afd91356745`, fetched at the 40-character SHA and Git-blob verified, with the Colab CLI (`colab exec -f`) on a fresh Colab Tesla T4 session (not a browser `Run all`). Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true (the VM was released), and the SHA-256 of each executed notebook equals the summary's `executed_sha256`.

**Default path (2026-10-29 17:03:20 UTC start).** Default settings, no overrides; session `suite-tabicl-25a7874-1cb9`; wall 160.4 s (`wall_s`); `code_cells` 14, `executed_cells` 14, 0 errors. Retained evidence: [`docs/execution-evidence/2026-10-29-25a7874/`](execution-evidence/2026-10-29-25a7874/).

Printed by the default run (verbatim lines from the executed outputs):

```text
{'verified_files': 1, 'revision': '4dcd344ece2c00be9e831fdd35bed57b5ad83e19', 'sha256': ['0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a']}
training-mean baseline on the holdout {'mae': 67.5162, 'mse': 5800.5533, 'rmse': 76.1614, 'r2': -0.0062}
pretrained holdout {'mae': 38.8274, 'mse': 2408.1409, 'rmse': 49.0728, 'r2': 0.5823, 'pearsonr': 0.766}
pretrained independent test {'mae': 44.7756, 'mse': 2971.2319, 'rmse': 54.509, 'r2': 0.4863, 'pearsonr': 0.7185}
tabicl_pretrained                mae=38.83 rmse=49.07 r2=0.582
lightgbm                         mae=44.94 rmse=57.55 r2=0.426
random_forest                    mae=42.28 rmse=53.97 r2=0.495
standardised_linear_regression   mae=38.22 rmse=49.15 r2=0.581
PASS: bundle extracted safely, digests verified, regressor rebuilt from the bundle alone; predictions equivalent (rtol=1e-05, atol=1e-07).
```

**DAT20 refusal record (2026-10-29, blob `5e7d4c4e`).** Overrides, set by `--set` on a copy of the blob: `DATA_SOURCE` `'Sample: Diabetes'` → `'Upload CSV'` and `USE_BYOD` `False` → `True` and `BYOD_CSV_PATH` `''` → `'/content/sample_data/california_housing_test.csv'` (the run copy differs from the blob in exactly those 3 source lines). The suite cut the run copy right after the refusing cell `tabiclv2_regressor-14`, dropping 29 cells (9 of them code cells), which reproduces `Run all` stopping at the first error; the cut copy is `run_copy.ipynb`, SHA-256 `0f7802b4605698b142534f79d6f3a28a92fab5d6e82f77cf41bed04e77632a07` (Git blob `317fb216`), equal to the summary's `run_copy_sha256`. The cut copy's earlier code cells completed without an error output; the refusing cell's only error output has `ename` `Refused` and no traceback, and its output holds exactly one `Refused:` line:

```text
Refused: california_housing_test.csv: the target column 'target' is not in the header ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']. Set TARGET_COLUMN to the name of your label column. Fix the file (or the Section 4 fields) and run the Section 4 cell again.
```

This run says nothing about the 29 cut cells, which did not run in it. Session `suite-tabicl-25a7874-1cb9`, wall 94.0 s. Retained evidence: [`docs/execution-evidence/2026-10-29-25a7874-refusal/`](execution-evidence/2026-10-29-25a7874-refusal/).

**Local-path redaction.** Local paths were replaced byte-wise by `<local-run-dir>` (and the Colab CLI install path by `<colab-cli-tool-dir>/`) before the first commit; each executed notebook and run copy was checked against its `run_summary.json` before any edit.

| File (under `docs/execution-evidence/2026-10-29-25a7874/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `tabiclv2_regressor_colab_25a7874_colab-cli-t4_default.ipynb` | `2860be561b8d3beed6a29c3ebb0102abf1724ce3b304a75f1b83adaceb5bcde2` | `2860be561b8d3beed6a29c3ebb0102abf1724ce3b304a75f1b83adaceb5bcde2` (unchanged, no local path) |
| `tabiclv2_regressor_colab_25a7874_colab-cli-t4_default_exec.log` | `d788e4defa2b044f48f4d47ebf7abc1fb2a52ae048b6db1f641f6f802276cfc0` | `3f986c3901c0502c4fd77c70aaef3462ab6e6d8147a5b9b6201b54f473c23cc9` (2 local-path replacements) |
| `tabiclv2_regressor_colab_25a7874_colab-cli-t4_default_run_summary.json` | `91dce40f90e1114026c321f46bf058dfcf4d79972cc00d63ccf3df316306b5ad` | `91dce40f90e1114026c321f46bf058dfcf4d79972cc00d63ccf3df316306b5ad` (unchanged, no local path) |

| File (under `docs/execution-evidence/2026-10-29-25a7874-refusal/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `tabiclv2_regressor_colab_25a7874_colab-cli-t4_refusal.ipynb` | `910ba9f7c485c54333178d158f0a6d21e5c5c62a75bbd729f2b4a15baac23a86` | `910ba9f7c485c54333178d158f0a6d21e5c5c62a75bbd729f2b4a15baac23a86` (unchanged, no local path) |
| `tabiclv2_regressor_colab_25a7874_colab-cli-t4_refusal_exec.log` | `a1a1e5b6f0772dabff2eff8473ef2f6cefbc2c00758e037b7e9e98548bb04d3d` | `c3f10c45ef036e59160b766de3220283af11153ba910381748b8242f1c6b12d3` (2 local-path replacements) |
| `tabiclv2_regressor_colab_25a7874_colab-cli-t4_refusal_run_summary.json` | `172775dfd18705951da165b5e706f25b5d7959964b7ec8f9b28de415777c103f` | `172775dfd18705951da165b5e706f25b5d7959964b7ec8f9b28de415777c103f` (unchanged, no local path) |
| `tabiclv2_regressor_colab_25a7874_colab-cli-t4_refusal_run_copy.ipynb` | `0f7802b4605698b142534f79d6f3a28a92fab5d6e82f77cf41bed04e77632a07` | `0f7802b4605698b142534f79d6f3a28a92fab5d6e82f77cf41bed04e77632a07` (unchanged, no local path) |

Not exercised by these runs: an accepted BYOD input (REL12), a browser `Run all` of the interactive journeys (REL2), and every other open gate named in this file; these runs close none of them.

**Hosted default-path run of blob `a84d74ba` (hosted Colab T4, 2026-10-25, `100088f`; the previous blob).** The default path
of `tutorials/tabiclv2_regressor_colab.ipynb` at `100088f` (blob `a84d74ba`, PR #37, tier C review fixes generated at `299b529`,
executed as committed; the fetched source's Git blob id was re-checked), run by the Colab CLI (`colab exec -f`, a scripted run,
not a browser Run all) on a fresh hosted T4 session, started 2026-10-25T07:36:04Z, wall 157.8 s, exit 0, input unchanged, 14/14
code cells, 0 errors, verdict PASS. Evidence: [`execution-evidence/2026-10-25-100088f/`](execution-evidence/2026-10-25-100088f/).
Printed: `served_sha256` equal to the pinned safetensors base (`ad4b0ec0…`); pretrained holdout MAE 38.8274 / R² 0.5823,
independent test MAE 44.7756 / R² 0.4863 (training-mean baseline holdout MAE 67.5162); the bounded CUDA fine-tune ran
(`requested: True`, `ran: True`, converted `checkpoint_bytes` 114223596); `recommended_for_export` was `pretrained`, so the
fine-tuned-checkpoint export path was again not exercised; the export printed `zip_sha256`
`2d9abac6225ba835b5af77d182ec44c605cff02b103adc3d7c92c9806770b62b` (`checkpoint_bytes` 114223620; the `426cd4d6` run printed
`c79d2555…`); the fresh-process reload printed PASS, `max_abs_prediction_difference` 1.52587890625e-05 (rtol 1e-05, atol 1e-07).
Section 10 verified the v2 asset `693cc996…` (`trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`) with no
`DeprecationWarning`. The activity cell printed "Optional activity skipped". Every metric the notebook's prose quotes equals this
run's output. Local paths were replaced by `<local-run-dir>/` before the first commit; the executed notebook was checked against
`run_summary.json`'s `executed_sha256` before any edit.

| File (under `execution-evidence/2026-10-25-100088f/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_100088f_colab-cli-t4.ipynb` | `783c0a08df2aebe3bc7aa400546960268e9596182f52a0650f99a07ca3df6abc` | unchanged (no local path) |
| `tabiclv2_regressor_colab_100088f_colab-cli-t4_exec.log` | `f9a95f9f01b8522fc514b9b1fde6f587db1c56468c202e8528c08c9d4cff6f3d` | `e4277eb621190bf62c7a13945b6d7bdea9ce2da925f12361eb7508cbe476a127` (2 replacements) |
| `tabiclv2_regressor_colab_100088f_colab-cli-t4_run_summary.json` | `74f5bda9bdcca0c22ec9f71c79f30c115a2e2a0800f695f2e500d647bcbf7099` | unchanged (no local path) |

**Optional-activity run of blob `a84d74ba` (hosted Colab T4, 2026-10-25, `100088f`, `RUN_ACTIVITY = True`; the previous blob).**
A run copy of committed blob `a84d74ba` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `05b84afb93cc`,
SHA-256 `60ac8f09db569bbb6edb1b8123eddc6da0fdf7ad8bf38642d00f374757c6edf6`; it differs from the committed blob only in that
assignment literal, and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by the Colab CLI
(`colab exec -f`, scripted, not a browser Run all) on a fresh hosted T4 session, started 2026-10-25T07:36:58Z, wall 174.1 s,
exit 0, input (the run copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence:
[`execution-evidence/2026-10-25-100088f-activity/`](execution-evidence/2026-10-25-100088f-activity/). The default-path outputs
printed the same metrics, the same `recommended_for_export` (`pretrained`) and the same export digest (`2d9abac6…`) as the
default-path run above; the fresh-process reload printed PASS with `max_abs_prediction_difference` 0.0. Section 10 verified the
v2 asset `693cc996…` with no `DeprecationWarning`. The Section 11 activity (`n_estimators` 8 → 32) printed holdout (88 rows) MAE
38.8274 → 38.7106, `mse_delta_changed_minus_canonical` −7.3516 with `paired_se` 6.6904 (`within_two_se: True`), independent
test (89 rows) MAE 44.7756 → 44.6211, delta −8.0908, `paired_se` 5.4178 (`within_two_se: True`), and
`{'canonical_outputs_unchanged': True}`. Both deltas are within two paired standard errors, so the change is within split noise,
not an improvement. The executed notebook and the run copy were checked against `run_summary.json`'s `executed_sha256` and
`run_copy_sha256` before any edit.

| File (under `execution-evidence/2026-10-25-100088f-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_100088f_colab-cli-t4_activity.ipynb` | `f9a4c8e07745be0861f521847ceae60d3f81f17e280e13e1b4eb32312571e52e` | unchanged (no local path) |
| `tabiclv2_regressor_colab_100088f_colab-cli-t4_activity_exec.log` | `a263e62e247f73fdb2d5efd32bd7035ed664497b17ca50b3fa782c9ac0042618` | `e0ed22c8eba2607d32f6833e39713fdf5876744b3e16d4dd401b63b3dd81ba20` (2 replacements) |
| `tabiclv2_regressor_colab_100088f_colab-cli-t4_activity_run_summary.json` | `71ef98e2933374bd29376be2602e04fa40e0c56c42ac9174d7bc2adf3b881e56` | unchanged (no local path) |
| `tabiclv2_regressor_colab_100088f_colab-cli-t4_activity_run_copy.ipynb` | `60ac8f09db569bbb6edb1b8123eddc6da0fdf7ad8bf38642d00f374757c6edf6` | unchanged (no local path) |

**Hosted default-path run of blob `426cd4d6` (hosted Colab T4, 2026-10-24, `1f69142`; the current blob until `100088f`, now history, REL14).** The default path
of `tutorials/tabiclv2_regressor_colab.ipynb` at `1f69142` (blob `426cd4d6`, PR #36, executed as committed; the fetched source's
Git blob id was re-checked), run by the Colab CLI (`colab exec -f`, a scripted run, not a browser Run all) on a fresh hosted T4
session, started 2026-10-24T16:01:37Z, wall 165.9 s, exit 0, input unchanged, 14/14 code cells, 0 errors, verdict PASS.
Evidence: [`execution-evidence/2026-10-24-1f69142/`](execution-evidence/2026-10-24-1f69142/). Printed: carried package revision
`384864f`; `served_sha256` equal to `pinned` (`ad4b0ec0…`, the safetensors base); pretrained holdout MAE 38.8274 / R² 0.5823,
independent test MAE 44.7756 / R² 0.4863 (training-mean baseline holdout MAE 67.5162); the bounded CUDA fine-tune ran
(`requested: True`, `ran: True`, converted `checkpoint_bytes` 114223596); `recommended_for_export` was `pretrained`
(`holdout:mae`), so the fine-tuned-checkpoint export path was again not exercised; the export printed `zip_sha256`
`c79d255560f1e71ecdaddc06da590d89b36afb0920630d29d7e986f40a8229d8` (`mode: pretrained`, `checkpoint_bytes` 114223620); the
fresh-process reload printed PASS, `max_abs_prediction_difference` 0.0 (rtol 1e-05, atol 1e-07). **Section 10 loaded the v2
default:** it downloaded the asset of release `sample-artifact-diabetes-v2`, reported `trusted_digest: verified (pinned
SAMPLE_ARTIFACT_SHA256)` for `693cc996249e3b8364745244d0bcfe8e7ab9cec1591e6f98f014ec54cd39dbce`, `artifactFormat`
`tabicl-dimer-regressor-v2` with `checkpoint_binding` "identical to the pinned safetensors conversion of the base (SHA-256
checked)", rebuilt the regressor from the bundle and wrote its two `outputs/external_artifact/` files. No `DeprecationWarning`
and no "deprecated v1 pickle layout" line appears in the executed outputs or `exec.log` (the `c49e98e7` run below printed one
for the v1 pin). The activity cell printed "Optional activity skipped". Local paths were replaced by `<local-run-dir>/` before
the first commit; the executed notebook was checked against `run_summary.json`'s `executed_sha256` before any edit.

| File (under `execution-evidence/2026-10-24-1f69142/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_1f69142_colab-cli-t4.ipynb` | `510475b746a45b222732cc590fa61a14bd767c45ce425f63a793af54af27ad91` | unchanged (no local path) |
| `tabiclv2_regressor_colab_1f69142_colab-cli-t4_exec.log` | `c0b9034b12c067b7804ae4c4ae322732b7f1b12c16fa5ff1cf8c6c83ff0f0d8a` | `b3940b11fa3b608f02d94637dee8ce8374d527346e749b28ea6b714e34453746` (2 replacements) |
| `tabiclv2_regressor_colab_1f69142_colab-cli-t4_run_summary.json` | `456a4f4c107a78662f0f2a7a75a2591c8b28c744744972974241fee9e0e95e79` | unchanged (no local path) |

**Optional-activity run of blob `426cd4d6` (hosted Colab T4, 2026-10-24, `1f69142`, `RUN_ACTIVITY = True`; the current blob until `100088f`, now history, REL14).**
A run copy of committed blob `426cd4d6` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `c1d4cdbfd4fe`,
SHA-256 `f70bfdb55877b5ca81e665542ec984c2834855f0092409fff47af94e1d882452`; it differs from the committed blob only in that
assignment literal, and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by the Colab CLI
(`colab exec -f`, scripted, not a browser Run all) on a fresh hosted T4 session, started 2026-10-24T16:02:17Z, wall 192.1 s,
exit 0, input (the run copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence:
[`execution-evidence/2026-10-24-1f69142-activity/`](execution-evidence/2026-10-24-1f69142-activity/). The default-path outputs
printed the same metrics, the same `recommended_for_export` (`pretrained`) and the same export digest (`c79d2555…`) as the
default-path run above; the fresh-process reload printed PASS with `max_abs_prediction_difference` 1.52587890625e-05 (within
rtol 1e-05, atol 1e-07). Section 10 verified the v2 asset `693cc996…` (`tabicl-dimer-regressor-v2`) with no
`DeprecationWarning`. The Section 11 activity (`n_estimators` 8 → 32) printed holdout (88 rows) MAE 38.8274 → 38.7106,
`mse_delta_changed_minus_canonical` −7.3514 with `paired_se` 6.6904 (`within_two_se: True`), independent test (89 rows) MAE
44.7756 → 44.6211, delta −8.0907, `paired_se` 5.4178 (`within_two_se: True`), and `{'canonical_outputs_unchanged': True}`. Both
deltas are within two paired standard errors, so the change is within split noise, not an improvement. The executed notebook
and the run copy were checked against `run_summary.json`'s `executed_sha256` and `run_copy_sha256` before any edit.

| File (under `execution-evidence/2026-10-24-1f69142-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_1f69142_colab-cli-t4_activity.ipynb` | `89793f35f4e1af34b36e1e069450b7a4b380efd62cc017d73ad1b5be3e22a5b7` | unchanged (no local path) |
| `tabiclv2_regressor_colab_1f69142_colab-cli-t4_activity_exec.log` | `75ad7f4751d9dfb635711bd134d16678925ca3386a93dd5f5c6eed332a644306` | `a57327cd15db768877e55df3e997de4bafbb432d08839f5504e053b64b56192b` (2 replacements) |
| `tabiclv2_regressor_colab_1f69142_colab-cli-t4_activity_run_summary.json` | `6c8e38f0fcfe65fd41eba1da3959e8e88c0652536d4d9b801fe0740387101310` | unchanged (no local path) |
| `tabiclv2_regressor_colab_1f69142_colab-cli-t4_activity_run_copy.ipynb` | `f70bfdb55877b5ca81e665542ec984c2834855f0092409fff47af94e1d882452` | unchanged (no local path) |

**Hosted default-path run of blob `c49e98e7` (hosted Colab T4, 2026-10-24, `48885fc`; the current blob until `69f7efb`, now history, REL14).** The default path
of `tutorials/tabiclv2_regressor_colab.ipynb` at `48885fc` (blob `c49e98e7`, PR #35, executed as committed; the fetched source's
Git blob id was re-checked), run by the Colab CLI (`colab exec -f`, a scripted run, not a browser Run all) on a fresh hosted T4
session, started 2026-10-24T11:27:22Z, wall 167.0 s, exit 0, input unchanged, 14/14 code cells, 0 errors, verdict PASS.
Evidence: [`execution-evidence/2026-10-24-48885fc/`](execution-evidence/2026-10-24-48885fc/). Printed: `served_sha256` equal to
`pinned` (`ad4b0ec0…`, the safetensors base); pretrained holdout MAE 38.8274 / R² 0.5823, independent test MAE 44.7756 / R²
0.4863 (training-mean baseline holdout MAE 67.5162); the bounded CUDA fine-tune ran (`requested: True`, `ran: True`) and its
converted checkpoint printed `checkpoint_bytes` 114223596 (the `.ckpt` fine-tune of blob `7baea7e2` printed 342875297);
`recommended_for_export` was `pretrained` (`holdout:mae`), so the fine-tuned-checkpoint export path was again not exercised; the
export printed `zip_sha256` `28692a90373dbae356c6d9dbe03f97739a56d9420df4cee06a5501b028e5a826` (`mode: pretrained`,
`checkpoint_bytes` 114223620); the fresh-process reload printed PASS, `max_abs_prediction_difference` 0.0 (rtol 1e-05, atol
1e-07); Section 10 reported `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)` for `3cd5abc8…`, `artifactFormat`
`tabicl-dimer-regressor-v1` with `checkpoint_binding` "identical to the pinned base checkpoint (deprecated v1 pickle layout)",
and wrote its two `outputs/external_artifact/` files; the activity cell printed "Optional activity skipped". Local paths were
replaced by `<local-run-dir>/` before the first commit; the executed notebook was checked against `run_summary.json`'s
`executed_sha256` before any edit.

| File (under `execution-evidence/2026-10-24-48885fc/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_48885fc_colab-cli-t4.ipynb` | `2e4357578790e79736c5ed4198b3d019d9037d5ea6284f5b01f1ab223901b502` | unchanged (no local path) |
| `tabiclv2_regressor_colab_48885fc_colab-cli-t4_exec.log` | `e8df00bad252789cb2253d28302328db3897ec65ea1747e68c98509782d56926` | `33315664710e9ed4ed5b3834a000a3b761cea4fe502688cdcf0f12665d1a5c70` (2 replacements) |
| `tabiclv2_regressor_colab_48885fc_colab-cli-t4_run_summary.json` | `276cd923a34cc054fff8b44f853ca57f15e4e99d870eda5006615ce4a21b905c` | unchanged (no local path) |

**Optional-activity run of blob `c49e98e7` (hosted Colab T4, 2026-10-24, `48885fc`, `RUN_ACTIVITY = True`; the current blob until `69f7efb`, now history, REL14).**
A run copy of committed blob `c49e98e7` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `903b96afa1fa`,
SHA-256 `7400554669c0dd2d11afab61a7666692ed91d7851cfda9f8ee3350603b9b0a05`; it differs from the committed blob only in that
assignment literal, and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by the Colab CLI
(`colab exec -f`, scripted, not a browser Run all) on a fresh hosted T4 session, started 2026-10-24T11:30:32Z, wall 189.9 s,
exit 0, input (the run copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence:
[`execution-evidence/2026-10-24-48885fc-activity/`](execution-evidence/2026-10-24-48885fc-activity/). The default-path outputs
printed the same metrics, the same `recommended_for_export` (`pretrained`) and the same export digest (`28692a90…`) as the
default-path run above; the fresh-process reload printed PASS with `max_abs_prediction_difference` 0.0. The Section 11 activity
(`n_estimators` 8 → 32) printed holdout (88 rows) MAE 38.8274 → 38.7106, `mse_delta_changed_minus_canonical` −7.3516 with
`paired_se` 6.6904 (`within_two_se: True`), independent test (89 rows) MAE 44.7756 → 44.6211, delta −8.0908, `paired_se` 5.4178
(`within_two_se: True`), and `{'canonical_outputs_unchanged': True}`. Both deltas are within two paired standard errors, so the
change is within split noise, not an improvement. The executed notebook and the run copy were checked against
`run_summary.json`'s `executed_sha256` and `run_copy_sha256` before any edit.

| File (under `execution-evidence/2026-10-24-48885fc-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_48885fc_colab-cli-t4_activity.ipynb` | `b145edb64c62f8d925940c2f6f7a48454f6fe5cf15024c3417310b0f9812dd42` | unchanged (no local path) |
| `tabiclv2_regressor_colab_48885fc_colab-cli-t4_activity_exec.log` | `aee2803322272cf2b8eb5fd3dd427be076715a0d19da769fa1f5e44dbe241660` | `86286cd2f99b7897c4575becf3ea6e13498938da6ceb2af7d1fead51ff07e7a3` (2 replacements) |
| `tabiclv2_regressor_colab_48885fc_colab-cli-t4_activity_run_summary.json` | `6fb16e288b596541253f75abd804d1366a83f437d41411fe1baa7022e318bbff` | unchanged (no local path) |
| `tabiclv2_regressor_colab_48885fc_colab-cli-t4_activity_run_copy.ipynb` | `7400554669c0dd2d11afab61a7666692ed91d7851cfda9f8ee3350603b9b0a05` | unchanged (no local path) |

**Hosted default-path run of blob `7baea7e2` (hosted Colab T4, 2026-10-24, `e80d976`; the current blob until `a43d226`, now history, REL14).** The default path
of `tutorials/tabiclv2_regressor_colab.ipynb` at `e80d976` (blob `7baea7e2`, PR #34, executed as committed; the fetched source's
Git blob id was re-checked), run by the Colab CLI (`colab exec -f`, a scripted run, not a browser Run all) on a fresh hosted T4
session, started 2026-10-24T10:17:02Z, wall 166.8 s, exit 0, input unchanged, 14/14 code cells, 0 errors, verdict PASS.
Evidence: [`execution-evidence/2026-10-24-e80d976/`](execution-evidence/2026-10-24-e80d976/). Printed: `'served_weights':
'tabicl-regressor-v2-20260212.safetensors'` with `served_sha256` equal to `pinned` (`ad4b0ec0…`); pretrained holdout MAE 38.8274 /
R² 0.5823, independent test MAE 44.7756 / R² 0.4863 (training-mean baseline holdout MAE 67.5162), the same values as the `.ckpt`
runs of blob `9610654e`; the bounded CUDA fine-tune ran (`requested: True`, `ran: True`) and `recommended_for_export` was
`pretrained` (`holdout:mae`), so the fine-tuned-checkpoint export path was again not exercised; the export printed `zip_sha256`
`57a190d3acd27c8c209a3b7a8359dc64655ced646d3f1cabb56925d9e9cbc6a9` (`mode: pretrained`); the fresh-process reload printed PASS,
max abs prediction difference 0.0 (rtol 1e-05, atol 1e-07); Section 10 reported `trusted_digest: verified (pinned
SAMPLE_ARTIFACT_SHA256)` for `3cd5abc8…` and wrote its two `outputs/external_artifact/` files; the activity cell printed "Optional
activity skipped". The default path refuses nothing, so this run shows no `Refused:` line. Local paths were replaced by
`<local-run-dir>/` before the first commit; the executed notebook was checked against `run_summary.json`'s `executed_sha256`
before any edit.

| File (under `execution-evidence/2026-10-24-e80d976/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_e80d976_colab-cli-t4.ipynb` | `2cf2514efb9c5297f198ee5a434df914aa46bcba18e563fd7165b3a63b8261d7` | unchanged (no local path) |
| `tabiclv2_regressor_colab_e80d976_colab-cli-t4_exec.log` | `ceb97da39c7a53d3ca45239f75b4124ca3dbbefa8708a4f43b666c39c87ddc38` | `876a354fb617629314791523f9cae3995820a4be305ff5eb5610301fd8f9ea52` (2 replacements) |
| `tabiclv2_regressor_colab_e80d976_colab-cli-t4_run_summary.json` | `8aca9687387b4e2913ecce9416403d7ba673d966a0c5f4ea71054754f417d00d` | unchanged (no local path) |

**Optional-activity run of blob `7baea7e2` (hosted Colab T4, 2026-10-24, `e80d976`, `RUN_ACTIVITY = True`; the current blob until `a43d226`, now history, REL14).**
A run copy of committed blob `7baea7e2` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `cdf28cb5e48b`,
SHA-256 `381d38a3c7969efe3d367eefa6a0e8c1128d8e5397c86047e01410303f82b351`; it differs from the committed blob only in that
assignment literal, and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by the Colab CLI
(`colab exec -f`, scripted, not a browser Run all) on a fresh hosted T4 session, started 2026-10-24T10:20:00Z, wall 190.1 s,
exit 0, input (the run copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence:
[`execution-evidence/2026-10-24-e80d976-activity/`](execution-evidence/2026-10-24-e80d976-activity/). The served weights are the
pinned safetensors file (`ad4b0ec0…`). The default-path outputs printed the same metrics, the same `recommended_for_export`
(`pretrained`) and the same export digest (`57a190d3…`) as the default-path run above; the fresh-process reload printed PASS
with max abs prediction difference 0.0. The Section 11 activity (`n_estimators` 8 → 32) printed holdout (88 rows) MAE 38.8274 →
38.7106, `mse_delta_changed_minus_canonical` −7.3514 with `paired_se` 6.6904 (`within_two_se: True`), independent test (89 rows)
MAE 44.7756 → 44.6211, delta −8.0907, `paired_se` 5.4178 (`within_two_se: True`), and `{'canonical_outputs_unchanged': True}`.
Both deltas are within two paired standard errors, so the change is within split noise, not an improvement. The executed
notebook and the run copy were checked against `run_summary.json`'s `executed_sha256` and `run_copy_sha256` before any edit.

| File (under `execution-evidence/2026-10-24-e80d976-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_e80d976_colab-cli-t4_activity.ipynb` | `54d9549f6cec0b9ff557ad95663e0ae0ca7ad530c4fc00f7b74b5d5dff99093f` | unchanged (no local path) |
| `tabiclv2_regressor_colab_e80d976_colab-cli-t4_activity_exec.log` | `74ddb6968bdf3943afa5873b4ff678c585e8b9caf431154c8f716f15a959bffc` | `c8f68a686cc76bf87a21b9838b2637983200c627a0215d46ef26aecadaaf6f31` (2 replacements) |
| `tabiclv2_regressor_colab_e80d976_colab-cli-t4_activity_run_summary.json` | `0a11b3d3397ab6cdec1c4fe82d63bacfc185d73e86a424da30b78a9701a0874c` | unchanged (no local path) |
| `tabiclv2_regressor_colab_e80d976_colab-cli-t4_activity_run_copy.ipynb` | `381d38a3c7969efe3d367eefa6a0e8c1128d8e5397c86047e01410303f82b351` | unchanged (no local path) |

**Hosted default-path run of blob `9610654e` (hosted Colab T4, 2026-10-23, `ff3532c`; the current blob until `8a48d76`, now history, REL14).** The default path
of `tutorials/tabiclv2_regressor_colab.ipynb` at `ff3532c` (blob `9610654e`, executed as committed; the fetched source's Git
blob id was re-checked), run by the Colab CLI (`colab exec -f`, a scripted run, not a browser Run all) on a fresh hosted T4
session, started 2026-10-23T15:45:24Z, wall 173.0 s, exit 0, sequential, input unchanged, 14/14 code cells, 0 errors, verdict
PASS. Evidence: [`execution-evidence/2026-10-23-ff3532c/`](execution-evidence/2026-10-23-ff3532c/). Printed: pretrained holdout MAE 38.8274 / R² 0.5823, independent test MAE 44.7756 /
R² 0.4863 (training-mean baseline holdout MAE 67.5162); the bounded CUDA fine-tune ran (`requested: True`, `ran: True`) and
`recommended_for_export` was `pretrained` (`holdout:mae`), so the fine-tuned-checkpoint export path was again not exercised;
the export printed `zip_sha256` `b655db52f70e977406b77d23211d4ecedd840430f151bed010f40819a44ec227` (`mode: pretrained`); the
fresh-process reload printed PASS, max abs prediction difference 0.0 (rtol 1e-05, atol 1e-07); Section 10 reported
`trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)` for `3cd5abc8…` and wrote its two `outputs/external_artifact/`
files; the activity cell printed "Optional activity skipped". The default path refuses nothing, so this run shows no
`Refused:` line. Local paths were replaced by `<local-run-dir>` before the first commit.

| File (under `execution-evidence/2026-10-23-ff3532c/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_ff3532c_colab-cli-t4.ipynb` | `3d28a4cd081b6d6a28146d35a0d979a528a60c60c004dd12f5341e5df127c4a1` | unchanged (no local path) |
| `tabiclv2_regressor_colab_ff3532c_colab-cli-t4_exec.log` | `960f5c461faba9a4511ea3d73e69bdee962fa186c331c51cf629b37dba02c34b` | `4394a37ef3fce77effad1f095a6eb016da3b3a3909f2a6ab2418303797003dec` (2 replacements) |
| `tabiclv2_regressor_colab_ff3532c_colab-cli-t4_run_summary.json` | `4b60f36ba73e4fc77b70574708c54e4f1923d071ba706f6d6fc2c747448593bc` | unchanged (no local path) |

**Optional-activity run of blob `9610654e` (hosted Colab T4, 2026-10-23, `ff3532c`, `RUN_ACTIVITY = True`; the current blob until `8a48d76`, now history, REL14).**
A run copy of committed blob `9610654e` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `aaa1e4956d48`,
SHA-256 `088426bec5dc56074b152d767a4a510d8da7172c999642cd496f189445f37a2f`; it differs from the committed blob only in that
assignment literal, and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by the Colab CLI
(`colab exec -f`, scripted, not a browser Run all) on a fresh hosted T4 session, started 2026-10-23T15:45:44Z, wall 158.4 s,
exit 0, sequential, input (the run copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence:
[`execution-evidence/2026-10-23-ff3532c-activity/`](execution-evidence/2026-10-23-ff3532c-activity/). The default-path outputs printed the same metrics, the same
`recommended_for_export` (`pretrained`) and the same export digest (`b655db52…`) as the default-path run above; the
fresh-process reload printed PASS with max abs prediction difference 1.52587890625e-05. The Section 11 activity
(`n_estimators` 8 → 32) printed holdout (88 rows) MAE 38.8274 → 38.7106, `mse_delta_changed_minus_canonical` −7.3514 with
`paired_se` 6.6904 (`within_two_se: True`), independent test (89 rows) MAE 44.7756 → 44.6211, delta −8.0907, `paired_se`
5.4178 (`within_two_se: True`), and `{'canonical_outputs_unchanged': True}`. Both deltas are within two paired standard
errors, so the change is within split noise, not an improvement.

| File (under `execution-evidence/2026-10-23-ff3532c-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_ff3532c_colab-cli-t4_activity.ipynb` | `040e9b3a89f5546a1f314bfc7071de6339eff74c2f33d5e68f838b5f3d75e0fe` | unchanged (no local path) |
| `tabiclv2_regressor_colab_ff3532c_colab-cli-t4_activity_exec.log` | `aba7e425af6f55f6e7d508bd8f2dfc63465fd9dcc5caa9834677e42eb74f3220` | `bfcf7cffaa9d377693e3fb893db3db8951428d2e7030c3aef0c7b6544e257bd1` (2 replacements) |
| `tabiclv2_regressor_colab_ff3532c_colab-cli-t4_activity_run_summary.json` | `172154921b4f6120f5488f8f376f74a17de3cd215f799f56d732026978d2c007` | unchanged (no local path) |
| `tabiclv2_regressor_colab_ff3532c_colab-cli-t4_activity_run_copy.ipynb` | `088426bec5dc56074b152d767a4a510d8da7172c999642cd496f189445f37a2f` | unchanged (no local path) |

**Hosted default-path run of blob `de171c3d` (hosted Colab T4, 2026-10-23, `936cc5c`; the current blob until `6c9f84c`, now history, REL14).** The default path
of `tutorials/tabiclv2_regressor_colab.ipynb` at `936cc5c` (blob `de171c3d`, executed as committed; the fetched source's Git
blob id was re-checked), run by the Colab CLI (`colab exec -f`, a scripted run, not a browser Run all) on a fresh hosted T4
session, started 2026-10-23T09:45:22Z, wall 167.4 s, exit 0, sequential, input unchanged, 14/14 code cells, 0 errors, verdict
PASS. Evidence: [`execution-evidence/2026-10-23-936cc5c/`](execution-evidence/2026-10-23-936cc5c/). Printed: pretrained holdout MAE 38.8274 / R² 0.5823, independent test MAE 44.7756 /
R² 0.4863 (training-mean baseline holdout MAE 67.5162); the bounded CUDA fine-tune ran (`requested: True`, `ran: True`) and
`recommended_for_export` was `pretrained` (`holdout:mae`), so the fine-tuned-checkpoint export path was again not exercised;
the export printed `zip_sha256` `a07696240133e5c637a189a993149a1232d756e5bc945285436f73fe049136d1` (`mode: pretrained`); the
fresh-process reload printed PASS, max abs prediction difference 1.52587890625e-05 (rtol 1e-05, atol 1e-07); Section 10
reported `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)` for `3cd5abc8…` and wrote its two
`outputs/external_artifact/` files; the activity cell printed "Optional activity skipped". The default path refuses nothing,
so this run shows no `Refused:` line. Local paths were replaced by `<local-run-dir>` before the first commit.

| File (under `execution-evidence/2026-10-23-936cc5c/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_936cc5c_colab-cli-t4.ipynb` | `455d2be6e3fa08685dd74498146e1051c77efb1af126fb0fdfbd8d4062699931` | unchanged (no local path) |
| `tabiclv2_regressor_colab_936cc5c_colab-cli-t4_exec.log` | `28779ae211e08dde6071f76cbbb9c3dc24c2721cedcf9facb1b8cc5f4419b8ad` | `015ee8e27e3da3d07eb9bf2cc4c64ee0bcc998c7d0d340f4ad9abd723cbf843d` (2 replacements) |
| `tabiclv2_regressor_colab_936cc5c_colab-cli-t4_run_summary.json` | `14678f0bcbb8d01da760b4dc66fe37338168a121d3352325ad8ebe78e14dbdfa` | unchanged (no local path) |

**Optional-activity run of blob `de171c3d` (hosted Colab T4, 2026-10-23, `936cc5c`, `RUN_ACTIVITY = True`; history, REL14).**
A run copy of committed blob `de171c3d` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `95bcf6203436`,
SHA-256 `a3033e0259a2b99e57d692151962d96423d76f6c8b2d3714427d959d5e1a0317`; it differs from the committed blob only in that
assignment literal, and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by the Colab CLI
(`colab exec -f`, scripted, not a browser Run all) on a fresh hosted T4 session, started 2026-10-23T09:45:22Z, wall 184.2 s,
exit 0, sequential, input (the run copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence:
[`execution-evidence/2026-10-23-936cc5c-activity/`](execution-evidence/2026-10-23-936cc5c-activity/). The default-path outputs printed the same metrics and export digest as the default-path run above. The
Section 11 activity (`n_estimators` 8 → 32) printed holdout (88 rows) MAE 38.8274 → 38.7106,
`mse_delta_changed_minus_canonical` −7.3514 with `paired_se` 6.6904 (`within_two_se: True`), independent test (89 rows) MAE
44.7756 → 44.6211, delta −8.0907, `paired_se` 5.4178 (`within_two_se: True`), and `{'canonical_outputs_unchanged': True}`.
Both deltas are within two paired standard errors, so the change is within split noise, not an improvement.

| File (under `execution-evidence/2026-10-23-936cc5c-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_936cc5c_colab-cli-t4_activity.ipynb` | `650124bfb56dc057ff714e14f724a39f53fc8946cbd0288590e64061375900dd` | unchanged (no local path) |
| `tabiclv2_regressor_colab_936cc5c_colab-cli-t4_activity_exec.log` | `5f55659d8a27f935898f0bf7db4f03dfa5c02458afe902ee6e7f6e3a2c4c91ef` | `add99723bf7872d52d4b5d179eaa51e031f08e8bce36f3babe4078447fc4a56f` (2 replacements) |
| `tabiclv2_regressor_colab_936cc5c_colab-cli-t4_activity_run_summary.json` | `d499581869af8ea7214aed1e92582ee6f9c6427cc7e2c88a8b3955464d4c0984` | unchanged (no local path) |
| `tabiclv2_regressor_colab_936cc5c_colab-cli-t4_activity_run_copy.ipynb` | `a3033e0259a2b99e57d692151962d96423d76f6c8b2d3714427d959d5e1a0317` | unchanged (no local path) |

**Hosted default-path run of blob `9595d23b` (hosted Colab T4, 2026-10-22, `e848401`; superseded by blob `de171c3d` from `7178e9a`; history, REL14).** The default path
of `tutorials/tabiclv2_regressor_colab.ipynb` at `e848401` (blob `9595d23bab53`, executed as committed), run by the Colab CLI
0.7.5 (`colab exec -f`, a scripted run, not a browser Run all) on a fresh hosted T4 session, started 2026-10-22T13:20:42Z,
wall 229.8 s, exit 0, sequential, input unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence:
[`execution-evidence/2026-10-22-e848401/`](execution-evidence/2026-10-22-e848401/). Printed: pretrained holdout MAE 38.8274 / R² 0.5823, independent test
MAE 44.7756 / R² 0.4863; the bounded CUDA fine-tune ran and `recommended_for_export` was `pretrained` (`holdout:mae`); the
fresh-process reload printed PASS, max abs prediction difference 0.0; Section 10 reported `trusted_digest: verified (pinned
SAMPLE_ARTIFACT_SHA256)` for `3cd5abc8…`; the activity cell printed "Optional activity skipped". Local paths were replaced by
`<local-run-dir>` before the first commit.

| File (under `execution-evidence/2026-10-22-e848401/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_e848401_colab-cli-t4.ipynb` | `90d26192cc9410801064b2f3582466a9276df229ddaa5fa6f72d425ba454266f` | unchanged (no local path) |
| `tabiclv2_regressor_colab_e848401_colab-cli-t4_exec.log` | `03f956fe6d3287e5aa4c039c1027aa4c63a5876a3516508687231cb5a9a0b3e4` | `1b2f03b19ec8e536cc364be011055abd090b6033bcd139c42aeff90bd50eb230` (2 replacements) |
| `tabiclv2_regressor_colab_e848401_colab-cli-t4_run_summary.json` | `5b9dce3bb94934b3ef9c5ab41923f8554887f8f380b48baa3a7e39cf71f3d075` | unchanged (no local path) |

**Optional-activity run of blob `9595d23b` (hosted Colab T4, 2026-10-22, `e848401`, `RUN_ACTIVITY = True`; history).**
A run copy of committed blob `9595d23b` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `4c30b14b4c90`,
SHA-256 `33a578d2af3d9b26211ffde63bacb69e2e20e42f409d4b5686008d2d4de94faf`; it differs from the committed blob only in that
assignment literal, and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by the Colab CLI 0.7.5
(`colab exec -f`, scripted, not a browser Run all) on a fresh hosted T4 session, started 2026-10-22T13:28:57Z, wall 234.9 s,
exit 0, sequential, input (the run copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence:
[`execution-evidence/2026-10-22-e848401-activity/`](execution-evidence/2026-10-22-e848401-activity/). The Section 11 activity (`n_estimators` 8 → 32) printed holdout (88
rows) MAE 38.8274 → 38.7106, `mse_delta_changed_minus_canonical` −7.3514 with `paired_se` 6.6904 (`within_two_se: True`),
independent test (89 rows) MAE 44.7756 → 44.6211, delta −8.0907, `paired_se` 5.4178 (`within_two_se: True`), and
`{'canonical_outputs_unchanged': True}`.

| File (under `execution-evidence/2026-10-22-e848401-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_regressor_colab_e848401_colab-cli-t4_activity.ipynb` | `809b5f52a2e4648330bdfb14ac7ae2e4ce866b0dcf4fe028184feefda40b12a3` | unchanged (no local path) |
| `tabiclv2_regressor_colab_e848401_colab-cli-t4_activity_exec.log` | `239607c37693d074df6f5e53be982e0a33fe022095a9d32806cfd7d94dc69897` | `92892e2b86a135878a780f87f1a9489867fd49ebfa5061a5f7927af777835936` (2 replacements) |
| `tabiclv2_regressor_colab_e848401_colab-cli-t4_activity_run_summary.json` | `28b7fce7c21a11253e65208cff003710cb951d5c788c9da0008f9ef60fd5004f` | unchanged (no local path) |
| `tabiclv2_regressor_colab_e848401_colab-cli-t4_activity_run_copy.ipynb` | `33a578d2af3d9b26211ffde63bacb69e2e20e42f409d4b5686008d2d4de94faf` | unchanged (no local path) |

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
- **v2 tag (2026-10-24):** the default pin is now `sample-artifact-diabetes-v2` (`693cc996…`). That release is not immutable either (`.immutable` returns `false`), so the same deviation applies to it and integrity rests on the same pinned-digest check; the v1 tag stays published unchanged.

## Current status

**Current status (2026-10-29): Candidate.** The current notebook blob is `5e7d4c4e` (regenerated from `46101cf`: a cancelled or empty upload is now refused with one `Refused: (no file): …` line and Run all stops at that cell, DAT20). For this blob, its default path (2026-10-29; 14 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (2026-10-29; `DATA_SOURCE=Upload CSV;USE_BYOD=True;BYOD_CSV_PATH=/content/sample_data/california_housing_test.csv`: exactly one `Refused:` line at cell `tabiclv2_regressor-14`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`); its optional-activity run (`RUN_ACTIVITY = True`) is not recorded for this blob, and the runs of earlier blobs are history for it (REL14). The records of the previous blob `a84d74ba` below are history for it.

**Pending (tier C review fixes, 2026-10-25):** `tutorials/tabiclv2_regressor_colab.ipynb` blob `a84d74ba7ea4870f792a1b83b6e8f73158c94a3f`, generated at `299b529` (T98-1, T98-2, T98-4, T98-5). It changes markdown, the use notice, `metadata.dimer` and the `NOTEBOOK_SOURCE` record only. Status **Candidate**. Its hosted default-path run and optional-activity run (`RUN_ACTIVITY = True` run copy) are recorded: Colab CLI `colab exec -f` scripted executions (not a browser Run all) on fresh Colab T4s at PR #37 head `100088f`, 2026-10-25, both PASS, 14/14 code cells, 0 errors, input unchanged (evidence in `execution-evidence/2026-10-25-100088f/` and `execution-evidence/2026-10-25-100088f-activity/`); the metrics equal the `426cd4d6` runs. These runs cover the default path and the activity. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser run of the interactive journeys). The runs of `426cd4d6` and earlier blobs are history (REL14).

**Current status (2026-10-24): Candidate.** **2026-10-24 (v2 sample artifact by default):** the current E2E notebook blob is `426cd4d6` (`tutorials/tabiclv2_regressor_colab.ipynb` at `69f7efb`, code commit `384864f`, branch `chore/sample-artifact-v2-default`): Section 10 now pins `sample-artifact-diabetes-v2` (`tabiclv2_regressor_artifact.zip`, SHA-256 `693cc996249e3b8364745244d0bcfe8e7ab9cec1591e6f98f014ec54cd39dbce`, 106,347,998 bytes), the bundle-format v2 re-issue of the recorded `6b95e73` run's export: `checkpoints/best.safetensors` is the pinned base's safetensors conversion, the training context is byte-identical, every other manifest field is unchanged and a `reissue` block records the conversion. It is not the export of a new hosted run. The v1 asset (`sample-artifact-diabetes-v1`, `3cd5abc8…`) stays published and readable through the deprecated v1 path. Its hosted default-path run and optional-activity run (`RUN_ACTIVITY = True` run copy) are recorded: Colab CLI `colab exec -f` scripted executions (not a browser Run all) on fresh Colab T4s at commit `1f69142`, 2026-10-24, both PASS, 14/14 code cells, 0 errors, input unchanged (evidence in `execution-evidence/2026-10-24-1f69142/` and `execution-evidence/2026-10-24-1f69142-activity/`). Section 10 verified the pinned v2 asset (`sample-artifact-diabetes-v2`, `693cc996…`, `tabicl-dimer-regressor-v2`) and loaded it with no v1 `DeprecationWarning`; the export reloaded with PASS. The `c49e98e7` runs below are history for this blob (REL14), and it stays **Candidate** pending only REL12 and REL2. A local CPU check (not clean-runtime evidence; `CUDA_VISIBLE_DEVICES=-1`, torch 2.11.0+cpu, 2026-10-24) ran the section's own stages (`weights`, then `external_artifact` with `source='pinned'`, `external_reconstruct` and `external_predict`, the last three with `torch.load` patched to raise): the stage downloaded the v2 asset fresh, verified `693cc996…`, reported `checkpoint_binding: identical to the pinned safetensors conversion of the base`, rebuilt serving state from the bundle and scored the eight diabetes rows (within 0.0001 of the `e80d976` hosted record of the v1 pin). **2026-10-24 (bundle format v2):** the previous E2E notebook blob was `c49e98e7` (`tutorials/tabiclv2_regressor_colab.ipynb` at `48885fc`, code commit `a43d226`, PR #35, branch `fix/bundle-safetensors`): exported serving bundles carry `checkpoints/best.safetensors` (format `tabicl-dimer-regressor-v2`), fine-tuned weights are converted from tabicl's `best.ckpt` right after training and no pickle is packaged, and deprecated v1 bundles stay readable with `torch.load(weights_only=True)` and a `DeprecationWarning`. At that blob the external-artifact section still pinned `sample-artifact-diabetes-v1` (a v1 bundle). Its default path is covered by the `48885fc` record (*Hosted default-path run of blob `c49e98e7`*, near the top of this file; Colab CLI `colab exec -f`, scripted, not a browser Run all, fresh Colab Tesla T4, 2026-10-24, exit 0, 14/14 code cells, 0 error outputs, 167.0 s, input unchanged, PASS, evidence in `execution-evidence/2026-10-24-48885fc/`): it served the pinned safetensors base, printed the same metrics as the `7baea7e2` runs, recommended the pretrained model for export (so the fine-tuned v2 export path was not exercised), reloaded the export in a fresh process (PASS, max abs prediction difference 0.0), and verified the pinned v1 asset `3cd5abc8…` before extraction. The optional-activity gate is covered by the `48885fc` optional-activity record (`RUN_ACTIVITY = True` run copy, PASS, 189.9 s, evidence in `execution-evidence/2026-10-24-48885fc-activity/`). The runs of earlier blobs, including `7baea7e2` (PR #34, `e80d976`), are history, not evidence for `c49e98e7` (REL14). The notebook stays **Candidate** pending only REL12 and REL2 for `c49e98e7`. **2026-10-24 (safetensors serving):** the E2E notebook blob is now `7baea7e2` (code commit `8a48d76`): the base model is served from a pinned safetensors conversion of the checkpoint (`tabicl-regressor-v2-20260212.safetensors`, SHA-256 `ad4b0ec0…`), read once from the digest-verified `.ckpt` with `torch.load(weights_only=True)`. Predictions are identical to the `.ckpt` load on CPU. Its hosted default-path run and optional-activity run (a `RUN_ACTIVITY = True` run copy) are recorded: Colab CLI scripted runs on a fresh T4, commit `e80d976`, 2026-10-24, both PASS, 14/14 code cells, 0 errors (PR #34); the `9610654e` runs named below are history for it (REL14). The rest of this paragraph is the 2026-10-23 status of the previous blob. **2026-10-23: Candidate.** The current E2E notebook blob is `9610654e` (`tutorials/tabiclv2_regressor_colab.ipynb`, regenerated at `de7d72e` from code commit `6c9f84c` on `fix/section8-own-rows-refusal`, PR head `ff3532c`: Section 8 checks your own rows for text in a numeric feature before the model is loaded and refuses it in one `Refused: <file>: <reason>` line naming the columns, values and CSV lines, with nothing scored or written; NOTEBOOK_SPEC 3.0 DAT19/DAT20). Its default path is covered by the `ff3532c` record (*Hosted default-path run of blob `9610654e`*, near the top of this file), evidence in `execution-evidence/2026-10-23-ff3532c/`: Colab CLI `colab exec -f` (scripted, not a browser Run all) on a fresh Colab Tesla T4, exit 0, 14/14 code cells, 0 error outputs, 173.0 s, input unchanged, verdict PASS. In that run the bounded CUDA fine-tune ran and the pretrained model was recommended for export (selection on holdout MAE), so the fine-tuned-checkpoint export path was again not exercised; the export reloaded in a fresh process (PASS, max abs prediction difference 0.0); and Section 10 verified the pinned `sample-artifact-diabetes-v1` asset (`3cd5abc8…`) before extraction. The optional-activity gate (the owner's 2026-10-22 ruling that the Section 11 activity runs once on the current blob before promotion) is covered by the `ff3532c` optional-activity record (`RUN_ACTIVITY = True` run copy, PASS, 158.4 s, evidence in `execution-evidence/2026-10-23-ff3532c-activity/`). The runs of earlier blobs are history, not evidence for `9610654e` (REL14): the `936cc5c` default-path and optional-activity records cover blob `de171c3d`, the `e848401` records cover blob `9595d23b`, and the `1e1a1ab` record covers blob `8cb91a8e`. Neither hosted run of `9610654e` exercised a refusal; the one-line `Refused:` display, including the Section 8 own-rows refusal, is covered by `tests/test_refusal_display.py` and is part of REL12 below. The notebook stays **Candidate** until these are recorded for the current blob `9610654e`:

- **REL12:** a BYOD run in which one representative input is accepted through the downstream stages and one incompatible input is refused (the browser kit also refuses your own Section 8 rows with text in a numeric feature);
- **REL2:** a browser run of the interactive journeys (BYOD upload dialog, optional-activity toggle, own-artifact ZIP); the scripted CLI runs count for the default path and the activity only.

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