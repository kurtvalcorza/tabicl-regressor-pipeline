"""The E2E notebook's closing external-artifact section (NOTEBOOK_SPEC 3.0 §19.1, XAS1–XAS7, SART1–SART11).

Offline and model-free (CI's light pins only): the pin, the digest refusal before extraction, the SART10 contract check,
that the section's cells use only names the setup cells define (XAS5), the `outputs/external_artifact/` path (XAS6), the
exact-URL repository-reference rule (ST1) and the deterministic export.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabiclv2_regressor_colab.ipynb"
PINNED_URL = "https://github.com/kurtvalcorza/tabicl-regressor-pipeline/releases/download/sample-artifact-diabetes-v1/tabiclv2_regressor_artifact.zip"
PINNED_SHA256 = "3cd5abc8d364398b6ff2b2716b48ca3d45781f6893accc097c93eec768dbe828"
SECTION_START = "## 10. Using an artifact made elsewhere"
SECTION_END = "## 11. Optional activity"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("xas_tutorial_stages", TOOLS / "tutorial_stages.py")


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _cells() -> list[dict]:
    return json.loads(E2E.read_text(encoding="utf-8"))["cells"]


def _section_cells() -> list[dict]:
    cells = _cells()
    start = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(SECTION_START))
    end = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(SECTION_END))
    return cells[start:end]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    try:
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    except OSError:
        shutil.copytree(ROOT / "src", root / "src")
    return root


def _torch_stub(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.11.0", cuda=types.SimpleNamespace(is_available=lambda: False)))


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


# ---------------------------------------------------------------- the pin (SART6, XAS2)


def test_section_pins_the_release_asset_by_url_and_digest() -> None:
    assert STAGES.SAMPLE_ARTIFACT_URL == PINNED_URL
    assert STAGES.SAMPLE_ARTIFACT_SHA256 == PINNED_SHA256
    assert STAGES.SAMPLE_ARTIFACT_PRODUCER["commit"].startswith("6b95e73")
    assert STAGES.SAMPLE_ARTIFACT_PRODUCER["release_tag"] == "sample-artifact-diabetes-v1"


def test_default_path_fetches_the_pin_and_never_this_runs_export(tmp_path, monkeypatch) -> None:
    """XAS2: the default source downloads the pinned URL at the pinned digest; the current run's export is never read."""
    _torch_stub(monkeypatch)
    root = _root(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / f"{STAGES.STEM}_artifact.zip").write_bytes(b"this run's export: must never be opened here")
    seen = []

    def fake_fetch(cache, url, expected):
        seen.append((cache, url, expected))
        raise RuntimeError("stop after the fetch")

    monkeypatch.setattr(STAGES, "fetch_pinned_artifact", fake_fetch)
    run = STAGES.Run(root, outputs, tmp_path / "weights", {})
    with pytest.raises(RuntimeError, match="stop after the fetch"):
        _quiet(STAGES.stage_external_artifact, run)
    assert seen == [(root / "inputs" / "external_artifact.zip", PINNED_URL, PINNED_SHA256)]
    for fn in (STAGES.stage_external_artifact, STAGES.stage_external_reconstruct, STAGES.stage_external_predict, STAGES.external_serving, STAGES.external_sample_rows):
        body = inspect.getsource(fn)
        assert "{STEM}_artifact.zip" not in body and "artifact-reload" not in body and "export_reference" not in body, fn.__name__
        assert "read_state(\"encoded" not in body and "read_frame(" not in body and "data.json" not in body, fn.__name__


# ---------------------------------------------------------------- integrity (XAS4, VER6)


def _zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("artifact.json", "{}")
    return path


def test_pinned_digest_mismatch_refuses_before_extraction(tmp_path) -> None:
    archive = _zip(tmp_path / "tampered.zip")
    cache = tmp_path / "cache" / "external_artifact.zip"
    with pytest.raises(ValueError, match=r"Trusted digest mismatch .* the pinned SAMPLE_ARTIFACT_SHA256 is .*nothing was extracted"):
        STAGES.fetch_pinned_artifact(cache, archive.resolve().as_uri(), PINNED_SHA256)
    assert not cache.exists() and not cache.with_name(cache.name + ".part").exists()


def test_pinned_digest_mismatch_stops_the_stage_with_nothing_extracted(tmp_path, monkeypatch) -> None:
    """The stage itself, given a tampered download, refuses before safe_extract_zip runs and writes no output."""
    _torch_stub(monkeypatch)
    root = _root(tmp_path)
    archive = _zip(tmp_path / "tampered.zip")
    monkeypatch.setattr(STAGES, "SAMPLE_ARTIFACT_URL", archive.resolve().as_uri())
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "pinned"})
    with pytest.raises(ValueError, match="Trusted digest mismatch"):
        _quiet(STAGES.stage_external_artifact, run)
    assert not (root / "state" / "external_bundle").exists()
    assert not (root / "state" / "external_artifact.json").exists()
    assert not (tmp_path / "outputs" / STAGES.EXTERNAL_DIR).exists()


def test_pinned_digest_match_is_kept_and_reused(tmp_path) -> None:
    archive = _zip(tmp_path / "standin.zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cache = tmp_path / "cache" / "external_artifact.zip"
    assert STAGES.fetch_pinned_artifact(cache, archive.resolve().as_uri(), digest) == cache
    archive.unlink()  # a second call reuses the verified cache without the network
    assert STAGES.fetch_pinned_artifact(cache, "file:///nonexistent.zip", digest) == cache


def test_own_artifact_with_a_wrong_digest_is_refused_before_extraction(tmp_path, monkeypatch) -> None:
    _torch_stub(monkeypatch)
    root = _root(tmp_path)
    archive = _zip(tmp_path / "mine.zip")
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "own", "zip_path": str(archive), "expected_zip_sha256": "0" * 64})
    with pytest.raises(ValueError, match="EXPECTED_ZIP_SHA256 is 0{64}.*nothing was extracted"):
        _quiet(STAGES.stage_external_artifact, run)
    assert not (root / "state" / "external_bundle").exists()
    assert not (tmp_path / "outputs" / STAGES.EXTERNAL_DIR).exists()


def test_artifact_from_another_dataset_breaks_the_data_contract() -> None:
    """SART10: a hash-valid artifact built on another table (California Housing) is refused."""
    contract = STAGES.diabetes_contract()
    assert contract["targetColumn"] == "target" and contract["featureColumns"] == ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    STAGES.check_data_contract({k: contract[k] for k in ("featureColumns", "targetColumn")}, contract)
    with pytest.raises(RuntimeError, match=r"default data contract \(SART10\)"):
        STAGES.check_data_contract({"featureColumns": ["MedInc", "HouseAge"], "targetColumn": "MedHouseVal"}, contract)
    with pytest.raises(RuntimeError, match=r"default data contract \(SART10\)"):
        STAGES.check_data_contract({"featureColumns": contract["featureColumns"], "targetColumn": "y"}, contract)


def test_pretrained_artifact_must_carry_the_base_checkpoint(tmp_path) -> None:
    P = STAGES.package(_root(tmp_path))
    manifest = {"baseCheckpoint": P.BASE_CHECKPOINT_NAME, "baseModelRevision": P.MODEL_REVISION, "baseModelSha256": P.BASE_MODEL_SHA256, "mode": "pretrained", "digests": {"checkpointSha256": P.BASE_MODEL_SHA256}}
    assert STAGES.check_bundle_identity(P, manifest)["mode"] == "pretrained"
    swapped = json.loads(json.dumps(manifest))
    swapped["digests"]["checkpointSha256"] = "f" * 64
    with pytest.raises(RuntimeError, match=f"must be the pinned base checkpoint with SHA-256 {P.BASE_MODEL_SHA256}"):
        STAGES.check_bundle_identity(P, swapped)
    other_base = json.loads(json.dumps(manifest))
    other_base["baseModelSha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="not produced on the pinned base checkpoint"):
        STAGES.check_bundle_identity(P, other_base)


def test_non_numeric_value_in_a_numeric_feature_is_refused() -> None:
    import pandas as pd

    rows = pd.DataFrame({"age": ["0.01", "abc"], "sex": [0.05, 0.05]})
    with pytest.raises(ValueError, match=r"mine\.csv: numeric feature 'age' has 1 non-numeric value\(s\), e\.g\. \['abc'\]"):
        STAGES.check_numeric_features("mine.csv", rows, ["age", "sex"], {})


# ---------------------------------------------------------------- XAS5: setup cells and its own cells only


def _defined(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def test_section_cells_reference_only_setup_cell_names() -> None:
    setup = [c for c in _cells() if c["cell_type"] == "code" and c.get("metadata", {}).get("cellView") == "form"]
    assert len(setup) == 4, "the four Infrastructure cells (Sections 1-3)"
    setup_names = set().union(*(_defined(ast.parse(_src(c))) for c in setup))
    section_code = [_src(c) for c in _section_cells() if c["cell_type"] == "code"]
    assert len(section_code) == 3
    own = set().union(*(_defined(ast.parse(s)) for s in section_code))
    used = {n.id for s in section_code for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    foreign = used - own - setup_names - set(dir(builtins))
    assert not foreign, f"the section uses names defined outside the setup cells and itself: {sorted(foreign)}"
    assert {"run_stage", "ROOT", "Path"} <= used & setup_names


def test_section_states_its_dependency_and_its_difference_from_the_reload() -> None:
    md = "\n".join(_src(c) for c in _section_cells() if c["cell_type"] == "markdown")
    assert "**This section needs only the setup cells (Sections 1–3).**" in md
    assert "the reload shows that this run's export survives serialization, and this section shows that an artifact made elsewhere can be validated and served here" in md
    assert "**Trust boundary.**" in md and "outputs/external_artifact/" in md


def test_section_defaults_run_the_pinned_artifact_and_sample_rows_without_upload(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "run_stage": lambda stage, **o: calls.append((stage, o))}
    for source in (_src(c) for c in _section_cells() if c["cell_type"] == "code"):
        exec(compile(source, "<section>", "exec"), ns)
    assert calls == [("external_artifact", {"source": "pinned"}), ("external_reconstruct", {}), ("external_predict", {"rows_path": ""})]


def test_section_runs_last_on_the_default_path_after_the_reload() -> None:
    code = [_src(c) for c in _cells() if c["cell_type"] == "code"]
    reload_at = next(i for i, s in enumerate(code) if "run_stage('reload')" in s)
    section_at = next(i for i, s in enumerate(code) if "run_stage('external_artifact'" in s)
    assert reload_at < section_at
    later = "\n".join(code[section_at + 3:])
    assert "run_stage(" not in later.replace("run_stage('activity'", ""), "only the optional activity may follow the section"
    meta = json.loads(E2E.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "3.0" and meta["notebook_kind"] == "TUTORIAL"
    assert not list((ROOT / "tutorials").glob("*_artifact_inference_colab.ipynb")), "KND11: no separate artifact-inference notebook"


def test_interpretation_precedes_the_section_which_closes_the_canonical_path() -> None:
    heads = [line for c in _cells() if c["cell_type"] == "markdown" for line in _src(c).split("\n") if line.startswith("## ")]
    order = ("## 9. Export a DIMER-style serving bundle", "## Interpretation and limits", SECTION_START, SECTION_END, "## Troubleshooting", "## References")
    at = [next(i for i, h in enumerate(heads) if h.startswith(prefix)) for prefix in order]
    assert at == sorted(at), dict(zip(order, at, strict=True))


# ---------------------------------------------------------------- XAS6: the outputs path


class _Serving:
    n_estimators, random_state, device = 8, 42, "cpu"

    def predict(self, X):
        return np.full(len(X), 150.0)


def test_predictions_and_provenance_go_to_outputs_external_artifact(tmp_path, monkeypatch) -> None:
    root = _root(tmp_path)
    outputs = tmp_path / "outputs"
    run = STAGES.Run(root, outputs, tmp_path / "weights", {})
    P = STAGES.package(root)
    contract = STAGES.diabetes_contract()
    manifest = {"featureColumns": contract["featureColumns"], "targetColumn": "target", "inference": {"categoricalEncoders": {}}}
    art = {"source": "pinned", "zip": "tabiclv2_regressor_artifact.zip", "zip_sha256": PINNED_SHA256, "url": PINNED_URL, "trusted_digest": "verified (pinned SAMPLE_ARTIFACT_SHA256)", "mode": "pretrained", "manifest": manifest}
    monkeypatch.setattr(STAGES, "external_serving", lambda _run: (P, art, manifest, _Serving()))
    _quiet(STAGES.stage_external_predict, run)
    assert STAGES.EXTERNAL_DIR == "external_artifact"
    written = sorted(p.relative_to(outputs).as_posix() for p in outputs.rglob("*") if p.is_file())
    assert written == ["external_artifact/tabiclv2_regressor_external_predictions.csv", "external_artifact/tabiclv2_regressor_external_result.json"]
    result = json.loads((outputs / "external_artifact" / "tabiclv2_regressor_external_result.json").read_text())
    assert result["artifact"]["zip_sha256"] == PINNED_SHA256 and result["input"]["kind"] == "sample" and result["input"]["rows"] == 8
    assert [f["verdict"] for f in result["input_manifest"]["findings"] if f["input"] == "missing-column-probe"] == ["rejected"]


def test_sample_rows_are_holdout_rows_outside_the_support_partition() -> None:
    contract = STAGES.diabetes_contract()
    rows = STAGES.external_sample_rows(contract["featureColumns"])
    train, holdout, _test = STAGES.random_60_20_20(STAGES.diabetes_frame())
    assert list(rows.columns) == ["row_id", *contract["featureColumns"]] and len(rows) == 8
    assert set(rows["row_id"]) <= set(holdout.index) and not set(rows["row_id"]) & set(train.index)


# ---------------------------------------------------------------- deterministic export (packet lesson 3)


def test_export_zip_is_deterministic(tmp_path) -> None:
    art = tmp_path / "artifact"
    (art / "checkpoints").mkdir(parents=True)
    (art / "checkpoints" / "best.ckpt").write_bytes(b"ckpt")
    (art / "artifact.json").write_text("{}\n")
    (art / "training_context.parquet").write_bytes(b"ctx")
    first = STAGES.deterministic_zip(art, tmp_path / "a.zip")
    for path in art.rglob("*"):
        if path.is_file():
            path.touch()  # new mtimes must not change the digest
    second = STAGES.deterministic_zip(art, tmp_path / "b.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as zf:
        infos = zf.infolist()
    assert [i.filename for i in infos] == ["artifact.json", "checkpoints/best.ckpt", "training_context.parquet"]
    assert {i.date_time for i in infos} == {(1980, 1, 1, 0, 0, 0)} and {i.external_attr >> 16 for i in infos} == {0o644}
    assert not list(tmp_path.glob("*.tmp"))
    assert "make_archive" not in inspect.getsource(STAGES.stage_export)


# ---------------------------------------------------------------- ST1: only the exact pinned URL


VALIDATOR = _load("xas_validate_release_assets", TOOLS / "validate_release_assets.py")
_REPO = "https://github.com/kurtvalcorza/tabicl-regressor-pipeline"


@pytest.mark.parametrize(
    "reference",
    [
        f"{_REPO}/releases/download/../../../archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v1/../../../archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/%2e%2e/%2E%2E/archive/refs/heads/main.zip",
        "https://GITHUB.COM/kurtvalcorza/tabicl-regressor-pipeline/releases/download/%2E%2E/archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v1/tabiclv2_regressor_artifact.zip?raw=1",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v1/tabiclv2_regressor_artifact.zip#x",
        "https://github.com/kurtvalcorza/tabicl-classifier-pipeline/releases/download/sample-artifact-magic-v1/tabiclv2_classifier_artifact.zip",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v2/tabiclv2_regressor_artifact.zip",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v1/tabiclv2_regressor_artifact.zip.sig",
        f"{_REPO}.git",
        "https://GitHub.com/KurtValcorza/tabicl-regressor-pipeline/archive/refs/heads/main.zip",
    ],
    ids=["traversal", "tag-then-traversal", "pct2e", "pct2E-upper-host", "query", "fragment", "other-repo", "other-tag", "suffix", "clone", "case"],
)
def test_st1_only_the_exact_pinned_release_asset_is_allowed(reference: str) -> None:
    source = f'SAMPLE_ARTIFACT_URL = "{reference}"\n'
    assert VALIDATOR.disallowed_own_references(source, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == [reference]


def test_st1_exact_pinned_url_passes_and_matches_the_runner() -> None:
    assert VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL == PINNED_URL == STAGES.SAMPLE_ARTIFACT_URL
    assert STAGES.SAMPLE_ARTIFACT_TAG in PINNED_URL
    source = f'SAMPLE_ARTIFACT_URL = "{PINNED_URL}"  # literal\n'
    assert VALIDATOR.disallowed_own_references(source, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == []
    assert VALIDATOR.disallowed_own_references(source) == [PINNED_URL], "kernel cells allow no repository reference at all"


def test_st1_real_notebook_passes_the_validator() -> None:
    runner = VALIDATOR.carried_runner(E2E)
    assert VALIDATOR.disallowed_own_references(runner, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == []
    VALIDATOR.validate_notebooks()
