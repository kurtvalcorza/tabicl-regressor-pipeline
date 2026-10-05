"""Regression tests for the 2026-10-05 notebook review (TIR-M1..M4, TIR-m1..m2; TIRA-M1..M3, TIRA-m1..m3).

They need only CI's dependencies (no torch, no tabicl, no lightgbm, no checkpoint): they exec the notebooks' own kernel
cells with stand-ins for `run_stage` and `google.colab`, run the stage runners' model-free parts (`data`, `validate`,
the sample-bundle assembly, `rows`, the trusted-digest and binding checks) in-process, and check the generated
notebooks statically. Each test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabiclv2_regressor_colab.ipynb"
AI = ROOT / "tutorials" / "tabiclv2_regressor_artifact_inference_colab.ipynb"
SAMPLE = ROOT / "examples" / "sample-bundle"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("tir_tutorial_stages", TOOLS / "tutorial_stages.py")
AI_STAGES = _load("tir_tutorial_stages_ai", TOOLS / "tutorial_stages_artifact_inference.py")


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_cells(path: Path) -> list[str]:
    return [_src(c) for c in _nb(path)["cells"] if c["cell_type"] == "code"]


def _cell_with(path: Path, needle: str) -> str:
    found = [s for s in _code_cells(path) if needle in s]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


@contextlib.contextmanager
def _colab(queue):
    files = types.SimpleNamespace(calls=0)

    def upload():
        files.calls += 1
        return queue.pop(0)

    files.upload = upload
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield files
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _no_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # `from google.colab import files` -> ImportError


def _kernel(tmp_path: Path) -> tuple[dict, list]:
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o))}
    return ns, calls


def _run(root: Path, outputs: Path, options: dict | None = None, module=STAGES):
    src = root / "src"
    if not src.exists():
        root.mkdir(parents=True, exist_ok=True)
        src.symlink_to(ROOT / "src", target_is_directory=True)
    return module.Run(root, outputs, root / "weights", options or {})


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _diabetes() -> pd.DataFrame:
    from sklearn.datasets import load_diabetes

    return load_diabetes(as_frame=True).frame


# ---------------------------------------------------------------- TIR-M1 / TIRA-M2: isolated runtime


@pytest.mark.parametrize("path", [E2E, AI], ids=["TIR-M1", "TIRA-M2"])
def test_m1_no_kernel_install_and_no_restart_instruction(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text and "restart the runtime" not in text.replace("no runtime restart", "")
    own = [_src(c) for c in _nb(path)["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for src in own:
        assert not re.search(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install", src, re.M), src[:80]
    install = _cell_with(path, "# @title Infrastructure: build (or reuse) the isolated")
    assert "'--require-hashes'" in install and "--managed-python" in install
    assert "MPLBACKEND='Agg'" in install and "'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'" in install


def _exec_check_cell(ns: dict, path: Path, **overrides) -> None:
    src = _cell_with(path, "# @title Infrastructure: check the runtime")
    src = re.sub(r"'environment': [0-9.]+}", "'environment': 0.0}", src, count=1)  # CI disks are small; the rule is tested, not the size
    src = re.sub(r"\{'weights': max\(0\.0, [0-9.]+", "{'weights': max(0.0, 0.0", src, count=1)
    for name, value in overrides.items():
        src = _set(src, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<check>", "exec"), ns)


@pytest.mark.parametrize("path", [E2E, AI], ids=["TIR-M1", "TIRA-M2"])
def test_m1_section1_is_idempotent(path: Path, tmp_path, monkeypatch) -> None:
    """Re-running the Section 1 cell keeps the run directory, so later cells are not stranded."""
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns, path)
    first = ns["ROOT"]
    (first / "tutorial_stages.py").write_text("# carried")
    _exec_check_cell(ns, path)
    assert ns["ROOT"] == first and (first / "tutorial_stages.py").is_file()
    _exec_check_cell(ns, path, NEW_RUN_DIRECTORY=True)
    assert ns["ROOT"] != first


def test_m1_second_run_all_reuses_the_matching_environment(tmp_path, monkeypatch) -> None:
    """TIR-M1: the venv is keyed on the lock digest; a second exec of the install cell builds nothing."""
    import subprocess as real_subprocess

    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns, E2E)
    ns["ENV_ROOT"] = tmp_path / "uvroot"
    ns["NOTEBOOK_SOURCE"] = {"revision": "test"}
    src = _cell_with(E2E, "# @title Infrastructure: build (or reuse) the isolated")
    version = re.search(r"UV = ENV_ROOT / 'uv-([0-9.]+)'", src).group(1)
    ns["ENV_ROOT"].mkdir()
    uv = ns["ENV_ROOT"] / f"uv-{version}"
    uv.write_bytes(b"uv stand-in")
    (ns["ENV_ROOT"] / f"uv-{version}.sha256").write_text(hashlib.sha256(b"uv stand-in").hexdigest())
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        out = json.dumps({"python": "3.12.12", "torch": "x", "tabicl": "x", "numpy": "x", "pandas": "x", "scikit-learn": "x", "lightgbm": "x", "cuda": True})
        return types.SimpleNamespace(stdout=out + "\n", returncode=0)

    fake = types.SimpleNamespace(run=fake_run, Popen=real_subprocess.Popen, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
    for attempt in range(2):
        ns["subprocess"] = fake
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src.replace("import zipfile\n", "import zipfile\nsubprocess = globals()['subprocess']\n", 1), "<install>", "exec"), ns)
        ns["subprocess"] = fake
        assert ns["environment_reused"] is (attempt == 1)
    builds = [c for c in commands if c[1] in ("venv", "pip")]
    assert len(builds) == 2, builds  # one venv + one install, on the first exec only
    assert ns["VENV"].name == "venv-" + ns["LOCK_SHA256"][:16]


# ---------------------------------------------------------------- TIR-M2: bounded fine-tuning on the default path


def test_M2_fine_tuning_runs_on_the_default_path_bounded_and_cpu_skip_is_recorded() -> None:
    """TIR-M2: RUN_FINE_TUNING defaults to True; the run is bounded (3 epochs, 300 s, patience 2), selected on the holdout,
    and a CPU runtime records why it was skipped instead of raising."""
    cell = _cell_with(E2E, "RUN_FINE_TUNING = True  # @param")
    assert "run_stage('condition', run_fine_tuning=RUN_FINE_TUNING" in cell
    assert STAGES.FINE_TUNE == {"epochs": 3, "time_limit": 300, "patience": 2, "eval_metric": "mae"}
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_condition"):source.index("def stage_report")]
    assert 'opts.get("run_fine_tuning", True)' in body
    assert 'fine_tune["skipped"] = "no CUDA device' in body and "raise RuntimeError('TabICLv2 fine-tuning requires CUDA')" not in body
    assert "time_limit=FINE_TUNE[\"time_limit\"]" in body and "patience=FINE_TUNE[\"patience\"]" in body
    assert 'selection = f"holdout:{metric}"' in body and "MIN_SELECTION_HOLDOUT_ROWS" in body
    run_all = _nb(E2E)["cells"][0]
    assert "bounded to 3 epochs and 5 minutes" in _src(run_all)


# ---------------------------------------------------------------- TIR-M3 / TIRA-M3: guided layer

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion")


@pytest.mark.parametrize("path", [E2E, AI], ids=["TIR-M3", "TIRA-M3"])
def test_m3_guided_layer_and_collapsed_infrastructure(path: Path) -> None:
    nb = _nb(path)
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for heading in GUIDED:
        assert heading in md, heading
    assert "{{" not in md and "{MODEL_ID}" not in md
    infra = [c for c in nb["cells"] if c["cell_type"] == "code" and _src(c).startswith("# @title Infrastructure:")]
    assert len(infra) == 4 and all(c["metadata"].get("cellView") == "form" for c in infra)
    assert _cell_with(path, "RUN_ACTIVITY = False  # @param")


def test_M3_ensemble_size_activity_reads_the_paired_se_and_writes_only_to_its_directory() -> None:
    """TIR-M3: the activity changes the ensemble size, reads the paired standard error, and leaves canonical outputs unchanged."""
    cell = _cell_with(E2E, "RUN_ACTIVITY = False  # @param")
    assert "ACTIVITY_N_ESTIMATORS = 32  # @param" in cell
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert 'write_output(f"activity/' in body and "before != after" in body and "paired_se(" in body and "within_two_se" in body


def test_TIRA_M3_every_bundle_refusal_has_a_troubleshooting_row() -> None:
    """TIRA-M3: the troubleshooting table names the refusals the artifact stage can raise."""
    md = "\n".join(_src(c) for c in _nb(AI)["cells"] if c["cell_type"] == "markdown")
    table = md[md.index("## Troubleshooting"):]
    for message in ("Trusted digest mismatch", "Unexpected or missing artifact files", "pinned base checkpoint", "non-numeric value"):
        assert message in table, message


@pytest.mark.parametrize("runner", ["tutorial_stages.py", "tutorial_stages_artifact_inference.py"])
def test_no_quality_assert_in_stage_runners(runner: str) -> None:
    tree = ast.parse((TOOLS / runner).read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]


# ---------------------------------------------------------------- TIR-M4: unparseable targets refused, blank targets counted


def _csv(tmp_path: Path, name: str, frame: pd.DataFrame) -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _byod_run(tmp_path: Path, path: Path, **extra):
    return _run(tmp_path / "run", tmp_path / "outputs", {"data_source": "Upload CSV", "use_byod": True, "target_column": "target", "paths": {"csv": str(path)}, **extra})


def test_M4_blank_targets_are_dropped_and_reported_in_the_manifest(tmp_path) -> None:
    """TIR-M4: a single-CSV upload with 5 blank targets reports 5 in the manifest (it used to report 0)."""
    frame = _diabetes().astype({"target": object})
    frame.loc[[3, 10, 50, 60, 70], "target"] = None
    run = _byod_run(tmp_path, _csv(tmp_path, "blank.csv", frame))
    _quiet(STAGES.stage_data, run)
    assert json.loads((run.state / "data.json").read_text())["dropped_missing_target_rows_before_split"] == {"csv": 5}
    _quiet(STAGES.stage_validate, run)
    manifest = json.loads((run.out / "tabiclv2_regressor_input_manifest.json").read_text())
    assert manifest["inputs"][0]["dropped_missing_target_rows_before_split"] == 5
    assert json.loads((run.state / "encoded.json").read_text())["dropped"]["before_split"] == 5


def test_M4_thousands_separators_are_refused_naming_the_target_and_a_value(tmp_path) -> None:
    """TIR-M4: the review's thousands-separator CSV stops in Section 4 naming the target and `1,510.0`; nothing is written."""
    frame = _diabetes().astype({"target": object})
    big = frame["target"].astype(float) * 10
    frame["target"] = [f"{v:,.1f}" if v >= 1000 else f"{v:.1f}" for v in big]
    path = tmp_path / "sep.csv"
    frame.to_csv(path, index=False)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "tabiclv2_regressor_result.json").write_text("{}")  # an earlier run's export
    run = _byod_run(tmp_path, path)
    with pytest.raises(ValueError, match=r"sep\.csv: the target column 'target' has \d+ value\(s\) written with thousands separators, e\.g\. \['1,510\.0'"):
        _quiet(STAGES.stage_data, run)
    assert not (run.state / "train.parquet").exists() and list(out.iterdir()) == []


def test_M4_text_targets_are_refused_and_too_many_blanks_are_refused(tmp_path) -> None:
    """TIR-M4: present-but-unparseable targets are refused (not silently dropped); more than 20 % blank is refused."""
    frame = _diabetes().astype({"target": object})
    frame.loc[[1, 2, 3, 4], "target"] = ["abc", "12 units", "?", "inf"]
    frame.loc[[5, 6, 7, 8, 9], "target"] = None
    run = _byod_run(tmp_path, _csv(tmp_path, "text.csv", frame))
    with pytest.raises(ValueError, match=r"text\.csv: the target column 'target' has 4 value\(s\) that are not finite numbers, e\.g\. \['abc', '12 units', '\?', 'inf'\] \(file lines \[3, 4, 5, 6\]\)"):
        _quiet(STAGES.stage_data, run)
    sparse = _diabetes().astype({"target": object})
    sparse.loc[: len(sparse) // 4, "target"] = None
    run2 = _run(tmp_path / "run2", tmp_path / "outputs2", {**run.options, "paths": {"csv": str(_csv(tmp_path, "sparse.csv", sparse))}})
    with pytest.raises(ValueError, match=r"sparse\.csv: 111 of 442 rows have a blank target 'target', more than 20%"):
        _quiet(STAGES.stage_data, run2)


def test_M4_presplit_partitions_get_the_same_target_check(tmp_path) -> None:
    """TIR-M4: the pre-split branch refuses an unparseable target in train.csv too."""
    frame = _diabetes().astype({"target": object})
    frame.loc[4, "target"] = "n.a."
    train = _csv(tmp_path, "train.csv", frame.iloc[:350])
    val = _csv(tmp_path, "val.csv", frame.iloc[350:])
    run = _run(tmp_path / "run", tmp_path / "outputs", {"data_source": "Upload pre-split train/val/test", "use_byod": True, "target_column": "target", "paths": {"train": str(train), "val": str(val), "test": ""}})
    with pytest.raises(ValueError, match=r"train\.csv: the target column 'target' has 1 value\(s\) that are not finite numbers, e\.g\. \['n\.a\.'\]"):
        _quiet(STAGES.stage_data, run)


def test_m1_numeric_column_with_a_stray_string_is_refused_naming_column_and_value(tmp_path) -> None:
    """TIR-m1: `bmi` with one `abc` stops in Section 4 naming the column and the value; TEXT_COLUMNS opts out."""
    frame = _diabetes().astype({"bmi": object})
    frame.loc[5, "bmi"] = "abc"
    run = _byod_run(tmp_path, _csv(tmp_path, "typo.csv", frame))
    with pytest.raises(ValueError, match=r"typo\.csv: column 'bmi' is numeric except for 1 value\(s\) \['abc'\]"):
        _quiet(STAGES.stage_data, run)
    run.options["text_columns"] = ["bmi"]
    _quiet(STAGES.stage_data, run)
    assert json.loads((run.state / "data.json").read_text())["text_columns"] == ["bmi"]


# ---------------------------------------------------------------- TIR-m1: BYOD by path, TARGET_COLUMN field


def test_m2_target_column_field_and_missing_target_message(tmp_path) -> None:
    """TIR-m1: a CSV labelled `progression` stops naming `target` and the header; TARGET_COLUMN='progression' runs."""
    frame = _diabetes().rename(columns={"target": "progression"})
    path = _csv(tmp_path, "diag.csv", frame)
    run = _run(tmp_path / "run", tmp_path / "outputs", {"data_source": "Upload CSV", "use_byod": True, "target_column": "target", "paths": {"csv": str(path)}})
    with pytest.raises(ValueError, match=r"diag\.csv: the target column 'target' is not in the header \['age'.*'progression'\]\. Set TARGET_COLUMN"):
        _quiet(STAGES.stage_data, run)
    run.options["target_column"] = "progression"
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    assert json.loads((run.state / "data.json").read_text())["target"] == "progression"


def test_m2_presplit_byod_by_path_without_colab(tmp_path, monkeypatch) -> None:
    """TIR-m1: the pre-split path fields run outside Colab; an empty path outside Colab, or a cancelled upload, gets a clear message."""
    cell = _cell_with(E2E, "BYOD_TRAIN_PATH = ''  # @param")
    presplit = _set(_set(cell, "USE_BYOD", True), "DATA_SOURCE", "Upload pre-split train/val/test")
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(_set(_set(presplit, "BYOD_TRAIN_PATH", "/d/train.csv"), "BYOD_VAL_PATH", "/d/val.csv"), "<s4>", "exec"), ns)
    assert calls == [("data", {"data_source": "Upload pre-split train/val/test", "use_byod": True, "target_column": "target", "paths": {"train": "/d/train.csv", "val": "/d/val.csv", "test": ""}, "text_columns": []})]
    ns, calls = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match="the upload dialog exists only in Google Colab: set the BYOD path field"):
        exec(compile(presplit, "<s4>", "exec"), ns)
    monkeypatch.undo()
    single = _set(_set(cell, "USE_BYOD", True), "DATA_SOURCE", "Upload CSV")
    with _colab([{}]) as files:
        ns, calls = _kernel(tmp_path)
        with pytest.raises(RuntimeError, match="cancelled or empty"):
            exec(compile(single, "<s4>", "exec"), ns)
        assert files.calls == 1 and calls == []
    ns, calls = _kernel(tmp_path)
    exec(compile(cell, "<s4>", "exec"), ns)  # default path: no colab import, no upload
    assert calls == [("data", {"data_source": "Sample: Diabetes", "use_byod": False, "target_column": "target", "paths": {}, "text_columns": []})]


def test_m2_presplit_byod_runs_through_validate(tmp_path) -> None:
    """TIR-m1: pre-split train/val CSVs given by path run through the model-free stages."""
    frame = _diabetes()
    train = _csv(tmp_path, "train.csv", frame.iloc[:350])
    val = _csv(tmp_path, "val.csv", frame.iloc[350:])
    run = _run(tmp_path / "run", tmp_path / "outputs", {"data_source": "Upload pre-split train/val/test", "use_byod": True, "target_column": "target", "paths": {"train": str(train), "val": str(val), "test": ""}})
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    data = json.loads((run.state / "data.json").read_text())
    assert (data["sample_kind"], data["train_rows"], data["holdout_rows"], data["test_rows"]) == ("BYOD", 350, 92, 0)
    missing = _run(tmp_path / "run2", tmp_path / "outputs2", {**run.options, "paths": {"train": str(tmp_path / "nope.csv"), "val": str(val)}})
    with pytest.raises(FileNotFoundError, match="BYOD train file .*nope.csv.* does not exist"):
        _quiet(STAGES.stage_data, missing)


# ---------------------------------------------------------------- TIR-m2: stated facts


def test_m2_stated_facts_match_the_pins_and_hosts() -> None:
    """TIR-m2: the unrecorded development numbers and the Python 3.11 floor are gone; figshare is named for the optional
    sample; the model card's stricter evaluation-row floor is stated as a difference; the linear reference is quoted."""
    text = E2E.read_text(encoding="utf-8")
    assert "development run of the previous notebook revision" not in text and "Python 3.11" not in text
    assert "figshare" in text and "the production DIMER validator described in the model card is stricter" in text
    assert "MAE 38.22 / R² 0.581" in text


# ---------------------------------------------------------------- TIRA-M1: pinned sample bundle, default path


def test_M1_sample_bundle_is_reproduced_byte_for_byte() -> None:
    """TIRA-M1: tools/build_sample_bundle.py reproduces examples/sample-bundle/ exactly (no model is run)."""
    builder = _load("tir_build_sample_bundle", TOOLS / "build_sample_bundle.py")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert builder.main(["--check"]) == 0


def test_M1_sample_rows_are_the_e2e_new_rows(tmp_path) -> None:
    """TIRA-M1 / TIR-S2: the carried rows are the E2E notebook's first eight holdout rows, which the support set does not contain."""
    run = _run(tmp_path / "run", tmp_path / "outputs")
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    enc = run.read_state("encoded.json", "t")
    rows = STAGES.new_rows_frame(run, enc)
    pinned = pd.read_csv(SAMPLE / "new_rows.csv")
    pd.testing.assert_frame_equal(rows.reset_index(drop=True), pinned, check_dtype=False)
    assert not set(pinned["row_id"]) & set(enc["train_row_ids"])


def test_M1_companion_default_path_uses_the_sample_without_upload(tmp_path, monkeypatch) -> None:
    """TIRA-M1: all fields at their defaults run the pinned sample bundle and rows; google.colab is never imported."""
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(_cell_with(AI, "ARTIFACT_ZIP_PATH = ''  # @param"), "<s4>", "exec"), ns)
    exec(compile(_cell_with(AI, "NEW_DATA_PATH = ''  # @param"), "<s6>", "exec"), ns)
    assert calls == [("artifact", {"source": "sample", "zip_path": "", "expected_zip_sha256": ""}), ("rows", {"source": "sample", "path": ""})]


def _ai_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    shutil.copytree(SAMPLE, root / "sample-bundle")
    return root


def test_M1_sample_bundle_assembles_with_the_verified_base_checkpoint(tmp_path) -> None:
    """TIRA-M1: the sample's artifact.json is checked against its pinned digest, and the base checkpoint becomes best.ckpt."""
    root = _ai_root(tmp_path)
    run = AI_STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "sample"})
    base = tmp_path / "weights" / "base.ckpt"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_bytes(b"verified base checkpoint stand-in")
    run.write_state("weights.json", {"checkpoint": str(base)})
    P = AI_STAGES.package(root)
    bundle = tmp_path / "bundle"
    trust = AI_STAGES.assemble_sample(run, P, bundle)
    record = json.loads((SAMPLE / "SAMPLE_BUNDLE.json").read_text())
    assert trust["artifact_json_sha256"] == record["artifact_sha256"]
    assert (bundle / "checkpoints" / "best.ckpt").read_bytes() == base.read_bytes()
    assert (bundle / "training_context.parquet").read_bytes() == (SAMPLE / "training_context.parquet").read_bytes()
    manifest = json.loads((bundle / "artifact.json").read_text())
    assert AI_STAGES.check_bundle_identity(P, manifest)["mode"] == "pretrained"
    (root / "sample-bundle" / "artifact.json").write_text("{}")
    with pytest.raises(ValueError, match="Trusted digest mismatch for artifact.json"):
        AI_STAGES.assemble_sample(run, P, tmp_path / "bundle2")


# ---------------------------------------------------------------- TIRA-m1: ZIP by path + rows by upload


def test_m1_zip_by_path_with_rows_by_upload_has_no_name_error(tmp_path, monkeypatch) -> None:
    """TIRA-m1: ARTIFACT_ZIP_PATH set, NEW_DATA_PATH empty: outside Colab a message naming NEW_DATA_PATH (no NameError);
    in Colab the dialog opens."""
    artifact_cell = _set(_cell_with(AI, "ARTIFACT_ZIP_PATH = ''  # @param"), "ARTIFACT_ZIP_PATH", "/d/bundle.zip")
    rows_cell = _cell_with(AI, "NEW_DATA_PATH = ''  # @param")
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(artifact_cell, "<s4>", "exec"), ns)
    with pytest.raises(RuntimeError, match="set NEW_DATA_PATH"):
        exec(compile(rows_cell, "<s6>", "exec"), ns)
    monkeypatch.undo()
    with _colab([{"rows.csv": b"a\n1\n"}]) as files:
        ns, calls = _kernel(tmp_path)
        exec(compile(artifact_cell, "<s4>", "exec"), ns)
        exec(compile(rows_cell, "<s6>", "exec"), ns)
        assert files.calls == 1 and calls[1][0] == "rows" and calls[1][1]["source"] == "upload"
        assert calls[1][1]["path"].endswith("inputs/rows.csv")


# ---------------------------------------------------------------- TIRA-M1 / TIRA-m2: trusted digest, checkpoint binding


def test_M1_wrong_trusted_zip_digest_is_refused_before_extraction(tmp_path, monkeypatch) -> None:
    """TIRA-M1 (trust): a ZIP whose SHA-256 differs from EXPECTED_ZIP_SHA256 is refused before anything is extracted."""
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.14.0", cuda=types.SimpleNamespace(is_available=lambda: False)))
    root = _ai_root(tmp_path)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(SAMPLE / "artifact.json", "artifact.json")
    run = AI_STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "path", "zip_path": str(archive), "expected_zip_sha256": "0" * 64})
    observed = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=f"EXPECTED_ZIP_SHA256 is {'0' * 64}, the supplied file's SHA-256 is {observed}.*nothing was extracted"):
        _quiet(AI_STAGES.stage_artifact, run)
    assert not (root / "state" / "bundle").exists()
    run.options["expected_zip_sha256"] = "abc"
    with pytest.raises(ValueError, match="must be 64 hexadecimal characters"):
        _quiet(AI_STAGES.stage_artifact, run)


def test_m2_pretrained_bundle_must_carry_the_base_checkpoint(tmp_path) -> None:
    """TIRA-m2: a `pretrained` bundle whose checkpoint digest is not BASE_MODEL_SHA256 is refused naming the expected digest."""
    P = AI_STAGES.package(_ai_root(tmp_path))
    manifest = json.loads((SAMPLE / "artifact.json").read_text())
    swapped = json.loads(json.dumps(manifest))
    swapped["digests"]["checkpointSha256"] = "f" * 64
    with pytest.raises(RuntimeError, match=f"must be the pinned base checkpoint with SHA-256 {P.BASE_MODEL_SHA256}"):
        AI_STAGES.check_bundle_identity(P, swapped)
    swapped["mode"] = "fine-tuned"
    assert "EXPECTED_ZIP_SHA256" in AI_STAGES.check_bundle_identity(P, swapped)["checkpoint_binding"]
    other_base = json.loads(json.dumps(manifest))
    other_base["baseModelSha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="not produced on the pinned base checkpoint"):
        AI_STAGES.check_bundle_identity(P, other_base)


# ---------------------------------------------------------------- TIRA-m3: numeric features, sample_kind


def _rows_run(tmp_path: Path, art_source: str, options: dict):
    root = _ai_root(tmp_path)
    run = AI_STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", options)
    manifest = json.loads((SAMPLE / "artifact.json").read_text())
    run.write_state("artifact.json", {"source": art_source, "manifest": manifest})
    return run, root


def test_m3_non_numeric_value_in_a_numeric_feature_is_refused_in_the_rows_stage(tmp_path) -> None:
    """TIRA-m3: `abc` in `age` stops in the rows stage naming the column and the value."""
    rows = pd.read_csv(SAMPLE / "new_rows.csv").astype({"age": object})
    rows.loc[2, "age"] = "abc"
    path = tmp_path / "mine.csv"
    rows.to_csv(path, index=False)
    run, root = _rows_run(tmp_path, "path", {"source": "path", "path": str(path)})
    with pytest.raises(ValueError, match=r"mine\.csv: numeric feature 'age' has 1 non-numeric value\(s\), e\.g\. \['abc'\]"):
        _quiet(AI_STAGES.stage_rows, run)
    assert not (root / "state" / "rows.parquet").exists()


def test_m3_sample_kind_follows_the_input_source(tmp_path) -> None:
    """TIRA-m3: sample rows on the sample bundle are `sample`; the user's rows are `BYOD`; sample rows on a user bundle are refused."""
    run, root = _rows_run(tmp_path, "sample", {"source": "sample"})
    _quiet(AI_STAGES.stage_rows, run)
    state = json.loads((root / "state" / "rows.json").read_text())
    assert (state["sample_kind"], state["rows"], state["extra_columns"]) == ("sample", 8, ["row_id"])
    path = tmp_path / "mine.csv"
    pd.read_csv(SAMPLE / "new_rows.csv").to_csv(path, index=False)
    run.options = {"source": "path", "path": str(path)}
    _quiet(AI_STAGES.stage_rows, run)
    assert json.loads((root / "state" / "rows.json").read_text())["sample_kind"] == "BYOD"
    run.write_state("artifact.json", {"source": "path", "manifest": json.loads((SAMPLE / "artifact.json").read_text())})
    run.options = {"source": "sample"}
    with pytest.raises(ValueError, match="The pinned sample rows match only the pinned sample bundle"):
        _quiet(AI_STAGES.stage_rows, run)
