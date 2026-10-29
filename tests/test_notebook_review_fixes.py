"""Regression tests for the 2026-10-05 notebook review (TIR-M1..M4, TIR-m1..m2; TIRA-M1..M3, TIRA-m1..m3).

They need only CI's dependencies (no torch, no tabicl, no lightgbm, no checkpoint): they exec the notebooks' own kernel
cells with stand-ins for `run_stage` and `google.colab`, run the stage runners' model-free parts (`data`, `validate`,
and check the generated notebook statically. The artifact-inference companion they also covered (TIRA-*) was folded
into the E2E notebook's external-artifact section under NOTEBOOK_SPEC 3.0; its checks now live in
`tests/test_external_artifact_section.py`. Each test names its finding.
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
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabiclv2_regressor_colab.ipynb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("tir_tutorial_stages", TOOLS / "tutorial_stages.py")


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
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o)), "Refused": type("Refused", (Exception,), {})}
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


@pytest.mark.parametrize("path", [E2E], ids=["TIR-M1"])
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


@pytest.mark.parametrize("path", [E2E], ids=["TIR-M1"])
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


@pytest.mark.parametrize("path", [E2E], ids=["TIR-M3"])
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


def test_TIRA_M3_every_external_artifact_refusal_has_a_troubleshooting_row() -> None:
    """TIRA-M3 (carried into the E2E external-artifact section): the troubleshooting table names its refusals."""
    md = "\n".join(_src(c) for c in _nb(E2E)["cells"] if c["cell_type"] == "markdown")
    table = md[md.index("## Troubleshooting"):]
    for message in ("Trusted digest mismatch", "default data contract (SART10)", "external_artifact.json is missing", "is numeric except for"):
        assert message in table, message


@pytest.mark.parametrize("runner", ["tutorial_stages.py"])
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
        with pytest.raises(ns["Refused"], match=r"^\(no file\): the upload was cancelled or empty"):  # DAT20: a refusal, not a RuntimeError
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
    with pytest.raises(ValueError, match="nope.csv: the BYOD train file does not exist"):
        _quiet(STAGES.stage_data, missing)


# ---------------------------------------------------------------- TIR-m2: stated facts


def test_m2_stated_facts_match_the_pins_and_hosts() -> None:
    """TIR-m2: the unrecorded development numbers and the Python 3.11 floor are gone; figshare is named for the optional
    sample; the model card's stricter evaluation-row floor is stated as a difference; the linear reference is quoted."""
    text = E2E.read_text(encoding="utf-8")
    assert "development run of the previous notebook revision" not in text and "Python 3.11" not in text
    assert "figshare" in text and "the production DIMER validator described in the model card is stricter" in text
    assert "MAE 38.22 / R² 0.581" in text


# ---------------------------------------------------------------- ENV15: no worker, no google.colab stub, no IPython in stages


@pytest.mark.parametrize("template", ["notebook_template.py"])
def test_stage_processes_import_neither_ipython_nor_google_colab(template: str) -> None:
    """Stages run in the isolated environment, which has neither IPython nor google.colab: only kernel cells may use them
    (the upload dialogs). There is no worker and no google.colab stub that would need a ModuleSpec (ENV15). The carried
    package imports google.colab only inside `read_single_input` / `download_output`, which no stage calls; a module-level
    import, or a stage calling either helper, would fail on Colab (tabpfn-regressor-pipeline e2d5fab pattern)."""
    build = _load(f"tir_build_notebook_{template[:-3]}", TOOLS / "build_notebook.py")
    tpl = _load(f"tir_{template[:-3]}", TOOLS / template).TEMPLATE
    carried = [ROOT / source for dest, source in build.carried_sources(ROOT, tpl).items() if dest.endswith(".py")]
    runner = ROOT / tpl["stage_runner"]
    assert runner in carried and any(p.name == "api.py" for p in carried)
    offenders = [str(p) for p in carried if re.search(r"^(from|import)\s+(IPython|google)\b", p.read_text(encoding="utf-8"), re.M)]
    assert not offenders, offenders
    for p in carried:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                if any(n.split(".")[0] in ("IPython", "google") for n in names):
                    owner = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.lineno <= node.lineno <= f.end_lineno]
                    assert owner and owner[-1].name in {"read_single_input", "download_output"}, (p.name, node.lineno)
    runner_text = runner.read_text(encoding="utf-8")
    assert "read_single_input" not in runner_text and "download_output" not in runner_text
    text = (ROOT / "tutorials" / tpl["notebook_name"]).read_text(encoding="utf-8")
    assert "sys.modules['google" not in text and 'sys.modules[\\"google' not in text and "_WORKER_SOURCE" not in text
    assert "IPython" not in text
