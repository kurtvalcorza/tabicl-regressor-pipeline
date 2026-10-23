"""Refused user input is shown as one clean line (NOTEBOOK_SPEC 3.0 DAT19/DAT20).

Each refusal path runs the generated notebook's own kernel code (the ``run_stage`` helper of the Section 2 cell and the
learner cell) against the real stage runner in a child process, as the notebook does, with only ``google.colab``'s
upload dialog stood in. For every path the cell must raise ``Refused`` (an exception, so Run all stops at that cell),
render as exactly one ``Refused: <file>: <reason>`` line through IPython's ``_render_traceback_`` hook, name the file
the learner supplied, and print no ``Traceback`` from the stage. A genuine failure still raises ``RuntimeError`` with
the stage's traceback. The Section 4 case is the browser kit's ``refuse.csv``: the unscaled diabetes table whose target
``progression`` holds three text values (``about 150``, ``high``, ``?``). CI's dependencies suffice (no torch, no
tabicl); the last test also runs a real IPython kernel when ``ipykernel`` is installed.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tutorials" / "tabiclv2_regressor_colab.ipynb"
STEM = "tabiclv2_regressor"
TEXT_ROWS = {3: "about 150", 17: "high", 40: "?"}  # data rows of the kit's refuse.csv (CSV lines 5, 19 and 42)


def _cells() -> list[str]:
    notebook = json.loads(E2E.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(needle: str) -> str:
    found = [c for c in _cells() if needle in c]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


def _kernel(tmp_path: Path) -> dict:
    """The notebook's own stage helper, bound to a run directory that carries the package and the stage runner."""
    run = tmp_path / "run"
    (run / "state").mkdir(parents=True)
    shutil.copytree(ROOT / "src", run / "src")
    shutil.copyfile(ROOT / "tools" / "tutorial_stages.py", run / "tutorial_stages.py")
    install = _cell("def run_stage(stage, **options):")
    helper = install[install.index("REFUSED_EXIT = 3"):]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    ns = {"ROOT": run, "OUTPUTS": tmp_path / "outputs", "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": env, "Path": Path, "json": json, "subprocess": subprocess, "shutil": shutil}
    exec(compile(helper, "<run_stage>", "exec"), ns)
    return ns


@contextlib.contextmanager
def _colab(*uploads: dict):
    queue = list(uploads)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: queue.pop(0))
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _run_all(ns: dict, cells: list[str]) -> tuple[BaseException | None, str, int]:
    """Run cells in order as Run all does: the first cell that raises stops the run. Returns the exception, the cell
    output (stdout plus the exception as IPython renders it) and how many cells ran."""
    out, ran, raised = io.StringIO(), 0, None
    with contextlib.redirect_stdout(out):
        for source in cells:
            ran += 1
            try:
                exec(compile(source, f"<cell {ran}>", "exec"), ns)
            except BaseException as exc:  # noqa: BLE001 -- what IPython would show is asserted below
                raised = exc
                break
    shown = out.getvalue()
    if raised is not None and hasattr(raised, "_render_traceback_"):
        shown += "\n".join(raised._render_traceback_()) + "\n"
    return raised, shown, ran


def _assert_refused(raised, shown: str, ran: int, file: str) -> str:
    assert type(raised).__name__ == "Refused", (raised, shown)
    lines = raised._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith(f"Refused: {file}: "), lines
    assert [line for line in shown.splitlines() if "Refused:" in line] == lines, shown
    assert "Traceback" not in shown, shown
    assert ran == 1, "Run all must stop at the refusing cell"
    return lines[0]


def _diabetes(path: Path, text_targets: bool = False) -> bytes:
    from sklearn.datasets import load_diabetes

    frame = load_diabetes(as_frame=True, scaled=False).frame.rename(columns={"target": "progression"})
    if text_targets:
        frame["progression"] = frame["progression"].astype(object)
        for row, value in TEXT_ROWS.items():
            frame.loc[row, "progression"] = value
    frame.to_csv(path, index=False)
    return path.read_bytes()


def _section4(**fields) -> str:
    cell = _cell("DATA_SOURCE = 'Sample: Diabetes'  # @param")
    for name, value in {"DATA_SOURCE": "Upload CSV", "USE_BYOD": True, "TARGET_COLUMN": "progression", **fields}.items():
        cell = _set(cell, name, value)
    return cell


def _accepted(tmp_path: Path, ns: dict) -> None:
    with _colab({"accept.csv": _diabetes(tmp_path / "accept.csv")}):
        raised, shown, _ = _run_all(ns, [_section4(), "run_stage('validate')"])
    assert raised is None, shown


# ---------------------------------------------------------------- Section 4: the learner's table


def test_browser_kit_refuse_csv_is_one_clean_line_naming_the_upload(tmp_path) -> None:
    """The kit's Pass B: refuse.csv uploaded with TARGET_COLUMN = 'progression' (three of its targets are text)."""
    ns = _kernel(tmp_path)
    stale = ns["OUTPUTS"] / f"{STEM}_input_manifest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    with _colab({"refuse.csv": _diabetes(tmp_path / "refuse.csv", text_targets=True)}):
        raised, shown, ran = _run_all(ns, [_section4(), "run_stage('validate')"])
    line = _assert_refused(raised, shown, ran, "refuse.csv")
    assert "the target column 'progression' has 3 value(s) that are not finite numbers" in line and "(file lines [5, 19, 42])" in line
    assert "'about 150'" in line and "run the Section 4 cell again" in line
    assert not stale.exists() and not (ns["ROOT"] / "state" / "data.json").exists()  # no earlier result survives
    error = json.loads((ns["ROOT"] / "state" / "data.error.json").read_text(encoding="utf-8"))
    assert error["refused"] is True and error["file"] == "refuse.csv"


@pytest.mark.parametrize(
    ("fields", "upload", "file", "needle"),
    [
        ({"TARGET_COLUMN": "target"}, "ok", "refuse.csv", "the target column 'target' is not in the header"),
        ({}, "stray", "refuse.csv", "column 'bmi' is numeric except for 1 value(s) ['abc']"),
        ({}, "separators", "refuse.csv", "written with thousands separators"),
        ({"TEXT_COLUMNS": ["site"]}, "ok", "refuse.csv", "TEXT_COLUMNS ['site'] are not in the header"),
        ({}, "text", "refuse.txt", "BYOD files must be CSV"),
        ({"BYOD_CSV_PATH": "/nowhere/refuse.csv"}, None, "/nowhere/refuse.csv", "does not exist in this runtime"),
        ({"DATA_SOURCE": "Sample: Diabetes"}, None, "DATA_SOURCE", "USE_BYOD=True requires DATA_SOURCE"),
    ],
    ids=["target-missing", "stray-string", "thousands-separators", "text-columns-missing", "wrong-format", "missing-path", "source-mismatch"],
)
def test_section4_refusals_are_one_clean_line(tmp_path, fields, upload, file, needle) -> None:
    ns = _kernel(tmp_path)
    good = _diabetes(tmp_path / "good.csv")
    lines = good.decode().splitlines()
    if upload == "stray":
        cells = lines[5].split(",")
        cells[2] = "abc"
        lines[5] = ",".join(cells)
        uploads = [{"refuse.csv": ("\n".join(lines) + "\n").encode()}]
    elif upload == "separators":
        lines[5] = lines[5].rsplit(",", 1)[0] + ',"1,510.0"'
        uploads = [{"refuse.csv": ("\n".join(lines) + "\n").encode()}]
    elif upload == "text":
        uploads = [{"refuse.txt": good}]
    elif upload == "ok":
        uploads = [{"refuse.csv": good}]
    else:
        uploads = []
    with _colab(*uploads):
        raised, shown, ran = _run_all(ns, [_section4(**fields), "run_stage('validate')"])
    line = _assert_refused(raised, shown, ran, file)
    assert needle in line, line


def test_section4_two_uploaded_files_are_refused_naming_both(tmp_path) -> None:
    ns = _kernel(tmp_path)
    with _colab({"a.csv": b"x\n1\n", "b.csv": b"x\n1\n"}):
        raised, shown, ran = _run_all(ns, [_section4(), "run_stage('validate')"])
    line = _assert_refused(raised, shown, ran, "a.csv, b.csv")
    assert "upload exactly one CSV" in line


# ---------------------------------------------------------------- Section 8: the learner's own rows


@pytest.mark.parametrize(
    ("rows", "needle"),
    [
        ("age,sex,bmi\n59.0,2.0,32.1\n", "bp"),
        ("", "own_rows.csv"),
    ],
    ids=["schema", "empty-file"],
)
def test_section8_own_rows_refusals_are_one_clean_line(tmp_path, rows, needle) -> None:
    ns = _kernel(tmp_path)
    _accepted(tmp_path, ns)
    stale = ns["OUTPUTS"] / f"{STEM}_predictions.csv"
    stale.write_text("old", encoding="utf-8")
    cell = _set(_cell("RUN_NEW_DATA_INFERENCE = False  # @param"), "RUN_NEW_DATA_INFERENCE", True)
    with _colab({"own_rows.csv": rows.encode()}):
        raised, shown, ran = _run_all(ns, [cell, "run_stage('export')"])
    line = _assert_refused(raised, shown, ran, "own_rows.csv")
    assert needle in line and "Section 8 cell again" in line
    assert not stale.exists()  # DAT20: no earlier prediction left on disk to be read as this one


STUB_MODEL = """

# ---- test stub (tests/test_refusal_display.py): a stand-in for the TabICL model, so Section 8 runs on CI's dependencies.
import sys as _sys
import types as _types
from pathlib import Path as _Path

try:
    import torch as _torch  # noqa: F401
except ImportError:  # CI has no torch: the stage's runtime record only reads its version
    _torch = _types.ModuleType("torch")
    _torch.__version__ = "stub"
    _torch.Tensor = type("Tensor", (), {})  # scipy probes torch.Tensor when torch is in sys.modules
    _sys.modules["torch"] = _torch
_LOADED = _Path(__file__).resolve().parents[2] / "model_loaded"


class TabICLRegressionPipeline:
    device = "cpu"

    @classmethod
    def from_pretrained(cls, **kwargs):
        _LOADED.write_text("the model was loaded", encoding="utf-8")
        return cls()

    def fit(self, X, y):
        return self

    def predict(self, X):
        import numpy as _np

        return _np.asarray(X, dtype=float).sum(axis=1)  # raises on a string: only numbers may reach the model
"""


def _stub_model(ns: dict) -> Path:
    """Replace the model in the run's copy of the package and record a conditioned pretrained model, so the real
    ``predict`` stage can score without torch or the checkpoint. Returns the file the stub writes when it is loaded."""
    run = ns["ROOT"]
    init = run / "src" / "tabicl_regressor_pipeline" / "__init__.py"
    init.write_text(init.read_text(encoding="utf-8") + STUB_MODEL, encoding="utf-8")
    dist = run / "src" / "tabicl-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: tabicl\nVersion: 0.0.0\n", encoding="utf-8")
    holdout = {"mae": 1.0, "mse": 1.0, "rmse": 1.0, "r2": 0.0, "pearsonr": 0.0}
    condition = {"active_mode": "pretrained", "active_checkpoint": None, "selection": "pretrained", "eval_metric": "mse", "fine_tune": {"ran": False}, "pretrained_holdout": holdout, "pretrained_test": None, "candidate_holdout": None, "candidate_test": None, "device": "cpu"}
    (run / "state" / "condition.json").write_text(json.dumps(condition), encoding="utf-8")
    (ns["OUTPUTS"] / f"{STEM}_evaluation_report.json").write_text("{}", encoding="utf-8")
    return run / "model_loaded"


def _own_rows(tmp_path: Path, stray: dict[tuple[int, str], str]) -> bytes:
    """Six unlabelled diabetes rows; ``stray`` puts text into (data row, column) cells (data row 0 is CSV line 2)."""
    from sklearn.datasets import load_diabetes

    frame = load_diabetes(as_frame=True, scaled=False).frame.drop(columns="target").head(6).astype(object)
    for (row, column), value in stray.items():
        frame.loc[row, column] = value
    path = tmp_path / "own_rows.csv"
    frame.to_csv(path, index=False)
    return path.read_bytes()


def _section8(ns: dict, payload: bytes, then: tuple[str, ...] = ("run_stage('export')",)):
    cell = _set(_cell("RUN_NEW_DATA_INFERENCE = False  # @param"), "RUN_NEW_DATA_INFERENCE", True)
    with _colab({"own_rows.csv": payload}):
        return _run_all(ns, [cell, *then])


@pytest.mark.parametrize(
    ("stray", "needles"),
    [
        ({(2, "bmi"): "abc"}, ["numeric feature 'bmi' has 1 non-numeric value(s), e.g. ['abc'] (file lines [4])"]),
        (
            {(1, "bp"): "high", (4, "bp"): "12 mmHg", (3, "s5"): "?"},
            ["numeric feature 'bp' has 2 non-numeric value(s), e.g. ['high', '12 mmHg'] (file lines [3, 6])", "numeric feature 's5' has 1 non-numeric value(s), e.g. ['?'] (file lines [5])"],
        ),
    ],
    ids=["one-column", "two-columns"],
)
def test_section8_stray_text_in_own_rows_is_refused_before_scoring(tmp_path, stray, needles) -> None:
    """A numeric feature holding text in your own Section 8 rows is refused in one line naming the file, the columns,
    the values and their CSV lines; the model is never loaded, nothing is scored or written, and Run all stops."""
    ns = _kernel(tmp_path)
    _accepted(tmp_path, ns)
    loaded = _stub_model(ns)
    stale = [ns["OUTPUTS"] / f"{STEM}_predictions.csv", ns["OUTPUTS"] / f"{STEM}_result.json"]
    for path in stale:
        path.write_text("old", encoding="utf-8")
    raised, shown, ran = _section8(ns, _own_rows(tmp_path, stray))
    line = _assert_refused(raised, shown, ran, "own_rows.csv")
    assert all(needle in line for needle in needles), line
    assert "Section 8 cell again" in line
    assert not loaded.exists(), "the model must not be loaded for refused rows"
    assert not any(path.exists() for path in stale)  # nothing scored, no earlier result left on disk (DAT20)
    error = json.loads((ns["ROOT"] / "state" / "predict.error.json").read_text(encoding="utf-8"))
    assert error["refused"] is True and error["file"] == "own_rows.csv"


def test_section8_clean_own_rows_still_score(tmp_path) -> None:
    """The same file without stray text passes the check and is scored by the real predict stage."""
    ns = _kernel(tmp_path)
    _accepted(tmp_path, ns)
    loaded = _stub_model(ns)
    raised, shown, ran = _section8(ns, _own_rows(tmp_path, {}), then=())
    assert raised is None and ran == 1, shown
    assert "Refused" not in shown and "Traceback" not in shown
    assert loaded.exists()
    import pandas as pd

    predictions = pd.read_csv(ns["OUTPUTS"] / f"{STEM}_predictions.csv")
    assert len(predictions) == 6 and predictions["prediction"].notna().all()
    result = json.loads((ns["OUTPUTS"] / f"{STEM}_result.json").read_text(encoding="utf-8"))
    assert result["new_data"]["input"] == "own_rows.csv" and result["new_data"]["rows"] == 6


# ---------------------------------------------------------------- Section 10: your own artifact ZIP and rows


def _external_upload(ns: dict) -> None:
    cell = _cell("ARTIFACT_SOURCE = 'Pinned sample artifact'")
    exec(compile(cell[cell.index("def external_upload"):cell.index("if ARTIFACT_SOURCE ==")], "<external_upload>", "exec"), ns)


def test_section10_own_artifact_zip_is_refused(tmp_path) -> None:
    import zipfile

    ns = _kernel(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "not an artifact")
    cell = _set(_cell("ARTIFACT_SOURCE = 'Pinned sample artifact'"), "ARTIFACT_SOURCE", "My own artifact ZIP")
    with _colab({"my_artifact.zip": buffer.getvalue()}):
        raised, shown, ran = _run_all(ns, [cell, "run_stage('external_reconstruct')"])
    line = _assert_refused(raised, shown, ran, "my_artifact.zip")
    assert "artifact.json" in line and "run this cell again" in line


def test_section10_wrong_digest_for_own_artifact_is_refused(tmp_path) -> None:
    ns = _kernel(tmp_path)
    cell = _set(_set(_cell("ARTIFACT_SOURCE = 'Pinned sample artifact'"), "ARTIFACT_SOURCE", "My own artifact ZIP"), "EXPECTED_ZIP_SHA256", "0" * 64)
    with _colab({"my_artifact.zip": b"PK\x05\x06" + b"\x00" * 18}):
        raised, shown, ran = _run_all(ns, [cell])
    assert "Trusted digest mismatch" in _assert_refused(raised, shown, ran, "my_artifact.zip")


@pytest.mark.parametrize(("rows", "needle"), [(b"a,c\n1,2\n", "b"), (b"a,b\n1,abc\n2,3\n", "numeric feature 'b' has 1 non-numeric value(s)")], ids=["schema", "stray-string"])
def test_section10_own_rows_are_refused_before_the_artifact_is_loaded(tmp_path, rows, needle) -> None:
    ns = _kernel(tmp_path)
    art = {"source": "own", "zip": "mine.zip", "manifest": {"featureColumns": ["a", "b"], "targetColumn": "target", "inference": {"categoricalEncoders": {}}}}
    (ns["ROOT"] / "state" / "external_artifact.json").write_text(json.dumps(art), encoding="utf-8")
    stale = ns["OUTPUTS"] / "external_artifact" / f"{STEM}_external_predictions.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")
    _external_upload(ns)
    cell = _set(_cell("USE_OWN_ROWS = False  # @param"), "USE_OWN_ROWS", True)
    with _colab({"own_rows.csv": rows}):
        raised, shown, ran = _run_all(ns, [cell])
    assert needle in _assert_refused(raised, shown, ran, "own_rows.csv")
    assert not stale.exists()  # DAT20: no earlier prediction left on disk to be read as this one


# ---------------------------------------------------------------- genuine failures keep their traceback


def test_a_genuine_failure_is_not_a_refusal(tmp_path) -> None:
    """A stage that fails for a reason other than the learner's input still raises RuntimeError with its traceback."""
    ns = _kernel(tmp_path)
    raised, shown, _ = _run_all(ns, ["run_stage('validate')"])
    assert type(raised) is RuntimeError and "failed (exit 1)" in str(raised)
    assert "Traceback" in shown and "Refused:" not in shown


def test_stage_runner_exit_codes(tmp_path) -> None:
    """The child process: a refusal exits REFUSED_EXIT (3) with no traceback on its streams; a crash exits 1."""
    ns = _kernel(tmp_path)
    run = ns["ROOT"]
    csv = tmp_path / "refuse.csv"
    _diabetes(csv, text_targets=True)
    base = [sys.executable, str(run / "tutorial_stages.py"), "--root", str(run), "--outputs", str(tmp_path / "outputs"), "--weights", str(tmp_path / "weights")]
    options = {"data_source": "Upload CSV", "use_byod": True, "target_column": "progression", "paths": {"csv": str(csv)}}
    refused = subprocess.run([*base, "--stage", "data", "--options", json.dumps(options)], capture_output=True, text=True, env=ns["ENV"])
    assert refused.returncode == 3 and "Traceback" not in refused.stdout + refused.stderr
    crashed = subprocess.run([*base, "--stage", "validate"], capture_output=True, text=True, env=ns["ENV"])
    assert crashed.returncode == 1 and "Traceback" in crashed.stderr


# ---------------------------------------------------------------- the real kernel (when ipykernel is installed)


def test_real_kernel_shows_one_line_and_stops_run_all(tmp_path) -> None:
    pytest.importorskip("ipykernel")
    nbclient = pytest.importorskip("nbclient")
    import nbformat

    ns = _kernel(tmp_path)
    payload = _diabetes(tmp_path / "refuse.csv", text_targets=True)
    setup = (
        "import json, os, shutil, subprocess, sys, types\nfrom pathlib import Path\n"
        f"ROOT = Path({str(ns['ROOT'])!r}); OUTPUTS = Path({str(ns['OUTPUTS'])!r}); WEIGHTS = Path({str(ns['WEIGHTS'])!r})\n"
        "PYTHON = Path(sys.executable)\nENV = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP')}\n"
        f"colab = types.ModuleType('google.colab'); colab.files = types.SimpleNamespace(upload=lambda: {{'refuse.csv': {payload!r}}})\n"
        "google = types.ModuleType('google'); google.colab = colab; sys.modules.update({'google': google, 'google.colab': colab})\n"
    )
    install = _cell("def run_stage(stage, **options):")
    cells = [setup, install[install.index("REFUSED_EXIT = 3"):], _section4(), "print('AFTER THE REFUSAL')"]
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(c) for c in cells])
    client = nbclient.NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}})
    with pytest.raises(nbclient.exceptions.CellExecutionError):
        client.execute()
    refused = notebook.cells[2]
    errors = [o for o in refused.outputs if o.output_type == "error"]
    text = "\n".join(o.get("text", "") for o in refused.outputs if o.output_type == "stream") + "\n".join("\n".join(o.traceback) for o in errors)
    assert len(errors) == 1 and errors[0].ename == "Refused"
    assert len(errors[0].traceback) == 1 and errors[0].traceback[0].startswith("Refused: refuse.csv: the target column 'progression' has 3 value(s)")
    assert text.count("Refused:") == 1 and "Traceback" not in text
    assert notebook.cells[3].outputs == [] and notebook.cells[3].execution_count is None  # Run all stopped
