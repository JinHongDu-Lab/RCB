"""Assert the properties the directory layout is supposed to have.

A layout is a convention until something checks it, and conventions erode the
first time someone is in a hurry.  These tests make the arrangement a property
of the repository:

* results live in study directories and figures have study prefixes, so two
  studies cannot overwrite each other's ``configuration.csv``;
* :class:`rcb.paths.StudyDir` cannot silently lose its prefix;
* notebooks read results and plot -- they never run an experiment.

The last one is the whole point of the layout.  Experiments belong in
``experiments/`` so they can run headlessly, and a notebook that quietly
recomputes a number defeats that the first time someone is in a hurry.

Run with::

    python -m pytest code/tests/test_layout.py -q
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from rcb import paths  # noqa: E402

GENERATED_TREES = ("results", "figures")


def _generated_files(tree: str) -> list[Path]:
    return sorted(
        p for p in (CODE / tree).rglob("*")
        if p.is_file() and p.name != ".DS_Store"
    )


@pytest.mark.parametrize("tree", GENERATED_TREES)
def test_every_generated_file_names_its_study(tree: str) -> None:
    unprefixed = [
        p.name for p in _generated_files(tree)
        if (not any(p.name.startswith(f"{s}_") for s in paths.STUDIES)
            if tree == "figures" else p.relative_to(CODE / tree).parts[0] not in paths.STUDIES)
    ]
    assert not unprefixed, (
        f"{len(unprefixed)} file(s) in {tree}/ do not name a study: "
        f"{unprefixed[:10]}"
    )


@pytest.mark.parametrize("tree", ("figures",))
def test_generated_trees_are_flat(tree: str) -> None:
    nested = [
        str(p.relative_to(CODE)) for p in (CODE / tree).iterdir() if p.is_dir()
    ]
    assert not nested, f"{tree}/ should be flat, found subdirectories: {nested}"


def test_study_dir_uses_subdirectory_and_figures_keep_prefix() -> None:
    directory = paths.results("simulation", create=False)
    assert directory / "tuning_summary.csv" == CODE / "results/simulation/tuning_summary.csv"
    assert paths.figures("simulation", create=False) / "risk.pdf" == CODE / "figures/simulation_risk.pdf"
    assert paths.data("lalonde") / "nsw_dw.dta" == CODE / "data/lalonde/nsw_dw.dta"


def test_study_dir_refuses_to_lose_its_prefix() -> None:
    """Coercion would discard lazy redirection; it must raise."""
    directory = paths.results("simulation", create=False)
    assert not isinstance(directory, os.PathLike)
    with pytest.raises(TypeError):
        Path(directory)  # type: ignore[arg-type]


def test_study_dir_refuses_nested_names() -> None:
    directory = paths.results("lalonde", create=False)
    with pytest.raises(ValueError):
        directory / "support/cv_fold_stability_draws.csv"


def test_unknown_study_is_rejected() -> None:
    with pytest.raises(KeyError):
        paths.results("section5")


def test_output_root_override_redirects_every_tree(tmp_path: Path) -> None:
    """Redirection must reach constants captured at import time, too.

    Modules routinely do ``OUT = paths.results("lalonde")`` at module level; if
    that froze the root, every test would have to import inside the context.
    """
    frozen = paths.results("lalonde", create=False)
    with paths.use_output_root(tmp_path):
        assert (frozen / "x.csv").is_relative_to(tmp_path)
        assert (paths.results("evaluation", create=False) / "x.csv").is_relative_to(
            tmp_path
        )
        assert paths.data("lalonde").root.is_relative_to(tmp_path)
    assert not (paths.results("evaluation", create=False) / "x.csv").is_relative_to(
        tmp_path
    )


def test_environment_variable_redirects_every_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_experiments.sh --smoke`` redirects through the environment.

    A smoke run writes eight-replication numbers.  If it could reach
    ``results/`` it would overwrite five-hundred-replication numbers with them
    and leave nothing on disk to say so, so the redirect has to work from a
    shell that never imports this module.
    """
    monkeypatch.setenv("RCB_OUTPUT_ROOT", str(tmp_path))
    assert (paths.results("lalonde", create=False) / "x.csv").is_relative_to(
        tmp_path
    )
    assert paths.figures("simulation", create=False).root.is_relative_to(tmp_path)
    assert paths.data("lalonde").root == CODE / "data/lalonde"


def test_explicit_output_root_beats_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RCB_OUTPUT_ROOT", str(tmp_path / "from-env"))
    chosen = tmp_path / "chosen"
    with paths.use_output_root(chosen):
        assert (paths.results("lalonde", create=False) / "x.csv").is_relative_to(
            chosen
        )


# ---------------------------------------------------------------------------
# The layering: experiments compute, notebooks display
# ---------------------------------------------------------------------------
NOTEBOOKS = sorted(CODE.glob("*.ipynb"))

#: Things a notebook must not do.  Reading a summary CSV and deriving a display
#: table from it is plotting; drawing a sample or fitting an estimator is not.
FORBIDDEN = (
    ("fit_rcb(", "fits estimators"),
    ("prepare_geometry(", "prepares estimator geometry"),
    ("read_h5ad(", "reads raw Replogle inputs"),
    ("fit_all_methods(", "fits Replogle estimators"),
    ("run_att_repeat(", "runs Replogle replications"),
    ("run_all(", "runs the full experiment pipeline"),
    ("draw_replication(", "draws its own Monte Carlo replication"),
    ("simulate_cell(", "runs a Monte Carlo sweep"),
    ("_main_draws(", "generates figure draws"),
    ("_aspect_draws(", "generates figure draws"),
    ("prepare_main_figure_inputs(", "computes figure inputs"),
    ("prepare_aspect_figure_inputs(", "computes figure inputs"),
    ("fit_primary(", "fits estimators"),
    ("fit_extended(", "fits estimators"),
    ("run_bootstrap(", "runs a bootstrap"),
)


def _code(notebook: Path) -> str:
    cells = json.loads(notebook.read_text())["cells"]
    return "\n".join(
        "".join(c["source"]) for c in cells if c["cell_type"] == "code"
    )


def test_there_are_notebooks_to_check() -> None:
    assert NOTEBOOKS, "no notebooks found at the top of code/"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_do_not_run_experiments(notebook: Path) -> None:
    source = _code(notebook)
    offenders = [
        f"{call} ({why})" for call, why in FORBIDDEN if call in source
    ]
    assert not offenders, (
        f"{notebook.name} computes rather than reads: {offenders}"
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_do_not_read_raw_data(notebook: Path) -> None:
    """Raw inputs are the experiments' business; notebooks read results."""
    source = _code(notebook)
    assert "paths.data(" not in source
    assert "DATA_DIR" not in source


# ---------------------------------------------------------------------------
# Experiments compute; none of them draw
# ---------------------------------------------------------------------------
def test_only_one_experiment_function_draws() -> None:
    """Plotting belongs in the notebooks; this stops it drifting back."""
    import ast

    offenders = []
    for path in sorted((CODE / "experiments").rglob("*.py")):
        tree = ast.parse(path.read_text())
        lines = path.read_text().split("\n")
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            body = "\n".join(lines[node.lineno - 1:node.end_lineno])
            # ".savefig(" is a call; a bare "savefig" is an rcParam key.
            if ".savefig(" not in body:
                continue
            here = (str(path.relative_to(CODE)), node.name)
            offenders.append(f"{here[0]}::{here[1]}")
    assert not offenders, (
        f"experiment scripts should not draw: {offenders}"
    )


#: Matches a quoted figure stem like ``"01_main_risk..."`` or
#: ``"02_required_metrics_phi0_0p5.pdf"``, capturing the leading ordinal and
#: the first descriptive word after it.  Intentional multi-file variants of
#: one logical figure (``01_ARB_..._phi0_0p5`` and ``01_ARB_..._phi0_1p25``,
#: or an f-string's static prefix ``10_split_sensitivity_eta_``) share both
#: the ordinal and that first word; a real collision -- two different figures
#: assigned the same ordinal -- does not.
_ORDINAL_STEM = re.compile(r'["\']([0-9]+[a-z]?)_([A-Za-z0-9]+)')


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_no_ordinal_collisions_within_a_notebook(notebook: Path) -> None:
    """Two figures in the same study must not share a leading ordinal."""
    by_ordinal: dict[str, set[str]] = {}
    for ordinal, word in _ORDINAL_STEM.findall(_code(notebook)):
        by_ordinal.setdefault(ordinal, set()).add(word)
    colliding = {o: words for o, words in by_ordinal.items() if len(words) > 1}
    assert not colliding, (
        f"{notebook.name} assigns ordinal(s) {colliding} to more than one "
        "figure"
    )


def test_notebooks_write_only_pdf() -> None:
    """PNGs existed so a notebook could display a saved image; plt.show() does that."""
    for notebook in NOTEBOOKS:
        source = _code(notebook)
        assert ".png" not in source, f"{notebook.name} still writes or reads a PNG"
        assert "IPython" not in source, f"{notebook.name} still imports IPython"


# ---------------------------------------------------------------------------
# Nothing machine- or person-specific is distributed
# ---------------------------------------------------------------------------
#: Directories that are local-only (ignored) rather than part of the repository.
_LOCAL_ONLY = {
    ".git", ".claude", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    "dev", "plan", "raw", "results",
}
_TEXT_SUFFIXES = {
    ".py", ".md", ".ipynb", ".sh", ".yml", ".yaml", ".txt", ".json", ".csv",
    ".R", ".cfg", ".toml",
}
_PERSONAL = re.compile(
    r"/Users/[^/\s\"']+/|/home/[^/\s\"']+/|/var/folders/"
    r"|[\w.+-]+@[\w-]+\.(?:com|edu|org|net)\b"
)


def test_no_personal_paths_or_addresses() -> None:
    """Home-directory paths, temporary paths and email addresses stay out."""
    offenders = []
    for path in sorted(CODE.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in _TEXT_SUFFIXES
            or _LOCAL_ONLY.intersection(path.relative_to(CODE).parts)
            or path == Path(__file__).resolve()
        ):
            continue
        match = _PERSONAL.search(path.read_text(errors="replace"))
        if match:
            offenders.append(f"{path.relative_to(CODE)}: {match.group(0)}")
    assert not offenders, f"personal or machine-specific strings: {offenders}"
