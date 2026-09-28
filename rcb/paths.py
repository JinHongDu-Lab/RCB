"""Where generated output goes.

`code/` holds three trees this module addresses, and their names say what is in them:

``results/``
    tables, JSON validation records, and prose summaries;
``figures/``
    PDF and PNG;
``data/``
    raw third-party inputs, read-only.

``results/`` and ``data/`` contain study subdirectories.
``figures/`` stays flat with study-prefixed filenames for manuscript compatibility.
Both conventions are applied by :class:`StudyDir` at the point of joining.

Paths resolve from this file, never from the working directory, so where a
script or notebook happens to be launched cannot change where results land.
:func:`use_output_root` redirects every tree beneath one directory, which is how
tests exercise the pipelines without touching committed results.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "STUDIES",
    "StudyDir",
    "data",
    "figures",
    "results",
    "use_output_root",
]

CODE = Path(__file__).resolve().parents[1]

#: Studies whose output may be written, each owning its results subdirectory
#: and its prefix in the flat figures tree.
#: ``lalonde_exploratory`` is the separate output of the exploratory pipeline in
#: ``experiments/lalonde/experiment.py``, which the audited R1--R9 run never
#: invokes; it is registered so its files stay visibly distinct from the formal
#: results rather than mingling with them.
STUDIES = ("simulation", "evaluation", "lalonde", "lalonde_exploratory", "replogle")

#: Set by :func:`use_output_root`.  ``None`` means "use whatever
#: ``RCB_OUTPUT_ROOT`` says, or ``code/``".
_override: Path | None = None

#: Environment override, read on every join rather than captured at import.
#:
#: ``run_experiments.sh --smoke`` sets this to a scratch directory.  A smoke run
#: uses eight replications where production uses five hundred; without a
#: redirect it would write that over the committed results, and the files give
#: no sign of which kind of run produced them.  Making the redirect the default
#: for smoke runs means the mistake cannot be made by forgetting something.
_ENV_ROOT = "RCB_OUTPUT_ROOT"


#: Trees the environment override does NOT move.  ``data/`` holds third-party
#: inputs, read-only; pointing a smoke run's ``data/`` at an empty scratch
#: directory just makes every study fail to find its input.  The context
#: manager still moves everything, because a test wants a sealed tree.
_INPUT_TREES = frozenset({"data"})


def _root(tree: str) -> Path:
    if _override is not None:
        return _override
    if tree in _INPUT_TREES:
        return CODE
    from os import getenv

    configured = getenv(_ENV_ROOT)
    return Path(configured) if configured else CODE


class Tree:
    """One output tree, resolved lazily.

    Modules routinely capture their output directory in a constant at import
    time.  Resolving the root on each join rather than at construction means
    :func:`use_output_root` still redirects those constants, so a test does not
    have to import the module it is testing from inside the context manager.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def root(self) -> Path:
        return _root(self.name) / self.name

    def __truediv__(self, name: str) -> Path:
        return self.root / str(name)

    def mkdir(self, *, parents: bool = True, exist_ok: bool = True) -> None:
        self.root.mkdir(parents=parents, exist_ok=exist_ok)

    def exists(self) -> bool:
        return self.root.exists()

    def __repr__(self) -> str:
        return f"Tree({self.name!r} -> {self.root})"


class StudyDir(Tree):
    """A study subdirectory, or a filename namespace for flat figures."""

    __slots__ = ("prefix",)

    def __init__(self, name: str, prefix: str) -> None:
        super().__init__(name)
        self.prefix = prefix

    @property
    def root(self) -> Path:
        root = super().root
        return root if self.name == "figures" else root / self.prefix

    def __truediv__(self, name: str) -> Path:
        name = str(name)
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"Nested output paths are not supported: {name!r}")
        return self.root / (f"{self.prefix}_{name}" if self.name == "figures" else name)

    # Deliberately not os.PathLike: coercion would freeze lazy redirection,
    # and for figures it would also discard the study filename prefix.

    def __repr__(self) -> str:
        return f"StudyDir({self.root}, prefix={self.prefix!r})"


@contextmanager
def use_output_root(root: Path):
    """Redirect every output tree beneath ``root`` for the duration."""
    global _override
    previous = _override
    _override = Path(root)
    try:
        yield _override
    finally:
        _override = previous


def _study_dir(tree: str, study: str, create: bool) -> StudyDir:
    if study not in STUDIES:
        raise KeyError(f"Unknown study: {study!r}; expected one of {STUDIES}")
    directory = StudyDir(tree, study)
    if create:
        directory.mkdir()
    return directory


def results(study: str, *, create: bool = True) -> StudyDir:
    """Where ``study`` writes its tables, JSON and prose summaries."""
    return _study_dir("results", study, create)


def figures(study: str, *, create: bool = True) -> StudyDir:
    """Where ``study`` writes its figures."""
    return _study_dir("figures", study, create)


def data(study: str) -> StudyDir:
    """Study inputs, read-only, under their canonical names."""
    return _study_dir("data", study, create=False)
