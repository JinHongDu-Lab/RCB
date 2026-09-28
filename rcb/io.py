"""Writing generated artifacts: tables, validation records, and provenance.

Every experiment wrote its own ``to_csv`` calls -- around thirty-five distinct
sites in Section 5 alone, and four separately hand-rolled validation-JSON
writers in Section 7.  Centralizing them buys one thing that matters beyond
tidiness: :func:`write_validation` refuses to record a validation file whose
checks did not pass, so a failed run cannot leave behind a document asserting
that it succeeded.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["package_versions", "save_json", "save_table", "write_validation"]


def save_table(frame: pd.DataFrame, directory, name: str) -> Path:
    """Write one table, compressing when the name asks for it.

    ``directory`` is joined with ``/`` rather than coerced to a ``Path``, so a
    :class:`rcb.paths.StudyDir` still applies its study prefix.
    """
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def save_json(payload: dict, directory, name: str) -> Path:
    """Write one JSON document, with numpy scalars coerced to plain Python."""
    path = directory / name
    path.write_text(json.dumps(payload, indent=2, default=_plain), encoding="utf-8")
    return path


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_validation(
    checks: dict, directory, name: str = "validation.json",
    advisory: frozenset[str] = frozenset(),
) -> Path:
    """Record a validation document, refusing to write it if a check failed.

    Boolean entries are the assertions; numeric entries are the measured
    quantities behind them and are recorded either way.  Writing a validation
    file for a run that did not validate would be worse than writing nothing.

    ``advisory`` names checks that are recorded but not required *for this
    run*.  It exists for assertions about the configuration rather than about
    the arithmetic -- "the dimension grid is the paper's 80/160/320/640" is
    true of a production run and false of a smoke run by construction, and a
    smoke run should say so in the file rather than abort.  Nothing about a
    computation ever belongs here.
    """
    failed = sorted(
        key for key, value in checks.items()
        if isinstance(value, (bool, np.bool_)) and not value
        and key not in advisory
    )
    if failed:
        raise RuntimeError(f"Validation failed, not recorded: {failed}")
    return save_json(checks, directory, name)


def package_versions() -> dict:
    """Versions of everything the numerical results depend on."""
    import matplotlib
    import scipy
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import seaborn

        versions["seaborn"] = seaborn.__version__
    except ImportError:
        pass
    return versions
