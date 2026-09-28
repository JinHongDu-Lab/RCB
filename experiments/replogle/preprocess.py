"""Build the two K562 inputs of the Replogle study from the public scPerturb file.

The source is the essential-gene K562 Perturb-seq screen of Replogle et al.
(2022, Cell 185:2559-2575), in the harmonized form distributed by scPerturb
(Peidli et al., 2024, Nature Methods; Zenodo record 10044268).  Stages:

1. ``download``  -- fetch ``ReplogleWeissman2022_K562_essential.h5ad`` and
   check its SHA-256 digest;
2. ``qc``        -- filter cells, genes and perturbations with crispyx;
3. ``normalize`` -- library-size normalize to 1e4 counts and log1p;
4. ``pseudobulk``-- mean log1p expression per (perturbation, batch) group
   with at least five cells -> ``Replogle-E-k562_pseudobulk.h5ad``;
5. ``wilcoxon``  -- batch-stratified Wilcoxon test of each perturbation
   against control -> ``Replogle-E-k562_wilcoxon_batch_corrected.h5ad``.

Intermediates go to ``data/replogle/build/`` and the two inputs to
``data/replogle/raw/``; both are ignored by Git::

    python -m experiments.replogle.preprocess
    python -m experiments.replogle.run

Every stage skips work whose output already exists unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from rcb import paths
from .run import sha256

ZENODO_RECORD = "10044268"
SOURCE_NAME = "ReplogleWeissman2022_K562_essential.h5ad"
SOURCE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{SOURCE_NAME}?download=1"
#: SHA-256 of the Zenodo file.  Zenodo lists only an MD5 checksum, so this
#: digest was computed from a copy whose MD5 matched the record.
SOURCE_SHA256 = "412fd0df8c4ccea9f4db91cd88033c49200838b29d40945e48574be588b48789"

PERTURBATION = "perturbation"
CONTROL = "control"
BATCH = "batch"

#: QC thresholds.  ``min_cells_per_perturbation=20`` reproduces the recorded
#: counts exactly (309,915 of 310,385 cells and 2,021 of 2,057 non-control
#: perturbations kept); the cell and gene thresholds are the crispyx defaults,
#: which remove nothing on this file.
QC = {"min_genes": 100, "min_cells_per_gene": 100, "min_cells_per_perturbation": 20}
PSEUDOBULK = {
    "groupby": [PERTURBATION, BATCH],
    "method": "mean_log1p",
    "min_cells": 5,
    "random_state": 0,
}
#: Shapes of the inputs the committed results were computed from.
EXPECTED = {
    "cells": 309_915,
    "genes": 8_563,
    "pseudobulk_rows": 22_075,
    "wilcoxon_rows": 2_021,
}

STAGES = ("download", "qc", "normalize", "pseudobulk", "wilcoxon", "check")
PSEUDOBULK_NAME = "Replogle-E-k562_pseudobulk.h5ad"
WILCOXON_NAME = "Replogle-E-k562_wilcoxon_batch_corrected.h5ad"


def _directories():
    root = paths.data("replogle")
    build, raw = root / "build", root / "raw"
    build.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    return build, raw


def _replace_atomically(write, target: Path) -> None:
    """Run ``write(temporary)`` and move the result onto ``target``."""
    fd, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".partial.h5ad", dir=target.parent
    )
    os.close(fd)
    temporary = Path(temporary)
    temporary.unlink()
    try:
        write(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def download(build: Path, force: bool) -> Path:
    target = build / SOURCE_NAME
    if target.exists() and not force:
        # A file placed here by hand is checked like a downloaded one.
        if sha256(target) != SOURCE_SHA256:
            raise IOError(f"SHA-256 mismatch for {target}; delete it and retry")
        return target
    partial = target.with_suffix(".h5ad.partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(SOURCE_URL, headers={"Range": f"bytes={offset}-"})
    print(f"Downloading {SOURCE_URL} (resuming at {offset} bytes)", flush=True)
    with urllib.request.urlopen(request) as response, partial.open("ab") as stream:
        if offset and response.status != 206:
            stream.truncate(0)
        shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    if sha256(partial) != SOURCE_SHA256:
        raise IOError(f"SHA-256 mismatch for {partial}; delete it and retry")
    partial.replace(target)
    return target


def quality_control(source: Path, build: Path, force: bool) -> Path:
    import crispyx as cx

    target = build / "Replogle-E-k562_qc.h5ad"
    if target.exists() and not force:
        return target
    result = cx.qc.quality_control_summary(
        source, perturbation_column=PERTURBATION, control_label=CONTROL, **QC
    )
    cells, genes = int(result.cell_mask.sum()), int(result.gene_mask.sum())
    if (cells, genes) != (EXPECTED["cells"], EXPECTED["genes"]):
        raise RuntimeError(
            f"QC kept {cells} cells x {genes} genes; expected "
            f"{EXPECTED['cells']} x {EXPECTED['genes']}"
        )
    _replace_atomically(
        lambda out: cx.qc.write_filtered_subset(
            source, cell_mask=result.cell_mask, gene_mask=result.gene_mask,
            output_path=out,
        ),
        target,
    )
    return target


def _is_log1p(path: Path) -> bool:
    """Raw counts are integers; log1p-normalized values mostly are not."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        x = handle["X"]
        sample = x["data"][:100_000] if isinstance(x, h5py.Group) else x[:50, :].ravel()
    nonzero = sample[sample != 0]
    return bool(nonzero.size) and np.mean(np.abs(nonzero - np.round(nonzero)) < 1e-6) < 0.5


def normalize(qc: Path, build: Path, force: bool) -> Path:
    import crispyx as cx

    target = build / "Replogle-E-k562.h5ad"
    if target.exists() and not force:
        return target
    if _is_log1p(qc):
        shutil.copyfile(qc, target)
        return target
    # crispyx streams the source twice, so it cannot overwrite its input.
    _replace_atomically(
        lambda out: cx.pp.normalize_total_log1p(qc, output_path=out, target_sum=1e4),
        target,
    )
    return target


def pseudobulk(single_cell: Path, raw: Path, force: bool) -> Path:
    import crispyx as cx

    target = raw / PSEUDOBULK_NAME
    if target.exists() and not force:
        return target
    _replace_atomically(
        lambda out: cx.pb.aggregate(single_cell, output_path=out, **PSEUDOBULK),
        target,
    )
    return target


def wilcoxon(single_cell: Path, build: Path, raw: Path, force: bool) -> Path:
    import crispyx as cx

    target = raw / WILCOXON_NAME
    if target.exists() and not force:
        return target
    # The rank test streams genes, which needs column access.
    csc = build / "Replogle-E-k562_csc.h5ad"
    if not csc.exists() or force:
        if cx.data.get_matrix_storage_format(single_cell) == "csc":
            shutil.copyfile(single_cell, csc)
        else:
            _replace_atomically(
                lambda out: cx.data.convert_to_csc(single_cell, output_path=out), csc
            )
    _replace_atomically(
        lambda out: cx.de.wilcoxon_test(
            csc,
            perturbation_column=PERTURBATION,
            control_label=CONTROL,
            batch_column=BATCH,
            corr_method="benjamini-hochberg",
            output_path=out,
        ),
        target,
    )
    return target


def check_shapes(raw: Path) -> None:
    import anndata as ad

    for name, rows in ((PSEUDOBULK_NAME, "pseudobulk_rows"), (WILCOXON_NAME, "wilcoxon_rows")):
        backed = ad.read_h5ad(raw / name, backed="r")
        try:
            shape = backed.shape
        finally:
            backed.file.close()
        if shape != (EXPECTED[rows], EXPECTED["genes"]):
            raise RuntimeError(f"{name} has shape {shape}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs")
    parser.add_argument("--list", action="store_true", help="List stages and exit")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(STAGES))
        return
    build, raw = _directories()
    source = download(build, args.force)
    print("Replogle preprocessing: qc", flush=True)
    qc = quality_control(source, build, args.force)
    print("Replogle preprocessing: normalize", flush=True)
    single_cell = normalize(qc, build, args.force)
    print("Replogle preprocessing: pseudobulk", flush=True)
    pseudobulk(single_cell, raw, args.force)
    print("Replogle preprocessing: wilcoxon", flush=True)
    wilcoxon(single_cell, build, raw, args.force)
    check_shapes(raw)
    print(f"Replogle inputs written to {raw}", flush=True)


if __name__ == "__main__":
    main()
