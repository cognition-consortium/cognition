#!/usr/bin/env python3
"""Shared mepylome CNV helpers (hg38 manifests + a common bin grid).

This module centralises the "IDAT -> CNV output" pipeline so it can be reused
from the smoke-test script (``sandbox/test_cnv_single.py``), the database
autocompletion (``libcognition.database.tumor_database.check_cnvs``) and the
standalone CLI (``libcognition.cli``) without duplicating the logic. The
shared computation lives in ``mepylome_idat_to_cnv_core``; the dataset-bound
and CLI entry points differ only in how they locate the reference set and
where they write output.

All array types are run on hg38:

* **epic** / **450k** use the custom hg38 manifests in
  ``<ASSETS_PATH>/reference/`` (drop-in copies of the mepylome
  processed manifests with Chromosome/Start/End replaced by hg38 coordinates;
  see ``scripts/make_hg38_manifest.py``). Each is accompanied by its original
  control-probes file (controls carry no genomic annotation, so the hg19 copy
  is reused) named so mepylome finds it locally and does not re-process the raw
  hg19 manifest over it.
* **epicv2** is already hg38 natively, so its stock mepylome manifest is used.

Every array type is then re-gridded onto one shared bin grid (the coarse 450k
grid) so all platforms yield the same bins and stay directly comparable.

Output filenames carry an ``.hg38`` tag to distinguish them from the legacy
hg19 output:

    <dataset>/<filename>.cnv_bins.hg38.csv
    <dataset>/<filename>.cnv_detail.hg38.csv
    <dataset>/<filename>.normals.hg38.txt
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import ASSETS_PATH, PROJECT_ROOT, __version__

# Output / input locations (match libcognition.database paths).
CNV_OUTPUT_BASE = PROJECT_ROOT / "data/DNA_methylation/CNV-mepylome"
# References are split out per array type: <NORMALS_BASE>/<array_type>/...
# The set of normals is curated in the database via the 'Include as CNV
# reference' column and materialized into this directory as symlinks by
# database.validate_normal_idats(); everything found here is used, so there is
# no separate whitelist to keep in sync.
NORMALS_BASE = PROJECT_ROOT / "data/DNA_methylation/idat_normals"
# Disk cache for the (expensive) per-array-type ReferenceMethylData.
REFERENCE_CACHE_DIR = ASSETS_PATH / "reference" / f"v{__version__}" / "CNV"

# hg38 manifests per array type (drop-in copies of the mepylome processed
# manifests with hg38 coordinates). EPICv2 is already hg38, so it is not here
# and falls back to the stock mepylome manifest.
HG38_MANIFEST_DIR = ASSETS_PATH / "reference"
HG38_MANIFESTS = {
    "epic": HG38_MANIFEST_DIR / "manifest-epic.hg38.csv.gz",
    "450k": HG38_MANIFEST_DIR / "manifest-450k.hg38.csv.gz",
}

# Shared bin grid for ALL array types, so they all yield the same number of
# bins. We use the bins that come out of the 450k array by default — these are
# the coarsest (fewest bins), so every other array type can be mapped onto them
# without losing resolution. Any 450k cnv_bins output defines the same grid;
# this copy lives at a fixed, non-sample-bound location next to the hg38
# manifests so it is not wiped when CNV outputs are regenerated/cleaned.
SHARED_BINS_CSV = HG38_MANIFEST_DIR / "shared_bins.hg38.csv"


def mepylome_output_paths(dataset: str, filename: str, output_base: Path = CNV_OUTPUT_BASE):
    """Return (out_dir, bins_file, detail_file, normals_file) for a sample.

    The hg38 tag in the filenames keeps these distinct from legacy hg19 output.
    """
    out_dir = output_base / dataset
    return (out_dir,
            out_dir / f"{filename}.cnv_bins.hg38.csv",
            out_dir / f"{filename}.cnv_detail.hg38.csv",
            out_dir / f"{filename}.normals.hg38.txt")


def mepylome_load_shared_bins(csv_path: Path = SHARED_BINS_CSV):
    """Build a mepylome-compatible bins PyRanges from a cnv_bins CSV.

    The CSV is already in mepylome's sorted bin order; we reproduce the
    Chromosome/Start/End grid (0-based, as written by cnv.bins.to_csv) plus a
    contiguous bins_index. N_probes is array-type specific and (re)computed in
    mepylome_apply_shared_bins, so it is not carried here.
    """
    import pyranges1 as pr

    df = pd.read_csv(csv_path)[["Chromosome", "Start", "End"]].reset_index(drop=True)
    bins = pr.PyRanges(df)
    bins["bins_index"] = np.arange(len(bins))
    return bins


def mepylome_apply_shared_bins(annotation, shared_bins):
    """Replace an Annotation's bin grid with the shared grid, in place.

    CNV() has no bins argument and set_bins() aggregates ratios through
    annotation._cpg_bins, so changing the grid means overwriting BOTH the bins
    and the cpg->bin mapping. The mapping is recomputed by overlapping the
    shared bins with this array type's own (hg38) probe coordinates, and
    N_probes is recomputed to reflect that array type.
    """
    import pyranges1 as pr

    cpg = pd.DataFrame(
        shared_bins.join_overlaps(annotation._adjusted_manifest)
    )[["bins_index", "IlmnID"]]
    annotation._cpg_bins = cpg.set_index("bins_index")

    counts = cpg.groupby("bins_index").size()
    bins_df = pd.DataFrame(shared_bins).copy()
    bins_df["N_probes"] = bins_df["bins_index"].map(counts).fillna(0).astype(int)
    annotation.bins = pr.PyRanges(bins_df)
    return annotation


# Caches so repeated samples of one array type don't rebuild heavy objects.
_ANNOTATION_CACHE: dict = {}
_REFERENCE_CACHE: dict = {}
_SHARED_BINS = None


def mepylome_shared_bins():
    """The shared bin grid, loaded once.

    Raises FileNotFoundError if the grid CSV is missing: it is a hard
    prerequisite (the common 450k grid every array type is mapped onto) and
    cannot be fabricated, so this must fail loudly and immediately.
    """
    global _SHARED_BINS
    if _SHARED_BINS is None:
        if not SHARED_BINS_CSV.exists():
            raise FileNotFoundError(
                f"Shared bin grid not found: {SHARED_BINS_CSV}\n"
                "This 450k '.cnv_bins.hg38.csv' defines the common bin grid for "
                "all array types and must exist before any CNV is generated. "
                "Generate or restore that 450k sample's CNV output first."
            )
        _SHARED_BINS = mepylome_load_shared_bins(SHARED_BINS_CSV)
    return _SHARED_BINS


def mepylome_get_annotation(array_type: str):
    """hg38 Annotation for an array type, re-gridded onto the shared bins.

    For 'epic'/'450k' the custom hg38 manifest is loaded; 'epicv2' is already
    hg38, so its stock mepylome manifest is used. Built once per array type.
    """
    from mepylome.dtypes.cnv import Annotation
    from mepylome.dtypes.manifests import Manifest

    if array_type not in _ANNOTATION_CACHE:
        proc_path = HG38_MANIFESTS.get(array_type)
        manifest = (Manifest(array_type, proc_path=str(proc_path))
                    if proc_path is not None else Manifest(array_type))
        annotation = Annotation(manifest=manifest, array_type=array_type)
        mepylome_apply_shared_bins(annotation, mepylome_shared_bins())
        _ANNOTATION_CACHE[array_type] = annotation
    return _ANNOTATION_CACHE[array_type]


def mepylome_normal_basepaths(array_type: str, normals_base: Path = NORMALS_BASE) -> list[Path]:
    """The reference idat basepaths for one array type.

    Every valid normal found under <normals_base>/<array_type>/ is used. That
    directory is the curated reference set: database.validate_normal_idats()
    keeps it in sync with the db 'Include as CNV reference' column, so there is
    no additional filtering here.

    Single source of truth for which normals go into the reference: both the
    build (mepylome_get_reference) and the cache filename
    (mepylome_reference_cache_path) go through here, so the filename can never
    describe a different set of normals than the one actually built.
    """
    from mepylome import idat_basepaths

    return idat_basepaths(str(Path(normals_base) / array_type), only_valid=True)


def mepylome_reference_cache_path(array_type: str, normals_base: Path = NORMALS_BASE,
                                  cache_dir: Path = REFERENCE_CACHE_DIR) -> Path:
    """Disk-cache file for one array type's ReferenceMethylData.

    Filename encodes today's date and the number of reference idat's found
    under <normals_base>/<array_type>/ (not hg19/hg38 -- the cached object is
    raw per-CpG intensities, not genome-build-specific), so a rebuilt or
    resized reference set gets its own file instead of silently reusing a
    stale one built from a different set of normals. (A same-day change that
    swaps one normal for another without changing the count is the one case the
    count+date tag does not catch.)
    """
    n_normals = len(mepylome_normal_basepaths(array_type, normals_base))
    return (Path(cache_dir) /
            f"{date.today().isoformat()}__normals__{array_type}__n{n_normals}.pkl")


def mepylome_get_reference(array_type: str, normals_base: Path = NORMALS_BASE):
    """ReferenceMethylData for one array type, from <normals_base>/<array_type>/.

    Cached both in-process (_REFERENCE_CACHE) and on disk: the first build
    serializes the pristine object so other processes (e.g. `parallel` jobs)
    load it instead of re-reading every normal idat. pickle is used over joblib
    here because the payload is DataFrame-backed (joblib's mmap is unusable since
    CNV mutates the reference in place, and its compression only trades CPU for
    size) and plain pickle is both smaller and faster to load for this object.
    """
    import os
    import pickle
    from mepylome import ReferenceMethylData

    if array_type in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[array_type]

    cache_file = mepylome_reference_cache_path(array_type, normals_base)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            out = pickle.load(f)
        print(f"Reference data (cache): {cache_file}")
    else:
        # An explicit basepath list (only_valid), not the directory: passing the
        # directory would let mepylome glob it itself and pick up half pairs.
        basepaths = mepylome_normal_basepaths(array_type, normals_base)
        if not basepaths:
            raise FileNotFoundError(
                f"No reference idat's found for '{array_type}' under "
                f"{Path(normals_base) / array_type}."
            )
        out = ReferenceMethylData([str(p) for p in basepaths])
        # Serialize the pristine object IMMEDIATELY — before it is handed to CNV,
        # which mutates the reference in place — so the cached file is always clean.
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_file)  # atomic: concurrent jobs never see a partial file
        print(f"Reference data (built + cached): {cache_file}")

    _REFERENCE_CACHE[array_type] = out
    return out


def mepylome_latest_reference(array_type: str, cache_dir: Path = REFERENCE_CACHE_DIR):
    """Newest compiled ReferenceMethylData for one array type, from the assets.

    For the CLI: loads the shipped pickle as-is, without looking at the normal
    idat's or rebuilding. Newest = latest date, then largest n.
    """
    import pickle

    def sort_key(p):
        return (p.stem.split("__")[0], int(p.stem.rsplit("__n", 1)[1]))

    candidates = sorted(Path(cache_dir).glob(f"*__normals__{array_type}__n*.pkl"), key=sort_key)
    if not candidates:
        raise FileNotFoundError(f"No compiled CNV reference for '{array_type}' in {cache_dir}")
    with open(candidates[-1], "rb") as f:
        out = pickle.load(f)
    print(f"Reference data (asset): {candidates[-1]}")
    return out


def mepylome_idat_to_cnv_core(idat_grn, idat_red, array_type: str | None, reference_loader,
                               verbose: bool = True):
    """Run mepylome CNV for one IDAT pair; returns (cnv, reference_samples, array_type).

    This is the part shared by every entry point: load the sample, resolve its
    hg38 annotation (cached per array type) and its reference set, then fit CNV
    bins + gene-level detail. No output is written here — callers differ in
    where/how they write, so that stays their own responsibility.

    `array_type` may be None, in which case it is detected from the IDAT header
    itself (mepylome does this internally) after loading; the detected value is
    then used for both annotation/reference lookup and the return value.

    `reference_loader` is a callable `(array_type) -> ReferenceMethylData`:
    callers locate references differently (a per-array-type subfolder under a
    shared normals base vs. a directly given path), so that lookup is left to
    the caller instead of being hardcoded here.
    """
    from mepylome import MethylData, CNV

    def log(msg):
        if verbose:
            print(msg)

    idat_grn = str(idat_grn)
    idat_red = str(idat_red)

    # Load sample
    methyl_data = MethylData(file=[idat_grn, idat_red], prep='illumina')
    if array_type is None:
        array_type = str(methyl_data.array_type)
    log(f"Array type: {methyl_data.array_type}")

    # hg38 annotation for this array type, re-gridded onto the shared 450k bins.
    annotation = mepylome_get_annotation(array_type)

    reference = reference_loader(array_type)

    # Run CNV (bins + gene-level detail; segments/CBS are not written)
    cnv = CNV(methyl_data, reference, annotation)
    cnv.fit()
    cnv.set_bins()
    cnv.set_detail()

    # Reference samples used for this array type. Index with the ArrayType enum
    # (not its string) — ReferenceMethylData.__getitem__ keys on the enum and
    # returns a MethylData whose betas columns are the reference sample names.
    ref_used = reference[methyl_data.array_type]
    reference_samples = list(ref_used.betas.columns)

    return cnv, reference_samples, array_type


def mepylome_idat_to_cnv_database_export(basename, dataset: str, filename: str, array_type: str,
                         normals_base: Path = NORMALS_BASE,
                         output_base: Path = CNV_OUTPUT_BASE,
                         verbose: bool = True):
    """Run mepylome CNV for one sample, from IDAT pair to written output.

    `array_type` ('epic' / '450k' / 'epicv2') selects both the hg38 manifest
    (custom for epic/450k, native for epicv2) and the matching reference set
    under <normals_base>/<array_type>/, so only the relevant normals are
    imported. The Annotation derives adjusted_manifest, bins and the cpg->bin
    mapping from hg38 coordinates and is re-gridded onto the shared bins, so the
    whole CNV runs on hg38 probe positions with a platform-common bin grid.

    `verbose` prints per-sample progress; set it False when driving this from a
    tqdm loop (e.g. database autocompletion) to keep the progress bar clean.

    Output is written under <output_base>/<dataset>/<filename>.* — this layout
    (and dataset/filename matching the database rows) is what
    Database.read_individual_cnvs_threaded later relies on to read the CSVs
    back into a bins x samples matrix, so it's not just an output convention.
    """
    def log(msg):
        if verbose:
            print(msg)

    basename = Path(basename)
    idat_grn = str(basename) + "_Grn.idat"
    idat_red = str(basename) + "_Red.idat"
    out_dir, bins_file, detail_file, normals_file = mepylome_output_paths(
        dataset, filename, output_base
    )

    log(f"Sample: {basename}  ({dataset}/{filename}) [{array_type}]")

    # Only the references for this array type (split out per type on disk).
    cnv, reference_samples, array_type = mepylome_idat_to_cnv_core(
        idat_grn, idat_red, array_type,
        reference_loader=lambda at: mepylome_get_reference(at, normals_base),
        verbose=verbose,
    )

    # Write output
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(cnv, 'bins', None) is not None:
        cnv.bins.to_csv(bins_file)
        log(f"✓ {bins_file} ({bins_file.stat().st_size / 1024:.1f} KB)")
    else:
        print("Warning: bins not available", file=sys.stderr)

    if getattr(cnv, 'detail', None) is not None:
        cnv.detail.to_csv(detail_file)
        log(f"✓ {detail_file} ({detail_file.stat().st_size / 1024:.1f} KB)")
    else:
        print("Warning: detail not available", file=sys.stderr)

    with open(normals_file, 'w') as f:
        for sample in reference_samples:
            f.write(f"{sample}\n")
    log(f"✓ {normals_file} ({len(reference_samples)} reference samples)")


def mepylome_idat_to_cnv_cli(idat_grn, idat_red, output_base: Path,
                              verbose: bool = True):
    """Run mepylome CNV for one sample given explicit IDAT files and an output stem.

    Standalone counterpart to mepylome_idat_to_cnv_database_export, for a single
    one-off run instead of the dataset-bound pipeline:

    * The reference is the newest compiled pickle in REFERENCE_CACHE_DIR
      (mepylome_latest_reference); no normal idat's are needed or checked.
    * `output_base` is the full output path stem (not a directory): output is
      written to exactly "<output_base>_cnv_bins.hg38.csv",
      "<output_base>_cnv_detail.hg38.csv" and "<output_base>_normals.hg38.txt".
    """
    def log(msg):
        if verbose:
            print(msg)

    output_base = Path(output_base)
    bins_file = Path(f"{output_base}.cognition-v{__version__}.cnv_bins.hg38.csv")
    detail_file = Path(f"{output_base}.cognition-v{__version__}.cnv_detail.hg38.csv")
    plot_file = Path(f"{output_base}.cognition-v{__version__}.cnv.hg38.pdf")
    normals_file = Path(f"{output_base}.cognition-v{__version__}.cnv_used_normals.hg38.txt")

    log(f"Sample: {idat_grn}, {idat_red}")

    cnv, reference_samples, array_type = mepylome_idat_to_cnv_core(
        idat_grn, idat_red, None,
        reference_loader=mepylome_latest_reference,
        verbose=verbose,
    )

    # Write output
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if getattr(cnv, 'bins', None) is not None:
        cnv.bins.to_csv(bins_file)
        log(f"✓ {bins_file} ({bins_file.stat().st_size / 1024:.1f} KB)")
    else:
        print("Warning: bins not available", file=sys.stderr)

    if getattr(cnv, 'detail', None) is not None:
        cnv.detail.to_csv(detail_file)
        log(f"✓ {detail_file} ({detail_file.stat().st_size / 1024:.1f} KB)")
    else:
        print("Warning: detail not available", file=sys.stderr)

    with open(normals_file, 'w') as f:
        for sample in reference_samples:
            f.write(f"{sample}\n")
    log(f"✓ {normals_file} ({len(reference_samples)} reference samples)")

    # Genomic plot: bins_file is the genome-wide track, detail_file adds the
    # per-gene emphasis on top. Needs both CSVs to have been written above.
    if bins_file.exists() and detail_file.exists():
        from .outputs import plot_cnv
        plot_cnv(bins_file, detail_file, plot_file, output_base.name)
        log(f"✓ {plot_file}")
    else:
        print("Warning: CNV plot skipped (bins and/or detail not available)", file=sys.stderr)
