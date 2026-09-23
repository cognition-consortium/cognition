#!/usr/bin/env python


"""
End-to-end smoke test for the mepylome CNV pipeline
(libcognition.mepylome_helpers.mepylome_idat_to_cnv_database_export).

Runs the full IDAT -> CNV pipeline (hg38 manifest, shared bin grid, reference
normals) on one real sample per array type (450k, EPIC, EPICv2) and checks
that bins/detail/normals output is written and looks sane. Array type is
auto-detected from the IDAT itself (mepylome's ArrayType.from_idat), so this
also guards against wrapper/mepylome API mismatches like the
annotation._adjusted_manifest regression.

Each sample is run TWICE: once against a freshly built reference (no on-disk
or in-process cache) and once again forced to load the reference back from
the on-disk pickle mepylome_get_reference() just wrote. This is deliberate:
the ReferenceMethylData disk cache in idat_normals_cache/ is only as valid as
the mepylome version that pickled it, and mepylome is pinned to a moving git
ref (see pyproject.toml) — a stale pickle built by an older mepylome silently
breaks on unpickle into a newer one (e.g. the MethylData._log_intensity_fit
AttributeError seen in practice). Running "clean" then "cached" exercises
both code paths in mepylome_get_reference() and catches that regression
class instead of only catching it whenever someone happens to hit a stale
cache by hand.

The IDAT files and reference normals live under ./data and are not shipped
with the package, so each case is skipped when a prerequisite is missing
(e.g. CI without the data mount).
"""

import os

import pandas as pd
import pytest

from mepylome.dtypes.arrays import ArrayType

from libcognition import mepylome_helpers
from libcognition.mepylome_helpers import (
    HG38_MANIFESTS,
    NORMALS_BASE,
    SHARED_BINS_CSV,
    mepylome_idat_to_cnv_database_export,
    mepylome_output_paths,
    mepylome_reference_cache_path,
)

# One real sample per array type; the array type itself is auto-detected from
# the idat (not hardcoded here), so these are just cohort/sample pairs.
CNV_SAMPLES = [
    ("Verheul-Cell-Lines",
     "data/DNA_methylation/idat/Verheul-Cell-Lines/203175830152_R04C01"),
    ("GSE136361",
     "data/DNA_methylation/idat/GSE136361/GSM4047406_200394970005_R05C01"),
    ("CPTAC-3",
     "data/DNA_methylation/idat/CPTAC-3/C3L-00104-01_A"),
]


def _run_and_check(basename, dataset, filename, array_type, output_base):
    """Run the pipeline once and assert the three output files are sane."""
    mepylome_idat_to_cnv_database_export(basename, dataset, filename, array_type,
                        output_base=output_base, verbose=False)

    _, bins_file, detail_file, normals_file = mepylome_output_paths(
        dataset, filename, output_base
    )

    assert bins_file.exists()
    bins = pd.read_csv(bins_file)
    assert len(bins) > 0
    assert {"Chromosome", "Start", "End", "N_probes"}.issubset(bins.columns)

    assert detail_file.exists()

    assert normals_file.exists()
    with open(normals_file) as f:
        reference_samples = [line.strip() for line in f if line.strip()]
    assert len(reference_samples) > 0


@pytest.mark.parametrize("dataset, basename", CNV_SAMPLES,
                         ids=[p[0] for p in CNV_SAMPLES])
def test_mepylome_idat_to_cnv_clean_and_cached(dataset, basename, tmp_path):
    """CNV output is sane both from a fresh reference build and a cached one."""
    grn_file = basename + "_Grn.idat"
    red_file = basename + "_Red.idat"
    for path in (grn_file, red_file):
        if not os.path.exists(path):
            pytest.skip(f"IDAT file not available: {path}")

    array_type = str(ArrayType.from_idat(basename))

    if array_type in HG38_MANIFESTS and not HG38_MANIFESTS[array_type].exists():
        pytest.skip(f"hg38 manifest not available: {HG38_MANIFESTS[array_type]}")
    if not SHARED_BINS_CSV.exists():
        pytest.skip(f"Shared bin grid not available: {SHARED_BINS_CSV}")
    normals_dir = NORMALS_BASE / array_type
    if not normals_dir.is_dir() or not any(normals_dir.iterdir()):
        pytest.skip(f"Reference normals not available: {normals_dir}")

    filename = os.path.basename(basename)

    # The on-disk reference cache is real production state (idat_normals_cache/),
    # so it is moved aside rather than clobbered, and restored in `finally`.
    cache_file = mepylome_reference_cache_path(array_type)
    backup_file = cache_file.with_suffix(cache_file.suffix + ".test-backup")
    had_existing_cache = cache_file.exists()
    if had_existing_cache:
        cache_file.rename(backup_file)

    try:
        # 1) Clean: no on-disk cache, no in-process cache -> builds
        #    ReferenceMethylData from scratch and writes a fresh disk cache.
        mepylome_helpers._REFERENCE_CACHE.pop(array_type, None)
        assert not cache_file.exists()
        _run_and_check(basename, dataset, filename, array_type, tmp_path / "clean")
        assert cache_file.exists()

        # 2) Cached: drop only the in-process cache, so mepylome_get_reference()
        #    is forced to unpickle the file just written in step 1 -> this is
        #    the exact code path where a version-mismatched pickle breaks.
        mepylome_helpers._REFERENCE_CACHE.pop(array_type, None)
        _run_and_check(basename, dataset, filename, array_type, tmp_path / "cached")
    finally:
        cache_file.unlink(missing_ok=True)
        if had_existing_cache:
            backup_file.rename(cache_file)
