#!/usr/bin/env python


"""
End-to-end test of the `cognition run` entry point on one real public sample:
GLASS-OD 0002-R1, deposited at GEO as GSM8997664 (sentrix 206467010068_R06C01).

Unlike the other tests here this one needs no local data mount: the Grn/Red
IDAT pair is fetched straight from the GEO supplementary files over HTTPS,
gunzipped, and handed to the installed `cognition` script. That makes it the
one test covering the whole chain a user actually runs -- entry point,
predictor discovery under assets/, mepylome preprocessing, CNV, survival export
and the PDF report -- instead of a single function.

Prerequisites are skipped, not failed: no network, no `cognition` on PATH, or
no model builds under assets/models/ (fetch those with `cognition pull`). The
downloads are cached in tests/data/geo/ so only the first run pays for them;
delete that directory to force a re-download.

Marked slow and network -- deselect with `pytest -m "not network"`.
"""

import gzip
import shutil
import subprocess
from pathlib import Path

import pytest
import requests

from libcognition import ASSETS_PATH, __version__

GEO_SUPPL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8997nnn/GSM8997664/suppl/"
SENTRIX = "206467010068_R06C01"
# %5F is _ and %2E is . -- GEO serves the supplementary files under the escaped
# names, so these are passed to requests as-is rather than "cleaned up"
IDAT_URLS = {
    "Grn": GEO_SUPPL + "GSM8997664%5F206467010068%5FR06C01%5FGrn%2Eidat%2Egz",
    "Red": GEO_SUPPL + "GSM8997664%5F206467010068%5FR06C01%5FRed%2Eidat%2Egz",
}

CACHE_DIR = Path(__file__).parent / "data" / "geo"

# generous on purpose: the run imports torch, preprocesses the array and builds
# the CNV panel against the reference normals, which is minutes on a cold cache
RUN_TIMEOUT = 3600


def download_and_gunzip(url, target):
    """Fetch a gzipped IDAT and write it out unpacked, or skip the test.

    Downloads to a temporary name and only moves it into place once both the
    transfer and the gunzip succeeded, so an interrupted run cannot leave a
    truncated file behind that every later run then happily reuses.
    """
    partial = target.with_name(target.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as packed:
                with open(partial, "wb") as unpacked:
                    shutil.copyfileobj(packed, unpacked)
    except Exception as error:
        partial.unlink(missing_ok=True)
        pytest.skip(f"could not download {url}: {type(error).__name__}: {error}")

    partial.replace(target)


@pytest.fixture(scope="module")
def idat_pair():
    """The unpacked Grn/Red pair, downloaded once and kept in tests/data/geo/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}
    for channel, url in IDAT_URLS.items():
        idat = CACHE_DIR / f"GSM8997664_{SENTRIX}_{channel}.idat"
        if not idat.exists():
            download_and_gunzip(url, idat)
        paths[channel] = idat

    return paths["Grn"], paths["Red"]


@pytest.fixture(scope="module")
def cognition_cli():
    executable = shutil.which("cognition")
    if executable is None:
        pytest.skip("`cognition` is not on PATH -- install the package first")
    return executable


@pytest.mark.slow
@pytest.mark.network
def test_run_GLASS_OD_0002_R1(idat_pair, cognition_cli, tmp_path):
    models = sorted(Path(f"{ASSETS_PATH}/models/v{__version__}").glob("*/*/*.info.json"))
    if not models:
        pytest.skip(f"no model builds under {ASSETS_PATH}/models/v{__version__} "
                    f"-- run `cognition pull` first")

    idat_grn, idat_red = idat_pair

    # run from tmp_path: `cognition run` writes its report next to the working
    # directory, not next to the idats
    completed = subprocess.run(
        [cognition_cli, "run", str(idat_grn), str(idat_red)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
    )

    assert completed.returncode == 0, (
        f"`cognition run` exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}"
    )

    stem = f"{SENTRIX}.cognition-v{__version__}"
    written = sorted(path.name for path in tmp_path.iterdir())

    export = tmp_path / f"{stem}.overall_survival.txt"
    report = tmp_path / f"{stem}.overall_survival.pdf"

    assert export.name in written, f"no survival export written; got {written}"
    assert report.name in written, f"no survival report written; got {written}"

    # more than the header line: an export that only has column names means the
    # predictors ran but contributed nothing
    assert len(export.read_text().splitlines()) > 1, "survival export holds no rows"
    assert report.stat().st_size > 0, "survival report is empty"
