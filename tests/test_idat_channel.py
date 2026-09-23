#!/usr/bin/env python


"""
Tests for idat_channel(): detecting whether an IDAT file is the Grn or Red
channel from the EXTENSION control probe intensities (independent of file name).

The test pairs cover multiple cohorts and array types (450k, EPIC, EPICv2):
CATNON, EORTC22033 (22033), Cognition, GLASS-NL, GLASS-OD, TCGA-GBM, TCGA-LGG.

The IDAT files live under ./data and are not shipped with the package, so each
case is skipped when the file is absent (e.g. CI without the data mount).
"""

import os

import pytest

from libcognition.utils import idat_channel

# Hard-coded Grn/Red IDAT pairs, one (or more) per cohort / array type.
IDAT_PAIRS = [
    ("CATNON",
     "data/DNA_methylation/idat/CATNON/200379150068_R07C01_Grn.idat",
     "data/DNA_methylation/idat/CATNON/200379150068_R07C01_Red.idat"),
    ("EORTC22033",
     "data/DNA_methylation/idat/EORTC22033/GSM2794518_9257626034_R01C02_Grn.idat",
     "data/DNA_methylation/idat/EORTC22033/GSM2794518_9257626034_R01C02_Red.idat"),
    ("Cognition",
     "data/DNA_methylation/idat/Cognition/207709860077_R02C01_Grn.idat",
     "data/DNA_methylation/idat/Cognition/207709860077_R02C01_Red.idat"),
    ("GLASS-NL",
     "data/DNA_methylation/idat/GLASS-NL/203175700013_R01C01_Grn.idat",
     "data/DNA_methylation/idat/GLASS-NL/203175700013_R01C01_Red.idat"),
    ("GLASS-OD",
     "data/DNA_methylation/idat/GLASS-OD/201496850071_R02C01_Grn.idat",
     "data/DNA_methylation/idat/GLASS-OD/201496850071_R02C01_Red.idat"),
    ("TCGA-GBM",
     "data/DNA_methylation/idat/TCGA-GBM/TCGA-06-0125-01A_Grn.idat",
     "data/DNA_methylation/idat/TCGA-GBM/TCGA-06-0125-01A_Red.idat"),
    ("TCGA-LGG-CS-4938",
     "data/DNA_methylation/idat/TCGA-LGG/TCGA-CS-4938-01B_Grn.idat",
     "data/DNA_methylation/idat/TCGA-LGG/TCGA-CS-4938-01B_Red.idat"),
    ("TCGA-LGG-DU-5870",
     "data/DNA_methylation/idat/TCGA-LGG/TCGA-DU-5870-02A_Grn.idat",
     "data/DNA_methylation/idat/TCGA-LGG/TCGA-DU-5870-02A_Red.idat"),
]


@pytest.mark.parametrize("cohort, grn_file, red_file", IDAT_PAIRS,
                         ids=[p[0] for p in IDAT_PAIRS])
def test_idat_channel_detection(cohort, grn_file, red_file):
    """The Grn file must be detected as 'Grn' and the Red file as 'Red'."""
    for path in (grn_file, red_file):
        if not os.path.exists(path):
            pytest.skip(f"IDAT file not available: {path}")

    assert idat_channel(grn_file) == "Grn"
    assert idat_channel(red_file) == "Red"
