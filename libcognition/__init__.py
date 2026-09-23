#!/usr/bin/env python3

import importlib
from pathlib import Path

import logging


def _suppress_logging():
    logging.getLogger().setLevel(logging.WARNING)
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.setLevel(logging.WARNING)


_suppress_logging() # suppresses some weird import errors


__version__ = '1.0.4'


PROJECT_ROOT = Path(__file__).resolve().parent.parent

#ASSETS_PATH = Path(__file__).parent.parent / "assets"
ASSETS_PATH = PROJECT_ROOT / "assets"

GITHUB_URL = "https://github.com/cognition-consortium/cognition"
HUGGINGFACE_URL = "https://huggingface.co/ErasmusMC-Neuro-Oncology/cognition"


DISCLAIMER = (
    "\n"
    "------------------------------------------------------------------------\n"
    "!!! This software is intended for research purposes only and has not !!!\n"
    "!!!   been validated for clinical decision-making or patient care.   !!!\n"
    "------------------------------------------------------------------------\n"
    "\n"
    "Written, developed and (C) by Dr. Youri Hoogstate and Dr. Richard Schoonhoven\n"
    f"<{GITHUB_URL}>"
)


DAYS_PER_YEAR = 365.24219
MAX_FOLLOW_UP_YEARS = 26.5



# trick for more efficient / lazy loading, otherwise all cude stuff is loaded even when listing models etc.

_LAZY_ATTRS = {
    'database': '.database',
    'idat': '.database',
    'tumor_db': '.tumortypes',

    'idat_to_data_container_mepylome': '.utils',
    'MEPYLOME_KEYS': '.utils',
    'epicv2_to_epic': '.utils',

    'EndPoint': '.predictors',
    'PredictorType': '.predictors',
    'Predictor': '.predictors',

    'plot_actual_vs_predicted_os': '.notebook_functions',
    'calibrate_classifier': '.notebook_functions',
    'export_classifier': '.notebook_functions',
    'build_pipeline': '.notebook_functions',
    'breakdown_notebook_filename': '.notebook_functions',

    'SurvivalCalibrator': '.calibrators',

    # Feature selection utilities
    #_logrank_feature_quantiles,
    #_logrank_feature,
    #fast_logrank_score_parallel,
    'fast_logrank_score_vectorized': '.custom_transformers',

    # Model wrappers
    'SurvivalPipeline': '.custom_transformers',
    'WeibullAFTFitterWrapperExtended': '.custom_transformers',
    'LogLogisticAFTFitterWrapperExtended': '.custom_transformers',
    'LogNormalAFTFitterWrapperExtended': '.custom_transformers',
    'CoxPHFitterWrapper': '.custom_transformers',
    'CoxPHFitterWrapperExtended': '.custom_transformers',
    'RandomSurvivalForestWrapperExtended': '.custom_transformers',
    'DeepSurvWrapper': '.custom_transformers',
    'DeepSurvWrapperExtended': '.custom_transformers',
    'DeepHitWrapper': '.custom_transformers',
    'DeepHitWrapperExtended': '.custom_transformers',
    'PCHazardWrapper': '.custom_transformers',
    'PCHazardWrapperExtended': '.custom_transformers',
}


def __getattr__(name):
    # Explicit name -> defining submodule (the submodule's own top-level
    # `from .x import *` chain may pull in more names as a side effect, e.g.
    # 'database' cascades into utils/tumortypes/mnp/methylscape/tumorlocations).
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        _suppress_logging()
        globals()[name] = value
        return value

    # Direct submodule access, e.g. `libcognition.database`, `libcognition.mnp`.
    if (Path(__file__).parent / f"{name}.py").exists():
        module = importlib.import_module(f".{name}", __name__)
        _suppress_logging()
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(__all__))


__all__ = [
    'database',
    'idat',
    'idat_to_data_container_mepylome',
    'MEPYLOME_KEYS',
    'tumor_db',

    # notebook_functions.py
    'plot_actual_vs_predicted_os',
    'calibrate_classifier',
    'export_classifier',
    'build_pipeline',
    'breakdown_notebook_filename',

    # calibrators.py
    'SurvivalCalibrator',

    # utils.py
    'epicv2_to_epic',

    # predictors.py:
    'EndPoint',
    'PredictorType',
    'Predictor',

    # custom_transformers.py:
    #'_logrank_feature_quantiles',
    #'_logrank_feature',
    #'fast_logrank_score_parallel',
    'fast_logrank_score_vectorized',

    'SurvivalPipeline',
    'WeibullAFTFitterWrapperExtended',
    'LogLogisticAFTFitterWrapperExtended',
    'LogNormalAFTFitterWrapperExtended',
    'CoxPHFitterWrapper',
    'CoxPHFitterWrapperExtended',
    'RandomSurvivalForestWrapperExtended',
    'DeepSurvWrapper',
    'DeepSurvWrapperExtended',
    'DeepHitWrapper',
    'DeepHitWrapperExtended',
    'PCHazardWrapper',
    'PCHazardWrapperExtended',
    'PROJECT_ROOT',
    'ASSETS_PATH',
    'GITHUB_URL',
    'DISCLAIMER',
    'HUGGINGFACE_URL',
    'DAYS_PER_YEAR',
    'MAX_FOLLOW_UP_YEARS',

]
