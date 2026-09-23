#!/usr/bin/env python3

import importlib
from pathlib import Path

import logging


def _suppress_logging():
    logging.getLogger().setLevel(logging.WARNING)
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.setLevel(logging.WARNING)


_suppress_logging()


__version__ = '1.0.4'

# Repo root: assets/ and data/ sit next to the package. Anchored on the module
# location rather than the cwd, so the data paths derived from it keep resolving
# when a caller is started from elsewhere (the CLI is run with the output dir as
# its working directory).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#ASSETS_PATH = Path(__file__).parent.parent / "assets"
ASSETS_PATH = PROJECT_ROOT / "assets"

GITHUB_URL = "https://github.com/cognition-consortium/cognition"
HUGGINGFACE_URL = "https://huggingface.co/ErasmusMC-Neuro-Oncology/cognition"

DAYS_PER_YEAR = 365.24219
MAX_FOLLOW_UP_YEARS = 26.5


# Public name -> submodule that defines it. Submodules are only imported the
# first time one of their names is actually accessed (PEP 562 module
# __getattr__), so e.g. `import libcognition; libcognition.HUGGINGFACE_URL`
# never has to load the heavy ML stack (torch, pycox, sksurv, lifelines)
# pulled in by custom_transformers/notebook_functions.
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
    'HUGGINGFACE_URL',
    'DAYS_PER_YEAR',
    'MAX_FOLLOW_UP_YEARS',

]
