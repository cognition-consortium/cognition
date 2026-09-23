"""
Tests for all Extended survival model wrappers on 4 synthetic datasets.

Purpose
-------
These are *wrapper correctness* tests, not performance benchmarks.  The goal
is to verify that every model exposes the full SurvivalPredictorExtendedBase
API (predict_risk_scores, predict_survival, predict_median, predict_all) with
correct output shapes and types.

C-index assertions are included only for the five statistical models (Cox,
Coxnet, Weibull, LogNormal, LogLogistic) on the two clean datasets (ds1, ds2),
where convergence on 50 samples is reliable.  Deep learning models (DeepSurv,
DeepHit, PCHazard) and RSF are only tested for API correctness — evaluating
their predictive performance requires much larger datasets than 50 samples.

Dataset characteristics
-----------------------
ds1 : low complexity, proportional hazards (Weibull), target C-index ~0.80
ds2 : low complexity, proportional hazards (Weibull), target C-index ~0.95
ds3 : high complexity, non-proportional hazards (log-normal AFT + non-linear /
      interaction terms); PH models will legitimately underperform
ds4 : pure noise; survival times independent of covariates, C-index ~0.50
      — evaluated on a held-out set for the statistical models to avoid
      overfitting masking the absence of signal
"""

import numpy as np
import pytest
from sksurv.metrics import concordance_index_censored

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libcognition.custom_transformers import (
    CoxPHFitterWrapperExtended,
    CoxnetSurvivalAnalysisWrapperExtended,
    WeibullAFTFitterWrapperExtended,
    LogNormalAFTFitterWrapperExtended,
    LogLogisticAFTFitterWrapperExtended,
    RandomSurvivalForestWrapperExtended,
    DeepSurvWrapperExtended,
    DeepHitWrapperExtended,
    PCHazardWrapperExtended,
    MTLRSklearnWrapperExtended,
    MTLRDeepWrapperExtended,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES  = 50
N_FEATURES = 15
SEED       = 42


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def _structured_y(times, events):
    y = np.empty(len(times), dtype=[("event", bool), ("time", float)])
    y["event"] = events.astype(bool)
    y["time"]  = times
    return y


def _censor(T, rng, target=0.08):
    """
    Apply ~target fraction of right-censoring.

    C is drawn from Uniform(q_{1 - 2*target}, T_max * 2), so roughly
    `target` proportion of observations end up censored.
    """
    q = np.quantile(T, 1.0 - 2.0 * target)
    C = rng.uniform(q, np.max(T) * 2.0, size=len(T))
    return np.minimum(T, C), (T <= C).astype(bool)


def make_ds1(seed=SEED):
    """
    Dataset 1 — low complexity, proportional hazards.

    Weibull PH model with 5 informative features (β ≈ ±0.5).
    Expected population C-index: ~0.80.
    """
    rng  = np.random.RandomState(seed)
    X    = rng.randn(N_SAMPLES, N_FEATURES).astype("float32")
    beta = np.zeros(N_FEATURES)
    beta[:5] = [0.5, 0.5, 0.5, -0.5, -0.5]

    eta = X @ beta
    T   = (-np.log(rng.uniform(size=N_SAMPLES)) * np.exp(-eta)) ** 0.5  # Weibull PH, shape=2

    times, events = _censor(T, rng)
    return X, _structured_y(times, events)


def make_ds2(seed=SEED):
    """
    Dataset 2 — low complexity, strong proportional hazards signal.

    Weibull PH model with 10 informative features (β ≈ ±0.7).
    Expected population C-index: ~0.95.
    """
    rng  = np.random.RandomState(seed)
    X    = rng.randn(N_SAMPLES, N_FEATURES).astype("float32")
    beta = np.zeros(N_FEATURES)
    beta[:10] = [0.7, 0.7, 0.7, 0.7, 0.7, -0.7, -0.7, -0.7, -0.7, -0.7]

    eta = X @ beta
    T   = (-np.log(rng.uniform(size=N_SAMPLES)) * np.exp(-eta)) ** 0.5

    times, events = _censor(T, rng)
    return X, _structured_y(times, events)


def make_ds3(seed=SEED):
    """
    Dataset 3 — high complexity, non-proportional hazards.

    Log-normal AFT model with quadratic, absolute-value, and interaction terms.
    The proportional hazards assumption is violated; Cox / Weibull PH models
    will underperform by design.  RSF and deep survival models should reach ~0.80.
    """
    rng  = np.random.RandomState(seed)
    X    = rng.randn(N_SAMPLES, N_FEATURES).astype("float32")
    beta = np.zeros(N_FEATURES)
    beta[:4] = [0.7, 0.7, -0.7, -0.7]

    eta  = X @ beta
    eta += 0.6 * X[:, 0] ** 2          # quadratic — breaks PH
    eta += 0.5 * X[:, 1] * X[:, 2]     # interaction
    eta -= 0.5 * np.abs(X[:, 3])       # non-monotonic

    log_T = -eta + 0.8 * rng.randn(N_SAMPLES)   # log-normal AFT
    T     = np.exp(log_T)

    times, events = _censor(T, rng)
    return X, _structured_y(times, events)


def make_ds4(seed=SEED):
    """
    Dataset 4 — pure noise.

    Survival times drawn from Exp(1) independently of all covariates.
    Expected C-index: ~0.50 regardless of method.
    """
    rng = np.random.RandomState(seed)
    X   = rng.randn(N_SAMPLES, N_FEATURES).astype("float32")
    T   = rng.exponential(scale=1.0, size=N_SAMPLES)

    times, events = _censor(T, rng)
    return X, _structured_y(times, events)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

# Reduced settings for deep learning models to keep the test suite fast.
# Full hyperparameter tuning is done in the notebooks.
_DL = dict(
    num_nodes  = [32, 16],
    epochs     = 100,
    patience   = 10,
    batch_size = 32,
    verbose    = False,
    seed       = SEED,
)

# All 11 Extended models — used for API-only tests
ALL_MODELS = [
    pytest.param(CoxPHFitterWrapperExtended,
                 {},
                 id="CoxPH"),
    pytest.param(CoxnetSurvivalAnalysisWrapperExtended,
                 {"fit_baseline_model": True, "l1_ratio": 0.5},
                 id="Coxnet"),
    pytest.param(WeibullAFTFitterWrapperExtended,
                 {},
                 id="WeibullAFT"),
    pytest.param(LogNormalAFTFitterWrapperExtended,
                 {},
                 id="LogNormalAFT"),
    pytest.param(LogLogisticAFTFitterWrapperExtended,
                 {},
                 id="LogLogisticAFT"),
    pytest.param(RandomSurvivalForestWrapperExtended,
                 {"n_estimators": 50, "n_jobs": 1, "verbose": 0, "random_state": SEED},
                 id="RSF"),
    pytest.param(DeepSurvWrapperExtended,   _DL,                       id="DeepSurv"),
    pytest.param(DeepHitWrapperExtended,    {**_DL, "num_durations": 20}, id="DeepHit"),
    pytest.param(PCHazardWrapperExtended,   {**_DL, "num_durations": 20}, id="PCHazard"),
    pytest.param(MTLRSklearnWrapperExtended,
                 {"num_durations": 20},
                 id="MTLRSklearn"),
    pytest.param(MTLRDeepWrapperExtended,
                 {**_DL, "num_durations": 20},
                 id="MTLRDeep"),
]

# Statistical models only — used for C-index assertions.
# Deep learning models and RSF require far more than 50 samples to produce
# reliable C-indices and are therefore excluded from performance assertions.
STAT_MODELS = [
    pytest.param(CoxPHFitterWrapperExtended,
                 {},
                 id="CoxPH"),
    pytest.param(CoxnetSurvivalAnalysisWrapperExtended,
                 {"fit_baseline_model": True, "l1_ratio": 0.5},
                 id="Coxnet"),
    pytest.param(WeibullAFTFitterWrapperExtended,
                 {},
                 id="WeibullAFT"),
    pytest.param(LogNormalAFTFitterWrapperExtended,
                 {},
                 id="LogNormalAFT"),
    pytest.param(LogLogisticAFTFitterWrapperExtended,
                 {},
                 id="LogLogisticAFT"),
    pytest.param(MTLRSklearnWrapperExtended,
                 {"num_durations": 20},
                 id="MTLRSklearn"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cindex(model, X, y):
    scores = model.predict_risk_scores(X)
    ci, *_ = concordance_index_censored(y["event"], y["time"], scores)
    return ci


def _assert_api(model, X, y):
    """Verify shapes and key presence for all SurvivalPredictorExtendedBase methods."""
    eval_times = np.percentile(y["time"], [25, 50, 75])

    # predict_risk_scores
    scores = model.predict_risk_scores(X)
    assert scores.shape == (N_SAMPLES,), f"risk_scores shape {scores.shape}"
    assert not np.isnan(scores).any(),   "risk_scores contains NaN"

    # predict_survival — without and with explicit times
    surv = model.predict_survival(X)
    assert len(surv) == N_SAMPLES

    surv_t = model.predict_survival(X, times=eval_times)
    assert len(surv_t) == N_SAMPLES

    # predict_median
    medians = model.predict_median(X)
    assert medians.shape == (N_SAMPLES,), f"median shape {medians.shape}"

    # predict_all
    result = model.predict_all(X, times=eval_times)
    assert set(result.keys()) == {"risk_scores", "survival_functions", "median"}
    assert result["risk_scores"].shape       == (N_SAMPLES,)
    assert len(result["survival_functions"]) == N_SAMPLES
    assert result["median"].shape            == (N_SAMPLES,)


# ---------------------------------------------------------------------------
# API tests — all models, all datasets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls,kwargs", ALL_MODELS)
def test_api_ds1(model_cls, kwargs):
    X, y = make_ds1()
    model = model_cls(**kwargs)
    model.fit(X, y)
    _assert_api(model, X, y)


@pytest.mark.parametrize("model_cls,kwargs", ALL_MODELS)
def test_api_ds2(model_cls, kwargs):
    X, y = make_ds2()
    model = model_cls(**kwargs)
    model.fit(X, y)
    _assert_api(model, X, y)


@pytest.mark.parametrize("model_cls,kwargs", ALL_MODELS)
def test_api_ds3(model_cls, kwargs):
    X, y = make_ds3()
    model = model_cls(**kwargs)
    model.fit(X, y)
    _assert_api(model, X, y)


@pytest.mark.parametrize("model_cls,kwargs", ALL_MODELS)
def test_api_ds4(model_cls, kwargs):
    X, y = make_ds4()
    model = model_cls(**kwargs)
    model.fit(X, y)
    _assert_api(model, X, y)


# ---------------------------------------------------------------------------
# C-index tests — statistical models only, ds1 / ds2 / ds4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls,kwargs", STAT_MODELS)
def test_cindex_ds1(model_cls, kwargs):
    """Low complexity signal: C-index > 0.60 on training data."""
    X, y = make_ds1()
    model = model_cls(**kwargs)
    model.fit(X, y)

    ci = _cindex(model, X, y)
    assert ci > 0.60, f"{model_cls.__name__}: C-index {ci:.3f} on ds1 (expected > 0.60)"


@pytest.mark.parametrize("model_cls,kwargs", STAT_MODELS)
def test_cindex_ds2(model_cls, kwargs):
    """Strong signal: C-index > 0.75 on training data."""
    X, y = make_ds2()
    model = model_cls(**kwargs)
    model.fit(X, y)

    ci = _cindex(model, X, y)
    assert ci > 0.75, f"{model_cls.__name__}: C-index {ci:.3f} on ds2 (expected > 0.75)"


@pytest.mark.parametrize("model_cls,kwargs", STAT_MODELS)
def test_cindex_ds4(model_cls, kwargs):
    """
    Pure noise: C-index < 0.70 on a held-out draw.

    A held-out set is used (different seed, same generative process) so that
    any training-set overfitting does not mask the absence of signal.
    """
    X_train, y_train = make_ds4(seed=SEED)
    X_test,  y_test  = make_ds4(seed=SEED + 1)

    model = model_cls(**kwargs)
    model.fit(X_train, y_train)

    ci = _cindex(model, X_test, y_test)
    assert ci < 0.70, (
        f"{model_cls.__name__}: C-index {ci:.3f} on held-out ds4 (expected < 0.70)"
    )
