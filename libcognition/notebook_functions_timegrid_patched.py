#!/usr/bin/env python3

import datetime
import itertools
import joblib
import json
import logging
import os
import re

from beartype import beartype
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sksurv.metrics import integrated_brier_score, brier_score, concordance_index_censored
#from sksurv.metrics import concordance_index_censored

# scipy >= 1.14 removed the deprecated `simps` (renamed to `simpson`); pycox's
# EvalSurv.integrated_brier_score still calls `scipy.integrate.simps`. The two
# are numerically identical, so we restore the old alias for compatibility.
import scipy.integrate as _scipy_integrate
if not hasattr(_scipy_integrate, "simps"):
    _scipy_integrate.simps = _scipy_integrate.simpson

from pycox.evaluation import EvalSurv

from libcognition.custom_transformers import *
from libcognition.calibrators import *
from libcognition.predictors import EndPoint, PredictorType

from . import ASSETS_PATH, __version__


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys handled by the pipeline builder; all other keys in param_grid are
# forwarded as keyword arguments directly to the classifier constructor.
_PIPELINE_PARAMS     = {"var_filter", "n_components", "select_k_best", "select_k_best_components"}
_DEFAULT_TUMOR_TYPES = ["Oligodendroglioma", "Astrocytoma", "Overall"]

# Default values for pipeline params; used by _normalize_params to ensure
# every output row always contains all columns regardless of which params
# were explicitly passed.
#
# Pipeline step order (each step only added when its param is not None):
#   var_filter  → SelectKBest(variance, k=var_filter)   — keeps top-k by variance
#   select      → SelectKBest(fast_logrank_score_vectorized, k=select_k_best)
#   scaler      → StandardScaler  (always)
#   pca         → PCA(n_components, random_state=42)
#   scaler_pca  → StandardScaler  (only when pca is added)
#   select_pca  → SelectKBest(fast_logrank_score_vectorized, k=select_k_best_components)
#   cox         → classifier
#
# select_k_best and n_components are independent; both, either, or neither
# can be active simultaneously.
#
# select_k_best_components selects among the PCA components rather than the raw
# features: PCA orders by variance, which need not coincide with prognostic
# value, so a low-variance component may still carry signal. It therefore
# requires n_components to be set.
_PARAM_DEFAULTS: dict = {
    "var_filter":              None,  # None → no variance pre-filter; int → keep top-k by variance
    "select_k_best":           None,  # None → no SelectKBest step; int → keep top-k by logrank
    "n_components":            None,  # None → no PCA step
    "select_k_best_components": None, # None → no post-PCA SelectKBest step; int → keep top-k components by logrank
}

# Metrics collected per fold per tumor type.
# Regression metrics are computed on event-only patients using
# expected_survival (trapezoid AUC over the observed time window) as
# the predicted value. Note that expected_survival systematically
# underestimates true survival time for patients with long follow-up
# (truncation bias); this is corrected post-hoc via the LOWESS calibrator
# rather than during CV.
_METRIC_KEYS = (
    "c_index_sksurv", "c_index_pycox",
    "brier_sksurv",
    "brier_pycox",
    "mae_linear",     "mae_log1p",
    "slope_linear",   "slope_log1p",
    "r2_linear",      "r2_log1p",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@beartype
def _derive_name(classifier_class: type) -> str:
    raw        = classifier_class.__name__
    normalized = re.sub(r"_+", "__", raw)
    return normalized[0].upper() + normalized[1:]


@beartype
def _normalize_params(params: dict) -> dict:
    """
    Fill missing pipeline params with their defaults so that every combo row
    has the same columns in the output CSV.
    """
    out = dict(_PARAM_DEFAULTS)
    out.update(params)
    return out


@beartype
def _nan_row(combo_tag, params: dict, tt: str, n_folds: int = 0) -> dict:
    row = {
        "combo_tag":  combo_tag,
        **_normalize_params(params),
        "tumor_type": tt,
        "n_folds":    n_folds,
    }
    for k in _METRIC_KEYS:
        row[f"{k}_mean"] = np.nan
        row[f"{k}_std"]  = np.nan
    return row


@beartype
def _combo_tag(params: dict) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, type):
            v_str = v.__name__
        elif isinstance(v, (list, tuple)):
            v_str = "-".join(str(x) for x in v)
        else:
            v_str = str(v)
        v_str = re.sub(r"[^\w.\-]", "_", v_str)
        parts.append(f"{k}={v_str}")
    return "__".join(parts)



@beartype
def _regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """
    Calibration and error metrics between actual and predicted survival time.
    Only call with arrays already filtered to event == 1.

    Metrics
    -------
    mae_linear / mae_log1p
        Mean absolute error on raw and log1p-transformed scale.
        Lower is better; log1p reduces the influence of long survivors.
    slope_linear / slope_log1p
        OLS slope of ``actual ~ predicted`` (intercept included).
        slope = 1  → perfectly calibrated.
        slope < 1  → model over-spreads predictions relative to actual values.
        slope > 1  → model under-spreads (predictions cluster near the mean).
    r2_linear / r2_log1p
        Coefficient of determination on the respective scale.
    """
    _KEYS = (
        "mae_linear", "mae_log1p",
        "slope_linear", "slope_log1p",
        "r2_linear", "r2_log1p",
    )
    if len(actual) < 2:
        return {k: np.nan for k in _KEYS}

    pred_clipped = np.clip(predicted, 0, None)

    def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Return (slope, R²) for OLS fit y ~ x."""
        slope, intercept = np.polyfit(x, y, 1)
        y_hat  = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(slope), r2

    diff_lin = actual - predicted
    diff_log = np.log1p(actual) - np.log1p(pred_clipped)

    slope_lin, r2_lin = _ols(predicted,              actual)
    slope_log, r2_log = _ols(np.log1p(pred_clipped), np.log1p(actual))

    return {
        "mae_linear":   float(np.mean(np.abs(diff_lin))),
        "mae_log1p":    float(np.mean(np.abs(diff_log))),
        "slope_linear": slope_lin,
        "slope_log1p":  slope_log,
        "r2_linear":    r2_lin,
        "r2_log1p":     r2_log,
    }


@beartype
def _make_eval_times(
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_times: int = 50,
    eval_time_range: tuple[float | int, float | int] | None = None,
) -> np.ndarray | None:
    """Build an IBS grid strictly inside supported train/test follow-up.

    With ``eval_time_range=None``, the grid is allocated over the complete
    overlap of the current training and test fold.  This is the intended
    behaviour of the direct notebook, but returns exactly ``n_times`` points
    rather than its accidental 49/50 off-by-one result.

    With ``eval_time_range=(t0, tau)``, that exact interval is required.  A
    fold that cannot support it returns ``None`` rather than silently changing
    the interval, which keeps IBS comparable across LODO folds.
    """
    if n_times < 2:
        raise ValueError("n_times must be at least 2")

    train_times = np.asarray(y_train["time"], dtype=np.float64)
    test_times = np.asarray(y_test["time"], dtype=np.float64)
    train_times = train_times[np.isfinite(train_times)]
    test_times = test_times[np.isfinite(test_times)]

    if train_times.size == 0 or test_times.size == 0:
        return None

    valid_lower = max(float(train_times.min()), float(test_times.min()))
    valid_upper = min(float(train_times.max()), float(test_times.max()))

    if eval_time_range is None:
        lower, upper = valid_lower, valid_upper
    else:
        lower, upper = map(float, eval_time_range)
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise ValueError(
                "eval_time_range must be a finite (t0, tau) pair with t0 < tau"
            )
        if lower < valid_lower or upper > valid_upper:
            return None

    # Both libraries require evaluation inside observed follow-up, not on the
    # upper boundary. Move each endpoint one floating-point step inward.
    lower = float(np.nextafter(lower, np.inf))
    upper = float(np.nextafter(upper, -np.inf))
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return None

    return np.linspace(lower, upper, num=n_times, endpoint=True, dtype=np.float64)


@beartype
def _truncate_survival_at_tau(y: np.ndarray, tau: float) -> np.ndarray:
    """Administratively censor observations after ``tau``.

    This is used only for scikit-survival's IPCW metric.  It leaves survival
    status unchanged at every evaluation time strictly below ``tau`` while
    ensuring that test follow-up does not extend beyond training support.
    """
    y_out = y.copy()
    after_tau = np.asarray(y_out["time"], dtype=np.float64) > float(tau)
    y_out["time"][after_tau] = float(tau)
    y_out["event"][after_tau] = False
    return y_out


@beartype
def _eval_tumor_type(
    tt: str,
    mask: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray,
    surv_preds: Union[list, np.ndarray],
    risk_scores: np.ndarray,
    expected_survival: np.ndarray,
    fold_num: int,
    global_times: np.ndarray | None = None,
    n_eval_times: int = 50,
    eval_time_range: tuple[float | int, float | int] | None = None,
) -> dict | None:
    """
    Compute C-index (sksurv + pycox), Brier, and regression metrics for one
    tumor-type subset.  Returns None and prints a warning when skipped.
    """
    if not mask.any():
        print(f"  ⚠ No {tt} samples in fold {fold_num}")
        return None

    y_sub      = y_test[mask]
    rs_sub     = risk_scores[mask]
    es_sub     = expected_survival[mask]
    event_mask = y_sub["event"].astype(bool)

    if not event_mask.any():
        print(f"  ⚠ No events for {tt} in fold {fold_num}, skipping metrics")
        return None

    # Backward compatibility for callers that still pass the old 50-point
    # global grid: keep its requested count, but allocate those points over
    # this fold/subgroup's actual supported interval.
    if global_times is not None:
        n_eval_times = int(len(global_times))

    times_sub = _make_eval_times(
        y_train=y_train,
        y_test=y_sub,
        n_times=n_eval_times,
        eval_time_range=eval_time_range,
    )
    if times_sub is None:
        requested = (
            "fold-specific overlap"
            if eval_time_range is None
            else f"fixed interval {eval_time_range}"
        )
        print(
            f"  ⚠ No valid IBS grid for {tt} in fold {fold_num} "
            f"using {requested}; skipping"
        )
        return None

    print(
        f"  [CV{fold_num}] {tt}: train time range = {y_train['time'].min()}-{y_train['time'].max()}, "
        f"test time range = {y_test['time'].min()}-{y_test['time'].max()}"
    )

    surv_preds_sub = [surv_preds[i] for i in np.flatnonzero(mask)]
    surv_matrix_sub = np.asarray(
        [sf(times_sub) for sf in surv_preds_sub],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(surv_matrix_sub)):
        print(f"  ⚠ Non-finite survival probabilities for {tt} in fold {fold_num}")
        return None
    surv_matrix_sub = np.clip(surv_matrix_sub, 0.0, 1.0)

    print(
        f"  [CV{fold_num}] {tt}: IBS grid n={len(times_sub)}, "
        f"range={times_sub[0]:.1f}-{times_sub[-1]:.1f}"
    )

    c_idx_sksurv = concordance_index_censored(event_mask, y_sub["time"], rs_sub)[0]

    surv_df = pd.DataFrame(surv_matrix_sub.T, index=times_sub).sort_index()
    ev = EvalSurv(
        surv=surv_df,
        durations=y_sub["time"],
        events=event_mask,
        censor_surv="km",
    )

    T_MAX = 6500

    sample_mask         = y_sub["time"] <= T_MAX
    y_sub_clipped       = y_sub[sample_mask]
    _t_clipped_stop     = min(y_sub_clipped["time"].max(), y_train["time"].max()) - 0.5
    time_mask           = times_sub <= _t_clipped_stop
    times_clipped       = times_sub[time_mask]
    surv_clipped        = surv_matrix_sub[np.ix_(sample_mask, time_mask)]

    # For an evaluation ending at tau, subjects observed beyond tau are known
    # to be event-free through tau. Administrative censoring at tau therefore
    # preserves the target while satisfying sksurv's IPCW support requirement.
    tau = (
        float(eval_time_range[1])
        if eval_time_range is not None
        else min(float(y_train["time"].max()), float(y_sub["time"].max()))
    )
    y_sub_sksurv = _truncate_survival_at_tau(y_sub, tau=tau)

    return {
        "c_index_sksurv":        c_idx_sksurv,
        "c_index_pycox":         ev.concordance_td(),
        "brier_sksurv":          float(integrated_brier_score(y_train, y_sub_sksurv, surv_matrix_sub, times_sub)),
        #"brier_sksurv_leq_6500": float(integrated_brier_score(y_train, y_sub_clipped, surv_clipped, times_clipped)),
        "brier_pycox":           float(ev.integrated_brier_score(times_sub)),
        "n_samples":             int(mask.sum()),
        **_regression_metrics(
            actual=y_sub["time"][event_mask],
            predicted=es_sub[event_mask],
        ),
    }


@beartype
def _empty_fold_results(tumor_types: list[str]) -> dict:
    return {tt: {k: [] for k in _METRIC_KEYS} for tt in tumor_types}


@beartype
def _accumulate(results: dict, tt: str, fold_metrics: dict) -> None:
    """
    Append per-fold metric values to the running lists in *results*.

    *results* has the shape produced by ``_empty_fold_results``:
    ``{tumor_type: {metric_name: [fold1_value, fold2_value, ...]}}``

    Called once per tumor type per completed fold so that after all folds
    ``results[tt][metric]`` is a list of length n_completed_folds, ready
    for ``np.mean`` / ``np.std`` aggregation.
    """
    for k in _METRIC_KEYS:
        results[tt][k].append(fold_metrics[k])


@beartype
def _median_expected_survival(surv_matrix: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median (first time the curve crosses 0.5) and expected (trapezoid AUC) survival."""
    argmin_idx        = np.argmin(np.abs(surv_matrix - 0.5), axis=1)
    median_survival   = np.where(surv_matrix.min(axis=1) > 0.5, np.nan, times[argmin_idx])
    expected_survival = np.trapezoid(surv_matrix, times, axis=1)
    return median_survival, expected_survival


@beartype
def _run_fold(
    pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    m_test: pd.DataFrame,
    tumor_types: list[str],
    output_file: str,
    fold_num: int,
    global_times: np.ndarray | None = None,
    n_eval_times: int = 50,
    eval_time_range: tuple[float | int, float | int] | None = None,
    verbose: bool = True,
    dataset_name: str | None = None,
) -> dict[str, dict[str, float]] | None:
    """
    Fit *pipeline*, predict on the test fold, write per-patient TSV rows, and
    compute all metrics per tumor type.

    Returns a dict ``{tumor_type: {metric: value}}`` or None if time range is
    invalid.
    """
    if global_times is not None:
        n_eval_times = int(len(global_times))

    times = _make_eval_times(
        y_train=y_train,
        y_test=y_test,
        n_times=n_eval_times,
        eval_time_range=eval_time_range,
    )
    if times is None:
        requested = (
            "fold-specific overlap"
            if eval_time_range is None
            else f"fixed interval {eval_time_range}"
        )
        print(
            f"  ⚠ No valid IBS grid in fold {fold_num} "
            f"using {requested}; skipping"
        )
        return None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pipeline = clone(pipeline)
    pipeline.fit(X_train, y_train)

    # Debug: check transformed data validity
    X_train_transformed = pipeline[:-1].transform(X_train) if len(pipeline.steps) > 1 else X_train
    X_test_transformed = pipeline[:-1].transform(X_test) if len(pipeline.steps) > 1 else X_test

    if verbose and np.any(~np.isfinite(X_test_transformed)):
        n_nan = np.isnan(X_test_transformed).sum()
        n_inf = np.isinf(X_test_transformed).sum()
        print(f"  ⚠ X_test after preprocessing: {n_nan} NaN, {n_inf} Inf values!")
        if n_nan + n_inf == X_test_transformed.size:
            print(f"  ⚠ All {X_test_transformed.size} values are invalid! Skipping fold.")
            return None

    if verbose and "var_filter" in pipeline.named_steps:
        vf     = pipeline.named_steps["var_filter"]
        n_kept = sum(vf.get_support())
        n_tot  = len(vf.get_support())
        print(f"\nFold {fold_num}: kept {n_kept:,}/{n_tot:,} features ({100*n_kept/n_tot:.1f}%)")

    # Keep each model's native survival curve, then evaluate that curve on the
    # selected IBS grid. This avoids resampling an already-resampled curve.
    predictions = pipeline.predict_all(X_test, times=None)
    surv_preds = predictions["survival_functions"]
    surv_matrix = np.asarray([sf(times) for sf in surv_preds], dtype=np.float64)

    if verbose:
        if np.any(~np.isfinite(surv_matrix)):
            n_nan = np.isnan(surv_matrix).sum()
            n_inf = np.isinf(surv_matrix).sum()
            print(f"  ⚠ surv_matrix has {n_nan} NaN, {n_inf} Inf values")
            if "risk_score" in predictions:
                risk_nan = np.isnan(predictions["risk_scores"]).sum()
                print(f"  ⚠ risk_scores has {risk_nan} NaN values")
        print(f"  surv_matrix shape={surv_matrix.shape}  "
              f"std/patient={surv_matrix.std(axis=1).mean():.4f}  "
              f"range=[{surv_matrix.min():.3f}, {surv_matrix.max():.3f}]")

    median_survival, expected_survival = _median_expected_survival(surv_matrix, times)

    export_df = pd.DataFrame({
        #"Sentrix_ID":        m_test["Sentrix ID"].values,
        "Tumor_Type":        m_test["Tumor Type"].values,
        "actual_time":       y_test["time"],
        "actual_event":      y_test["event"],
        "risk_score":        predictions["risk_scores"],
        "median_survival":   median_survival,
        "expected_survival": expected_survival,
        "fold":              fold_num,
        "dataset":           dataset_name,
    })
    export_df.to_csv(
        output_file, mode="a", header=(fold_num == 1),
        index=False, sep="\t", float_format="%.4f",
    )

    fold_metrics: dict[str, dict[str, float]] = {}
    for tt in tumor_types:
        mask = (
            np.ones(len(m_test), dtype=bool) if tt == "Overall"
            else (m_test["Tumor Type"].astype(str) == tt).values
        )
        
        result = _eval_tumor_type(
            tt=tt, mask=mask,
            y_test=y_test, y_train=y_train,
            surv_preds=surv_preds,
            risk_scores=predictions["risk_scores"],
            expected_survival=expected_survival,
            fold_num=fold_num,
            global_times=None,
            n_eval_times=n_eval_times,
            eval_time_range=eval_time_range,
        )

        if result is not None:
            fold_metrics[tt] = result
            
            print(
                f"  [CV{fold_num}] {tt:18s}  n={result['n_samples']:3d}  "
                f"C-idx(pycox)={result['c_index_pycox']:.3f}  "
                f"IBS sksurv={result['brier_sksurv']:.3f}  "
                #f"IBS sksurv ≤ 6500={result['brier_sksurv_leq_6500']:.3f}  "
                #f"MAE={result['mae_linear']:.1f}d  "
                #f"slope={result['slope_linear']:.2f}  "
                #f"R²={result['r2_linear']:.3f}"
            )

    return fold_metrics


@beartype
def _print_summary(results: dict, tumor_types: list[str], ibs_output_file: str | None = None) -> None:
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    for tt in tumor_types:
        r = results[tt]
        if not r["c_index_sksurv"]:
            print(f"\n{tt}: No data")
            continue
        print(f"\n{tt.upper()}")
        print("-" * 40)
        for k in _METRIC_KEYS:
            vals = r[k]
            print(f"  {k:22s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 80)

    if ibs_output_file:
        if os.path.dirname(ibs_output_file):
            os.makedirs(os.path.dirname(ibs_output_file), exist_ok=True)
        with open(ibs_output_file, "w") as f:
            for tt in tumor_types:
                r = results[tt]
                if r["brier_pycox"]:
                    vals = r["brier_pycox"]
                    f.write(f"{tt}\tbrier_pycox\t{np.mean(vals):.4f}\n")
                    f.write(f"{tt}\tbrier_pycox_std\t{np.std(vals):.4f}\n")
        print(f"IBS exported to: {ibs_output_file}")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def _variance_score(X, y):
    """Score function for SelectKBest that ranks features by variance (ignores y)."""
    return np.var(X, axis=0), np.ones(X.shape[1])


@beartype
def build_pipeline(classifier_class: type, params: dict) -> "SurvivalPipeline":
    """
    Construct a SurvivalPipeline from a flat params dict.

    Pipeline step order (each step only included when its param is not None):
        var_filter  → SelectKBest(variance, k=var_filter)   — keeps top-k by variance
        select      → SelectKBest(fast_logrank_score_vectorized, k=select_k_best)
        scaler      → StandardScaler
        pca         → PCA(n_components=n_components, random_state=42)
        scaler_pca  → StandardScaler  (only when pca is included)
        select_pca  → SelectKBest(fast_logrank_score_vectorized, k=select_k_best_components)
        cox         → classifier_class(**remaining_kwargs)

    select_k_best and n_components are independent — both, either, or neither
    can be active simultaneously.

    select_k_best_components selects among the PCA components by logrank score
    instead of taking the leading ones, since PCA orders by variance and a
    low-variance component may still be prognostic. It requires n_components.
    """
    p          = _normalize_params(params)
    vf_k       = p["var_filter"]
    select_k   = p["select_k_best"]
    n_comp     = p["n_components"]
    select_k_c = p["select_k_best_components"]
    clf_kwargs = {k: v for k, v in params.items() if k not in _PIPELINE_PARAMS}

    if select_k_c is not None:
        if n_comp is None:
            raise ValueError(
                "select_k_best_components requires n_components: there are no "
                "PCA components to select from when the pca step is absent."
            )
        if select_k_c > n_comp:
            raise ValueError(
                f"select_k_best_components ({select_k_c}) exceeds n_components ({n_comp})."
            )

    steps = []
    if vf_k is not None:
        steps.append(("var_filter", SelectKBest(score_func=_variance_score, k=vf_k)))
    if select_k is not None:
        steps.append(("select", SelectKBest(score_func=fast_logrank_score_vectorized, k=select_k)))
    steps.append(("scaler", StandardScaler()))
    if n_comp is not None:
        steps += [
            ("pca",        PCA(n_components=n_comp, random_state=42)),
            ("scaler_pca", StandardScaler()),
        ]
    if select_k_c is not None:
        steps.append(("select_pca", SelectKBest(score_func=fast_logrank_score_vectorized, k=select_k_c)))
    steps.append(("cox", classifier_class(**clf_kwargs)))
    return SurvivalPipeline(steps)


@beartype
def predict_curves_median_expected(pipeline, X: np.ndarray, times: np.ndarray | None = None) -> pd.DataFrame:
    """
    Predict survival curves for X, plus their median and expected (trapezoid AUC) survival.

    times: shared evaluation grid. Defaults to the model's own time grid
    (the ``.x`` of the first predicted curve) when not given.
    """
    curves = pipeline.predict_survival(X, times=times)
    t = curves[0].x if times is None else times

    surv_matrix = np.array([sf(t) for sf in curves])
    median_survival, expected_survival = _median_expected_survival(surv_matrix, t)

    return pd.DataFrame({
        "curve":             curves,
        "median_survival":   median_survival,
        "expected_survival": expected_survival,
    })


@beartype
def plot_calibration_comparison(oof, calibrations,
                                predicted_cols   = ("median_survival", "expected_survival"),
                                actual_time_col  = "actual_time",
                                event_col        = "actual_event",
                                classifier_name  = "Model"):

    methods    = ["raw", "linear", "lowess"]
    col_titles = ["No calibration", "Linear", "LOWESS"]

    event_mask = oof[event_col].astype(bool)

    palette_actual = {True: "steelblue", False: "lightgrey"}
    labels         = {True: "Event",     False: "Censored"}

    n_rows = len(predicted_cols) * 2
    n_cols = len(methods)

    fig, axes = plt.subplots(
        nrows   = n_rows,
        ncols   = n_cols,
        figsize = (15, len(predicted_cols) * 7),
        gridspec_kw = {"hspace": 0.45, "wspace": 0.3},
    )

    for row_block, survival_col in enumerate(predicted_cols):
        actual_log = np.log1p(oof[actual_time_col].values)

        for col_idx, method in enumerate(methods):

            if method == "raw":
                pred_log = np.log1p(oof[survival_col].values)
            else:
                cal      = SurvivalCalibrator.load(calibrations[survival_col][method]["path"])
                pred_log = np.log1p(cal.transform(oof[survival_col].values))

            resid = pred_log - actual_log

            finite_mask = event_mask & np.isfinite(pred_log) & np.isfinite(actual_log)

            ax_actual = axes[row_block * 2,     col_idx]
            ax_resid  = axes[row_block * 2 + 1, col_idx]

            for event_val in [False, True]:
                m = (event_mask == event_val) & np.isfinite(pred_log) & np.isfinite(actual_log)
                ax_actual.scatter(actual_log[m], pred_log[m],
                                  color=palette_actual[event_val],
                                  alpha=0.5, s=20,
                                  label=labels[event_val],
                                  zorder=2 if event_val else 1)

            ea = actual_log[finite_mask]
            ep = pred_log[finite_mask]
            s, i, r, *_ = stats.linregress(ea, ep)

            pad    = (ea.max() - ea.min()) * 0.05
            lims   = [ea.min() - pad, ea.max() + pad]
            x_line = np.linspace(lims[0], lims[1], 200)

            ax_actual.plot(lims, lims, 'k--', lw=1, alpha=0.4, label="Identity")
            ax_actual.plot(x_line, s * x_line + i, color="black", lw=1.5,
                           label=f"y = {s:.2f}x + {i:.2f}  (R²={r**2:.3f})")

            ax_actual.set_xlabel("log1p(Actual time)")
            ax_actual.set_ylabel("log1p(Predicted)")
            ax_actual.legend(fontsize=7)

            ax_resid.scatter(actual_log[finite_mask], resid[finite_mask],
                             color="tomato", alpha=0.5, s=20, label="Event")
            ax_resid.axhline(0, color='k', lw=1, linestyle='--', alpha=0.4)

            sort_idx = np.argsort(ea)
            smoothed = lowess(resid[finite_mask][sort_idx], ea[sort_idx], frac=0.5)
            ax_resid.plot(smoothed[:, 0], smoothed[:, 1], color="darkred", lw=1.5,
                          label=f"LOWESS  bias={resid[finite_mask].mean():+.3f}")

            ax_resid.set_xlabel("log1p(Actual time)")
            ax_resid.set_ylabel("log1p(Pred) − log1p(Actual)")
            ax_resid.legend(fontsize=7)

            if row_block == 0:
                ax_actual.set_title(col_titles[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax_actual.set_ylabel(f"[{survival_col}]\nlog1p(Predicted)")
                ax_resid.set_ylabel(f"[{survival_col}]\nResidual (log1p)")

    fig.suptitle(f"Calibration comparison — {classifier_name}", fontsize=13, y=1.01)
    plt.show()


@beartype
def plot_actual_vs_predicted_os(classifier_name: str):
    dat = (
        pd.read_csv("data/database.txt", sep="\t")
        .pipe(lambda x: x[x["Sentrix ID"] != ""])
        .pipe(lambda x: x[(x["Discarded"] == "") | (x["Discarded"].isna())])
    )

    pred = (
        pd.read_csv("output/cv/cv__" + classifier_name + ".csv", sep='\t')
        .assign(fold=lambda x: 'fold-' + x['fold'].astype(str))
    )

    plt_df = (
        dat.merge(pred, left_on="Sentrix ID", right_on="Sentrix_ID")
        .query("`Survival event` == 1 or `Survival event` == '1'")
        .dropna(subset=["median_survival", "expected_survival"])
        .assign(**{"Survival days": lambda x: pd.to_numeric(x["Survival days"])})
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    plots = [
        ("Oligodendroglioma", "median_survival",   "Median Survival",   axes[0, 0]),
        ("Astrocytoma",       "median_survival",   "Median Survival",   axes[0, 1]),
        ("Oligodendroglioma", "expected_survival",  "Expected Survival", axes[1, 0]),
        ("Astrocytoma",       "expected_survival",  "Expected Survival", axes[1, 1]),
    ]

    for tumor_type, y_var, y_label, ax in plots:
        data = plt_df[plt_df["Tumor Type"] == tumor_type]

        sns.scatterplot(data=data, x="Survival days", y=y_var, hue="fold", ax=ax)

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'gray', linestyle='--', alpha=0.5, zorder=0, linewidth=1)

        r, p = spearmanr(data["Survival days"], data[y_var])
        ax.text(
            0.05, 0.95,
            f"Spearman r={r:.2f}\np={p:.2g}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.set_title(f"{tumor_type}")
        ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.show()

    return plt


@beartype
def calibrate_classifier(classifier_name,
                         cv_results_file,
                         array_types,
                         n_samples,
                         endpoint: EndPoint,
                         classifier_algorithm: PredictorType):

    calibrations = {
        "expected_survival": {},
        "median_survival":   {},
    }

    oof             = pd.read_csv(cv_results_file, sep='\t')
    date_str        = datetime.datetime.now().strftime('%Y-%m-%d')
    array_types_str = '_'.join(sorted(set(array_types)))

    # Same v{version}/{endpoint}/{algorithm}/ layout as export_classifier, so
    # calibrations live alongside the model they belong to. The stored "path"
    # is relative to ASSETS_PATH/models/, matching what SurvivalCalibrator.load()
    # expects (it prepends ASSETS_PATH/models/ itself).
    relative_basedir = Path(f"v{__version__}") / endpoint.value / classifier_algorithm.value
    basedir           = Path(ASSETS_PATH) / "models" / relative_basedir
    basedir.mkdir(parents=True, exist_ok=True)

    for survival_col in calibrations:
        for method in ["linear", "lowess"]:
            cal_id        = f"cal-{survival_col}-{method}"
            filename      = f"{date_str}__{array_types_str}__n{n_samples}__{cal_id}.json"
            relative_path = relative_basedir / filename
            full_path     = basedir / filename

            cal = SurvivalCalibrator(method=method, frac=0.75)
            cal.fit(oof[survival_col], oof["actual_time"], oof["actual_event"])
            
            print(f"Exporting calibration to {full_path}")
            cal.save(str(full_path))

            calibrations[survival_col][method] = {
                "calibration-id" : cal_id,
                "classifier"     : classifier_name,
                "date"           : date_str,
                "array_types"    : array_types_str,
                "n"              : n_samples,
                "path"           : str(relative_path),
            }

    return calibrations


@beartype
def breakdown_notebook_filename(name: str) -> tuple[EndPoint, PredictorType]:
    params = name.replace("predict_", "", 1).split("_")

    e = EndPoint.from_string(" ".join(params[:-1]))
    p = PredictorType.from_string(params[-1])

    return(e, p)



@beartype
def export_classifier(
    pipeline,
    endpoint: EndPoint,
    classifier_algorithm: PredictorType,
    feature_names,
    array_types: list,
    n_samples: int,
    artifacts: dict | None = None,
    performance: dict | None = None,
    calibrations: dict | None = None,
    grade_only_results: pd.DataFrame | None = None,
) -> str:
    endpoint_value  = endpoint.value
    algorithm_value = classifier_algorithm.value

    array_types_str = "_".join(sorted(set(array_types)))

    now          = datetime.datetime.now()
    date_str     = now.strftime('%Y-%m-%d')
    datetime_str = now.strftime('%Y-%m-%d %H:%M:%S')
    base_name    = f"{date_str}__{array_types_str}__n{n_samples}"

    relative_model_basedir = Path(f"v{__version__}") / endpoint_value / algorithm_value
    model_basedir           = Path(ASSETS_PATH) / "models" / relative_model_basedir
    model_basedir.mkdir(parents=True, exist_ok=True)

    model_path           = model_basedir / f"{base_name}.joblib"
    meta_path             = model_basedir / f"{base_name}.info.json"
    relative_model_path   = relative_model_basedir / f"{base_name}.joblib"

    data_to_save = {'pipeline': pipeline, **(artifacts or {})}
    print(f"Saving model to: {model_path}")
    joblib.dump(data_to_save, model_path)

    meta = {
        'model':         str(relative_model_path),
        'endpoint':      endpoint_value,
        'predictiontype': algorithm_value,
        'version':          f'v{__version__}',
        'date':          datetime_str,
        "trained on":    {'total': n_samples},
        "applicable on": {
            "450k":   '450k'   in array_types,
            "epic":   'epic'   in array_types,
            "epicv2": 'epicv2' in array_types,
        },
        'features': feature_names,
    }
    if performance is not None:
        meta["performance"] = performance
    if calibrations is not None:
        meta["calibrations"] = calibrations
    if grade_only_results is not None:
        meta["grade_only_results"] = grade_only_results.drop(columns=["curve"]).to_dict(orient="records")

    print(f"Saving metadata to: {meta_path}")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return str(model_path)


@beartype
def run_cross_validation(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    cv_output_file: str,
    tumor_types: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    ibs_output_file: str | None = None,
) -> dict:
    """
    Stratified K-fold cross-validation for a survival pipeline.

    Returns
    -------
    dict
        ``results[tumor_type][metric]`` — per-fold lists.
        See ``_METRIC_KEYS`` for the full set of metrics.
        Regression metrics (mae, slope, r2) are computed on event-only
        patients using expected_survival (trapezoid AUC) as the predicted
        value. Truncation bias in expected_survival is not corrected here;
        use the LOWESS calibrator post-hoc instead.
    """
    if tumor_types is None:
        tumor_types = _DEFAULT_TUMOR_TYPES

    kf           = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results      = _empty_fold_results(tumor_types)
    global_times = np.linspace(y["time"].min(), y["time"].max() - 0.5, 50)

    if os.path.exists(cv_output_file):
        os.remove(cv_output_file)

    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        fold_metrics = _run_fold(
            pipeline=pipeline,
            X_train=X[train_idx], X_test=X[test_idx],
            y_train=y[train_idx], y_test=y[test_idx],
            m_test=metadata.iloc[test_idx],
            tumor_types=tumor_types,
            output_file=cv_output_file,
            fold_num=fold_num,
            global_times=global_times,
            verbose=True,
        )
        if fold_metrics is None:
            continue
        for tt, metrics in fold_metrics.items():
            _accumulate(results, tt, metrics)

    _print_summary(results, tumor_types, ibs_output_file=ibs_output_file)
    print(f"\nPredictions exported to: {cv_output_file}")
    print(f"Total samples: {len(pd.read_csv(cv_output_file, sep=chr(9)))}")
    return results


@beartype
def run_cross_validation_lodo(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    cv_output_file: str,
    tumor_types: list[str] | None = None,
    min_test_samples: int = 15,
    ibs_output_file: str | None = None,
    n_eval_times: int = 50,
    eval_time_range: tuple[float | int, float | int] | None = None,
) -> dict:
    """
    Leave-One-Dataset-Out (LODO) cross-validation for a survival pipeline.

    For each dataset in ``metadata['Dataset']`` with at least
    ``min_test_samples`` samples, that dataset is held out as test set and
    the remaining samples are used for training.  Datasets below the
    threshold are never used as test set (they remain in the training set
    of every fold).

    Returns
    -------
    dict
        ``results[tumor_type][metric]`` — per-fold lists, identical
        structure to ``run_cross_validation``.
    """
    if tumor_types is None:
        tumor_types = _DEFAULT_TUMOR_TYPES

    results = _empty_fold_results(tumor_types)

    if os.path.exists(cv_output_file):
        os.remove(cv_output_file)

    all_datasets = metadata["Dataset"].unique()
    eligible     = [
        ds for ds in sorted(all_datasets)
        if (metadata["Dataset"] == ds).sum() >= min_test_samples
    ]

    print(
        f"LODO: {len(eligible)}/{len(all_datasets)} datasets "
        f"meet n >= {min_test_samples} and will be used as test set"
    )

    for fold_num, dataset in enumerate(eligible, start=1):
        test_mask = (metadata["Dataset"] == dataset).values
        train_idx = np.where(~test_mask)[0]
        test_idx  = np.where(test_mask)[0]

        print(
            f"\n[LODO {fold_num}/{len(eligible)}] "
            f"Test dataset: {dataset}  (n={len(test_idx)})"
        )

        fold_metrics = _run_fold(
            pipeline=pipeline,
            X_train=X[train_idx], X_test=X[test_idx],
            y_train=y[train_idx], y_test=y[test_idx],
            m_test=metadata.iloc[test_idx],
            tumor_types=tumor_types,
            output_file=cv_output_file,
            fold_num=fold_num,
            global_times=None,
            n_eval_times=n_eval_times,
            eval_time_range=eval_time_range,
            verbose=True,
            dataset_name=dataset,
        )
        if fold_metrics is None:
            continue
        for tt, metrics in fold_metrics.items():
            _accumulate(results, tt, metrics)

    _print_summary(results, tumor_types, ibs_output_file=ibs_output_file)
    print(f"\nPredictions exported to: {cv_output_file}")
    if os.path.exists(cv_output_file):
        print(f"Total samples: {len(pd.read_csv(cv_output_file, sep=chr(9)))}")
    return results


@beartype
def get_cv_results(cv_output_file: str, ibs_output_file: str) -> list[dict]:
    """
    Extract CV results statistics (IBS, C-index, and standard deviations).

    Parameters
    ----------
    cv_output_file : str
        Path to the CV predictions CSV file (output/cv/cv__*.csv)
    ibs_output_file : str
        Path to the IBS statistics file (output/ibs/cv__*.csv)

    Returns
    -------
    list[dict]
        One dict per tumor type with keys: tumor_type, ibs_mean, ibs_std, c_index_mean, c_index_std.

    Notes
    -----
    IBS mean and std are read directly from the IBS output file (written by
    _print_summary during CV). Per-fold IBS cannot be recomputed from the
    predictions CSV because the full survival curves are not stored there.
    C-index std is computed per fold from risk_score, which is available.
    """
    ibs_df = pd.read_csv(ibs_output_file, sep='\t', header=None, names=['tumor_type', 'metric', 'value'])
    ibs_mean_dict = dict(zip(
        ibs_df[ibs_df['metric'] == 'brier_pycox']['tumor_type'],
        ibs_df[ibs_df['metric'] == 'brier_pycox']['value'],
    ))
    ibs_std_dict = dict(zip(
        ibs_df[ibs_df['metric'] == 'brier_pycox_std']['tumor_type'],
        ibs_df[ibs_df['metric'] == 'brier_pycox_std']['value'],
    ))

    pred_df = pd.read_csv(cv_output_file, sep='\t')

    stats = []
    for tt in ibs_mean_dict.keys():
        mask = (pred_df['Tumor_Type'].astype(str) == tt) if tt != "Overall" else np.ones(len(pred_df), dtype=bool)
        sub = pred_df[mask]

        if len(sub) > 0:
            event_mask = sub['actual_event'].astype(bool)
            if event_mask.any():
                c_indices = []

                for fold in sorted(sub['fold'].unique()):
                    fold_sub = sub[sub['fold'] == fold]
                    fold_event_mask = fold_sub['actual_event'].astype(bool)

                    if fold_event_mask.any():
                        c_idx = concordance_index_censored(
                            fold_event_mask.values, fold_sub['actual_time'].values, fold_sub['risk_score'].values
                        )[0]
                        c_indices.append(c_idx)

                if c_indices:
                    stats.append({
                        'tumor_type':    tt,
                        'ibs_mean':      float(ibs_mean_dict.get(tt, np.nan)),
                        'ibs_std':       float(ibs_std_dict.get(tt, np.nan)),
                        'c_index_mean':  float(np.mean(c_indices)),
                        'c_index_std':   float(np.std(c_indices)),
                    })
    return stats


@beartype
def get_lodo_results(lodo_output_file: str, ibs_output_file: str) -> list[dict]:
    """
    Extract LODO results statistics (IBS, C-index, and standard deviations).

    Parameters
    ----------
    lodo_output_file : str
        Path to the LODO predictions CSV file (output/cv/lodo__*.csv)
    ibs_output_file : str
        Path to the IBS statistics file (output/ibs/lodo__*.csv)

    Returns
    -------
    list[dict]
        One dict per tumor type with keys: tumor_type, ibs_mean, ibs_std, c_index_mean, c_index_std.

    Notes
    -----
    IBS mean and std are read directly from the IBS output file (written by
    _print_summary during LODO). Per-fold IBS cannot be recomputed from the
    predictions CSV because the full survival curves are not stored there.
    C-index std is computed per fold from risk_score, which is available.
    """
    ibs_df = pd.read_csv(ibs_output_file, sep='\t', header=None, names=['tumor_type', 'metric', 'value'])
    ibs_mean_dict = dict(zip(
        ibs_df[ibs_df['metric'] == 'brier_pycox']['tumor_type'],
        ibs_df[ibs_df['metric'] == 'brier_pycox']['value'],
    ))
    ibs_std_dict = dict(zip(
        ibs_df[ibs_df['metric'] == 'brier_pycox_std']['tumor_type'],
        ibs_df[ibs_df['metric'] == 'brier_pycox_std']['value'],
    ))

    pred_df = pd.read_csv(lodo_output_file, sep='\t')

    stats = []
    for tt in ibs_mean_dict.keys():
        mask = (pred_df['Tumor_Type'].astype(str) == tt) if tt != "Overall" else np.ones(len(pred_df), dtype=bool)
        sub = pred_df[mask]

        if len(sub) > 0:
            event_mask = sub['actual_event'].astype(bool)
            if event_mask.any():
                c_indices = []

                for fold in sorted(sub['fold'].unique()):
                    fold_sub = sub[sub['fold'] == fold]
                    fold_event_mask = fold_sub['actual_event'].astype(bool)

                    if fold_event_mask.any():
                        c_idx = concordance_index_censored(
                            fold_event_mask.values, fold_sub['actual_time'].values, fold_sub['risk_score'].values
                        )[0]
                        c_indices.append(c_idx)

                if c_indices:
                    stats.append({
                        'tumor_type':    tt,
                        'ibs_mean':      float(ibs_mean_dict.get(tt, np.nan)),
                        'ibs_std':       float(ibs_std_dict.get(tt, np.nan)),
                        'c_index_mean':  float(np.mean(c_indices)),
                        'c_index_std':   float(np.std(c_indices)),
                    })
    return stats


@beartype
def run_grid_search(
    classifier_class: type,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    classifier_name: str | None = None,
    output_dir: str = "output",
    n_splits: int = 5,
    random_state: int = 43,
    tumor_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Cross-validated grid search over *param_grid*.

    Pipeline params (var_filter, select_k_best, n_components, pca_class) are
    handled by ``build_pipeline``; all other keys are forwarded to the
    classifier. Missing pipeline params are filled with ``_PARAM_DEFAULTS``
    so that every output row has consistent columns.

    ``select_k_best`` and ``n_components`` are independent — both, either, or
    neither can be active. ``pca_class`` defaults to ``PCA`` and is
    only used when ``n_components`` is not None.

    Returns
    -------
    pd.DataFrame
        One row per (combo, tumor_type) with mean ± std for all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    if classifier_name is None:
        classifier_name = _derive_name(classifier_class)
    if tumor_types is None:
        tumor_types = _DEFAULT_TUMOR_TYPES

    summary_file = os.path.join(
        output_dir, f"grid_search/grid_search__{classifier_name}__summary.csv"
    )

    all_combos = [
        dict(zip(param_grid.keys(), combo))
        for combo in itertools.product(*param_grid.values())
    ]

    print(f"Classifier  : {classifier_name}")
    print(f"Combos      : {len(all_combos)}")
    print(f"CV folds    : {n_splits}")
    print(f"Output dir  : {output_dir}")

    kf           = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_times = np.linspace(y["time"].min(), y["time"].max() - 0.5, 50)
    summary_rows: list[dict] = []

    for combo_idx, params in enumerate(tqdm(all_combos, desc="Grid combos"), start=1):
        tag         = _combo_tag(params)
        output_file = os.path.join(
            output_dir,
            f"grid_search/individual/grid_search__{classifier_name}__{tag}.csv",
        )
        print(f"\n[{combo_idx}/{len(all_combos)}] {tag}")

        try:
            pipeline     = build_pipeline(classifier_class, params)
            fold_results = _empty_fold_results(tumor_types)

            for fold_num, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
                try:
                    fold_metrics = _run_fold(
                        pipeline=pipeline,
                        X_train=X[train_idx], X_test=X[test_idx],
                        y_train=y[train_idx], y_test=y[test_idx],
                        m_test=metadata.iloc[test_idx],
                        tumor_types=tumor_types,
                        output_file=output_file,
                        fold_num=fold_num,
                        global_times=global_times,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"  ⚠ Fold {fold_num} failed: {e}")
                    continue

                if fold_metrics is None:
                    continue
                for tt, metrics in fold_metrics.items():
                    _accumulate(fold_results, tt, metrics)

        except Exception as e:
            print(f"  ✖ Combo {tag} failed entirely: {e}")
            for tt in tumor_types:
                summary_rows.append(_nan_row(tag, params, tt))
            pd.DataFrame(summary_rows).to_csv(
                summary_file, index=False, sep="\t", float_format="%.4f"
            )
            continue

        for tt in tumor_types:
            fd = fold_results[tt]
            if not fd["c_index_sksurv"]:
                summary_rows.append(_nan_row(tag, params, tt))
                continue

            row = {
                "combo_tag":  tag,
                **_normalize_params(params),
                "tumor_type": tt,
                "n_folds":    len(fd["c_index_sksurv"]),
            }
            for k in _METRIC_KEYS:
                row[f"{k}_mean"] = float(np.mean(fd[k]))
                row[f"{k}_std"]  = float(np.std(fd[k]))

            summary_rows.append(row)
            print(
                f"  [Combo] {tt:18s}  "
                f"C-idx(pycox)={row['c_index_pycox_mean']:.3f}±{row['c_index_pycox_std']:.3f}  "
                #f"Brier={row['brier_sksurv_mean']:.3f}  "
                f"MAE={row['mae_linear_mean']:.1f}d  "
                f"slope={row['slope_linear_mean']:.2f}"
            )

        pd.DataFrame(summary_rows).to_csv(
            summary_file, index=False, sep="\t", float_format="%.4f"
        )

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(summary_file, index=False, sep="\t", float_format="%.4f")

    print(f"\n{'='*80}\nGRID SEARCH COMPLETE\n{'='*80}")
    print(f"Summary : {summary_file}")

    for tt in tumor_types:
        sub = results_df[results_df["tumor_type"] == tt].sort_values(
            "c_index_sksurv_mean", ascending=False
        )
        print(f"\nTop 5 — {tt.upper()}")
        print(sub[[
            "combo_tag",
            #"c_index_sksurv_mean", "c_index_sksurv_std",
            "c_index_pycox_mean", "c_index_pycox_std",
            #"brier_sksurv_mean",
            "mae_linear_mean", "slope_linear_mean",
        ]].head(5).to_string(index=False))

    return results_df
