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
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
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

from . import ASSETS_PATH, DAYS_PER_YEAR, MAX_FOLLOW_UP_YEARS, __version__


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys handled by the pipeline builder; all other keys in param_grid are
# forwarded as keyword arguments directly to the classifier constructor.
_PIPELINE_PARAMS     = {"var_filter", "n_components", "select_k_best", "select_k_best_components"}
_DEFAULT_TUMOR_TYPES = ["Oligodendroglioma", "Astrocytoma", "Glioblastoma", "Overall"]

# Shared time grid (days since the sample was taken) for the reference-cohort
# curves that go into a model's .info.json. It has to be one grid across every
# model: the report averages those curves member by member before reading a
# median off the result, and curves living on their own grids cannot be
# averaged column by column -- which is exactly what leaves a model's own
# median stuck on its own num_durations ladder.
#
# Monthly, because these are stored as decimal text: a thousand samples over
# 9700 days would be ten megabytes per model on top of a .info.json that is
# already close to seven, and every one of them is parsed on every report. The
# step costs no accuracy where it matters -- merge_grade_curves interpolates
# the 0.5 crossing between two grid points, so the median it reads off is
# continuous however coarse the grid, and a survival curve is smooth enough
# over a month for that interpolation to be exact to well under a day.
GRADE_CURVE_STEP_DAYS = 30
GRADE_CURVE_TIMES     = np.arange(
    1, round(DAYS_PER_YEAR * MAX_FOLLOW_UP_YEARS) + 1, GRADE_CURVE_STEP_DAYS,
)
# stored probabilities are rounded to this many decimals, for the same reason:
# four is already far finer than a survival probability carries meaning, and it
# is a third of the characters a full repr would spend
GRADE_CURVE_DECIMALS = 4

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
_METRIC_KEYS__TIME_TO_EVENT = (
    "c_index_sksurv", "c_index_pycox",
    "brier_sksurv",
    #"brier_pycox",
    "mae_linear",     "mae_log1p",
    "slope_linear",   "slope_log1p",
    "r2_linear",      "r2_log1p",
)

# Metrics collected per fold per group for class-based (classification)
# cross-validation. Unlike the time-to-event metrics these need no time grid:
# every metric is computed from the hard label predictions of the fold, except
# roc_auc which needs predict_proba and is NaN when the pipeline does not
# expose it (or when a fold misses a class, leaving the AUC undefined).
_METRIC_KEYS__CLASSIFICATION = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "mcc",
    "roc_auc",
)

# Groups evaluated by default in class-based CV. Only "Overall" - unlike
# survival, where the two glioma subtypes are the clinically interesting
# split, a class-based endpoint (sex, subtype, location) is normally judged
# on the whole cohort; pass `groups=` to subset anyway.
_DEFAULT_GROUPS__CLASSIFICATION = ["Overall"]


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
    for k in _METRIC_KEYS__TIME_TO_EVENT:
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
def _eval_group__time_to_event(
    tt: str,
    mask: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray,
    surv_preds: Union[list, np.ndarray],
    risk_scores: np.ndarray,
    expected_survival: np.ndarray,
    fold_num: int,
    global_times: np.ndarray,
    verbose: bool = True,
) -> dict | None:
    """
    Compute C-index (sksurv + pycox), Brier, and regression metrics for one
    tumor-type subset.  Returns None and prints a warning when skipped.

    With *verbose* False the skip warnings and the time-range line stay silent,
    for callers that evaluate the same cohort many times over (a leave-one-out
    ensemble sweep) and would otherwise drown in repeats. The return value is
    unaffected: a skipped group still comes back as None.
    """
    if not mask.any():
        if verbose:
            print(f"  ⚠ No {tt} samples in fold {fold_num}")
        return None

    y_sub      = y_test[mask]
    rs_sub     = risk_scores[mask]
    es_sub     = expected_survival[mask]
    event_mask = y_sub["event"].astype(bool)

    if not event_mask.any():
        if verbose:
            print(f"  ⚠ No events for {tt} in fold {fold_num}, skipping metrics")
        return None

    _ts_start = max(y_sub["time"].min(), y_train["time"].min())
    _ts_stop  = min(y_sub["time"].max(), y_train["time"].max()) - 0.5
    if _ts_stop <= _ts_start:
        if verbose:
            print(f"  ⚠ Invalid time range for {tt} in fold {fold_num}, skipping")
        return None

    times_sub = global_times[(global_times >= _ts_start) & (global_times <= _ts_stop)]
    if len(times_sub) < 2:
        if verbose:
            print(f"  ⚠ Too few time points for {tt} in fold {fold_num}, skipping")
        return None

    if verbose:
        print(
            f"  [CV{fold_num}] {tt}: train time range = {y_train['time'].min()}-{y_train['time'].max()}, "
            f"test time range = {y_test['time'].min()}-{y_test['time'].max()}"
        )

    surv_preds_sub  = [surv_preds[i] for i in np.where(mask)[0]]
    surv_matrix_sub = np.array([sf(times_sub) for sf in surv_preds_sub])

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

    return {
        "c_index_sksurv":        c_idx_sksurv,
        "c_index_pycox":         ev.concordance_td(),
        "brier_sksurv":          float(integrated_brier_score(y_train, y_sub, surv_matrix_sub, times_sub)),
        "brier_sksurv_leq_6500": float(integrated_brier_score(y_train, y_sub_clipped, surv_clipped, times_clipped)),
        #"brier_pycox":           float(ev.integrated_brier_score(times_sub)),
        "n_samples":             int(mask.sum()),
        **_regression_metrics(
            actual=y_sub["time"][event_mask],
            predicted=es_sub[event_mask],
        ),
    }


@beartype
def _empty_fold_results(tumor_types: list[str], metric_keys: tuple = _METRIC_KEYS__TIME_TO_EVENT) -> dict:
    return {tt: {k: [] for k in metric_keys} for tt in tumor_types}


@beartype
def _accumulate(results: dict, tt: str, fold_metrics: dict, metric_keys: tuple = _METRIC_KEYS__TIME_TO_EVENT) -> None:
    """
    Append per-fold metric values to the running lists in *results*.

    *results* has the shape produced by ``_empty_fold_results``:
    ``{tumor_type: {metric_name: [fold1_value, fold2_value, ...]}}``

    Called once per tumor type per completed fold so that after all folds
    ``results[tt][metric]`` is a list of length n_completed_folds, ready
    for ``np.mean`` / ``np.std`` aggregation.

    ``metric_keys`` selects which metric set is accumulated: the time-to-event
    keys by default, ``_METRIC_KEYS__CLASSIFICATION`` for class-based CV.
    """
    for k in metric_keys:
        results[tt][k].append(fold_metrics[k])


@beartype
def _median_expected_survival(surv_matrix: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median (first time the curve crosses 0.5) and expected (trapezoid AUC) survival."""
    argmin_idx        = np.argmin(np.abs(surv_matrix - 0.5), axis=1)
    median_survival   = np.where(surv_matrix.min(axis=1) > 0.5, np.nan, times[argmin_idx])
    expected_survival = np.trapezoid(surv_matrix, times, axis=1)
    return median_survival, expected_survival


@beartype
def _run_fold__time_to_event(
    pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    m_test: pd.DataFrame,
    tumor_types: list[str],
    output_file: str | None,
    fold_num: int,
    global_times: np.ndarray,
    verbose: bool = True,
    dataset_name: str | None = None,
    curve_sink: dict | None = None,
    X_unlabelled: np.ndarray | None = None,
) -> dict[str, dict[str, float]] | None:
    """
    Fit *pipeline*, predict on the test fold, write per-patient TSV rows, and
    compute all metrics per tumor type.

    When *curve_sink* is passed, this fold's full survival matrix is stored in
    it so the caller can merge all folds into a single out-of-fold file; see
    ``_write_oof_curves``. Nothing is stored when the fold returns early, so
    the caller must check for the ``surv`` key rather than assume it is there.

    *X_unlabelled* is handed straight to ``SurvivalPipeline.fit``: those
    samples only take part in the unsupervised steps (PCA above all). They are
    the same in every fold, which is leakage-free precisely because they carry
    no outcome and are never predicted on.

    Returns a dict ``{tumor_type: {metric: value}}`` or None if time range is
    invalid.
    """
    _t_start = max(y_test["time"].min(), y_train["time"].min())
    _t_stop  = min(y_test["time"].max(), y_train["time"].max()) - 0.5
    if _t_stop <= _t_start:
        print(f"  ⚠ Invalid time range in fold {fold_num}, skipping")
        return None

    times = global_times[(global_times >= _t_start) & (global_times <= _t_stop)]
    if len(times) < 2:
        print(f"  ⚠ Too few time points in fold {fold_num}, skipping")
        return None

    pipeline = clone(pipeline)
    pipeline.fit(X_train, y_train, X_unlabelled=X_unlabelled)

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

    predictions = pipeline.predict_all(X_test, times=times)
    surv_preds  = predictions["survival_functions"]
    surv_matrix = np.array([sf(times) for sf in surv_preds])

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

    # A survival curve is a probability, so anything outside [0, 1] means the
    # model came apart in this fold and every metric derived from it is
    # meaningless -- the IBS is a weighted mean of (1-s)^2 and s^2, which stays
    # below 1 only while s does, and expected_survival integrates the same
    # matrix. The cause sits upstream: sksurv builds a Coxnet curve as
    # baseline_survival ** exp(linear predictor), and an exploded linear
    # predictor turns a baseline a hair above 1 into an astronomical number.
    # Such a fold is dropped rather than repaired, exactly like an invalid time
    # range above; clipping would keep a broken model in the average. The
    # tolerance leaves ordinary rounding alone.
    if not np.all(np.isfinite(surv_matrix)) or surv_matrix.min() < -_SURV_TOL or surv_matrix.max() > 1.0 + _SURV_TOL:
        print(f"  ⚠ Survival curves outside [0, 1] in fold {fold_num} "
              f"(range [{np.nanmin(surv_matrix):.3g}, {np.nanmax(surv_matrix):.3g}]), skipping. "
              f"The model diverged here -- raise the regularisation (alphas) if this repeats.")
        return None

    median_survival, expected_survival = _median_expected_survival(surv_matrix, times)

    if output_file is not None:
        export_df = pd.DataFrame({
            "Sentrix_ID":        m_test["Sentrix ID"].values,
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

    if curve_sink is not None:
        curve_sink["surv"]        = surv_matrix.astype(np.float32)
        curve_sink["times"]       = times
        curve_sink["risk_scores"] = predictions["risk_scores"]
        curve_sink["fold"]        = fold_num

    fold_metrics: dict[str, dict[str, float]] = {}
    for tt in tumor_types:
        mask = (
            np.ones(len(m_test), dtype=bool) if tt == "Overall"
            else (m_test["Tumor Type"].astype(str) == tt).values
        )
        
        result = _eval_group__time_to_event(
            tt=tt, mask=mask,
            y_test=y_test, y_train=y_train,
            surv_preds=surv_preds,
            risk_scores=predictions["risk_scores"],
            expected_survival=expected_survival,
            fold_num=fold_num,
            global_times=global_times,
        )

        if result is not None:
            fold_metrics[tt] = result
            
            print(
                f"  [CV{fold_num}] {tt:18s}  n={result['n_samples']:3d}  "
                f"C-idx(pycox)={result['c_index_pycox']:.3f}  "
                f"IBS sksurv={result['brier_sksurv']:.3f}  "
                f"IBS sksurv ≤ 6500={result['brier_sksurv_leq_6500']:.3f}  "
                #f"IBS pycoxEV={result['brier_pycox']:.3f}  "
                #f"MAE={result['mae_linear']:.1f}d  "
                #f"slope={result['slope_linear']:.2f}  "
                #f"R²={result['r2_linear']:.3f}"
            )

    return fold_metrics


@beartype
def _print_summary__time_to_event(
    results: dict,
    tumor_types: list[str],
    ibs_output_file: str | None = None,
    verbose: bool = True,
) -> None:
    # The file export is not part of the chatter: verbose only silences the
    # table, ibs_output_file is still written when it was asked for.
    if not verbose:
        _export_ibs__time_to_event(results, tumor_types, ibs_output_file)
        return

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
        for k in _METRIC_KEYS__TIME_TO_EVENT:
            vals = r[k]
            print(f"  {k:22s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 80)

    _export_ibs__time_to_event(results, tumor_types, ibs_output_file, verbose=True)


@beartype
def _export_ibs__time_to_event(
    results: dict,
    tumor_types: list[str],
    ibs_output_file: str | None,
    verbose: bool = False,
) -> None:
    """Write the per-tumor-type IBS mean/std to a TSV; no-op without a path."""
    if not ibs_output_file:
        return
    if os.path.dirname(ibs_output_file):
        os.makedirs(os.path.dirname(ibs_output_file), exist_ok=True)
    with open(ibs_output_file, "w") as f:
        for tt in tumor_types:
            r = results[tt]
            if r["brier_pycox"]:
                vals = r["brier_pycox"]
                f.write(f"{tt}\tbrier_pycox\t{np.mean(vals):.4f}\n")
                f.write(f"{tt}\tbrier_pycox_std\t{np.std(vals):.4f}\n")
    if verbose:
        print(f"IBS exported to: {ibs_output_file}")


# ---------------------------------------------------------------------------
# Internal helpers - class-based (classification) CV
# ---------------------------------------------------------------------------

@beartype
def _group_mask(m_test: pd.DataFrame, group: str, group_column: str) -> np.ndarray:
    """Boolean mask selecting the samples of *group*; "Overall" selects all."""
    if group == "Overall":
        return np.ones(len(m_test), dtype=bool)
    return (m_test[group_column].astype(str) == group).values


@beartype
def _roc_auc(y_true: np.ndarray, y_proba: np.ndarray | None, classes: np.ndarray) -> float:
    """
    ROC-AUC for the binary (probability of the positive class) or multiclass
    (one-vs-rest, macro-averaged) case.

    Returns NaN rather than raising whenever the AUC is undefined for this
    subset - no predict_proba on the pipeline, or a fold/group in which only
    one class is present - so a single degenerate group never aborts a CV run.
    """
    if y_proba is None or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
        )
    except ValueError:
        # multiclass OvR needs every class represented in y_true; a group that
        # misses one leaves the macro average undefined.
        return float("nan")


@beartype
def _eval_group__classification(
    group: str,
    mask: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: np.ndarray,
    fold_num: int,
) -> dict | None:
    """
    Compute accuracy, balanced accuracy, macro-F1, MCC and ROC-AUC for one
    group subset. Returns None and prints a warning when skipped.
    """
    if not mask.any():
        print(f"  ⚠ No {group} samples in fold {fold_num}")
        return None

    yt = y_true[mask]
    yp = y_pred[mask]
    pr = None if y_proba is None else y_proba[mask]

    # MCC is undefined with a single class present (sklearn returns 0.0, which
    # would silently drag the mean over folds down); NaN keeps such a fold out
    # of the average instead, the same way _roc_auc handles it.
    single_class = len(np.unique(yt)) < 2

    return {
        "accuracy":          float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "f1_macro":          float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mcc":               float("nan") if single_class else float(matthews_corrcoef(yt, yp)),
        "roc_auc":           _roc_auc(yt, pr, classes),
        "n_samples":         int(mask.sum()),
    }


@beartype
def _run_fold__classification(
    pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    m_test: pd.DataFrame,
    groups: list[str],
    group_column: str,
    output_file: str | None,
    fold_num: int,
    label_encoder=None,
    verbose: bool = True,
    dataset_name: str | None = None,
) -> tuple[dict[str, dict[str, float]], np.ndarray] | None:
    """
    Fit *pipeline*, predict on the test fold, write per-sample TSV rows, and
    compute all metrics per group.

    Returns ``({group: {metric: value}}, y_pred)`` so the caller can pool the
    out-of-fold predictions into a single classification report, or None when
    the fold is unusable.
    """
    pipeline = clone(pipeline)
    pipeline.fit(X_train, y_train)

    # Debug: check transformed data validity
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

    y_pred  = pipeline.predict(X_test)
    classes = np.asarray(pipeline.classes_) if hasattr(pipeline, "classes_") else np.unique(y_train)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    if output_file is not None:
        # Decoded labels are what the report reads back; keep the encoded ints
        # alongside them so the TSV stays usable without the LabelEncoder.
        decode      = (lambda v: label_encoder.inverse_transform(v)) if label_encoder is not None else (lambda v: v)
        export_df = pd.DataFrame({
            "Sentrix_ID":       m_test["Sentrix ID"].values,
            "Tumor_Type":       m_test["Tumor Type"].values,
            "actual":           y_test,
            "predicted":        y_pred,
            "actual_label":     decode(y_test),
            "predicted_label":  decode(y_pred),
            "probability":      np.nan if y_proba is None else y_proba.max(axis=1),
            "correct":          (y_test == y_pred).astype(int),
            "fold":             fold_num,
            "dataset":          dataset_name,
        })
        export_df.to_csv(
            output_file, mode="a", header=(fold_num == 1),
            index=False, sep="\t", float_format="%.4f",
        )

    fold_metrics: dict[str, dict[str, float]] = {}
    for group in groups:
        result = _eval_group__classification(
            group=group,
            mask=_group_mask(m_test, group, group_column),
            y_true=y_test, y_pred=y_pred, y_proba=y_proba,
            classes=classes, fold_num=fold_num,
        )

        if result is not None:
            fold_metrics[group] = result

            print(
                f"  [CV{fold_num}] {group:18s}  n={result['n_samples']:3d}  "
                f"acc={result['accuracy']:.4f}  "
                f"bal_acc={result['balanced_accuracy']:.4f}"
            )

    return fold_metrics, y_pred


@beartype
def _print_summary__classification(
    results: dict,
    groups: list[str],
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    label_encoder=None,
    metrics_output_file: str | None = None,
) -> None:
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    for group in groups:
        r = results[group]
        if not r["accuracy"]:
            print(f"\n{group}: No data")
            continue
        print(f"\n{group.upper()}")
        print("-" * 40)
        for k in _METRIC_KEYS__CLASSIFICATION:
            # nanmean/nanstd: roc_auc and mcc are NaN for folds where they are
            # undefined (single-class fold, no predict_proba), and those folds
            # should drop out of the average instead of poisoning it.
            vals = r[k]
            print(f"  {k:22s}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
    print("=" * 80)

    # Pooled out-of-fold report: every sample here was predicted by a model
    # that never saw it, so the per-class scores below are computed directly
    # on those predictions instead of being averaged over folds. K-fold covers
    # the whole cohort; LODO only the held-out datasets.
    if y_true is not None and y_pred is not None and len(y_true):
        # Pin the label set to the encoder's classes: a LODO run covers only
        # the held-out datasets, so a class can be absent from the pooled
        # predictions and target_names would no longer line up.
        if label_encoder is None:
            labels, target_names = None, None
        else:
            labels       = np.arange(len(label_encoder.classes_))
            target_names = [str(c) for c in label_encoder.classes_]
        print("\nClassification report (pooled out-of-fold predictions):")
        print(classification_report(
            y_true, y_pred, labels=labels, target_names=target_names,
            digits=4, zero_division=0,
        ))
        print("Confusion matrix (rows = actual, cols = predicted):")
        print(confusion_matrix(y_true, y_pred, labels=labels))

    if metrics_output_file:
        if os.path.dirname(metrics_output_file):
            os.makedirs(os.path.dirname(metrics_output_file), exist_ok=True)
        with open(metrics_output_file, "w") as f:
            for group in groups:
                r = results[group]
                for k in _METRIC_KEYS__CLASSIFICATION:
                    if r[k]:
                        f.write(f"{group}\t{k}\t{np.nanmean(r[k]):.4f}\n")
                        f.write(f"{group}\t{k}_std\t{np.nanstd(r[k]):.4f}\n")
        print(f"Metrics exported to: {metrics_output_file}")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

# Rounding slack when checking that a survival curve stays a probability; a
# curve that leaves [0, 1] by more than this did not round, it diverged.
_SURV_TOL = 1e-6


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
    name = re.sub(r"(__build_model|__grid_search)+$", "", name)

    params = name.replace("predict_", "", 1).split("_")

    e = EndPoint.from_string(" ".join(params[:-1]))
    p = PredictorType.from_string(params[-1])

    return (e, p)


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
    oof_curves_file: str | None = None,
    exclude_from_running: bool = False,
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

    # The out-of-fold curves are produced by the cross-validation step, which
    # runs before this function mints `base_name` and therefore cannot know the
    # final filename. They are moved in here instead, so the curves end up next
    # to the model they belong to and the metadata can reference them with a
    # path relative to the assets tree -- keeping the exported model portable.
    relative_curves_path = None
    if oof_curves_file is not None:
        curves_source = Path(oof_curves_file)
        if not curves_source.exists():
            raise FileNotFoundError(
                f"oof_curves_file does not exist: {curves_source}. Run the "
                f"cross-validation with curve_output_file= before exporting."
            )
        relative_curves_path = relative_model_basedir / f"{base_name}__cv-oof-curves.npz"
        curves_target        = model_basedir / f"{base_name}__cv-oof-curves.npz"
        print(f"Moving OOF curves to: {curves_target}")
        curves_source.replace(curves_target)

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
    if relative_curves_path is not None:
        meta['cv oof curves'] = str(relative_curves_path)
    if exclude_from_running:
        # The model is exported and kept around -- its CV numbers and its OOF
        # curves stay usable for the offline analysis -- but the CLI skips it
        # entirely: no prediction, no stdout line, and no share in whatever the
        # report averages. Set it for a model that is kept for reference rather
        # than for use, such as an ensemble member a leave-one-out sweep shows
        # the ensemble does not need. Only written when True, so models
        # exported before this flag existed read as included.
        meta['exclude from running'] = True
    if performance is not None:
        meta["performance"] = performance
    if calibrations is not None:
        meta["calibrations"] = calibrations
    if grade_only_results is not None:
        records = grade_only_results.drop(columns=["curve"]).to_dict(orient="records")

        # The curve column was until now only ever used to be dropped. It goes
        # in alongside the scalars instead, re-read on the shared grid: the
        # curves come out of predict_curves_median_expected on whatever grid the
        # model happens to use -- the unique event times of a Cox or AFT fit, the
        # num_durations cut points of a discrete-time one -- and only on one
        # common grid can the report average them across models. That averaging
        # is the whole point: a median read off a single model's curve can only
        # land on that model's own grid, which is what collapses a thousand
        # reference samples onto a dozen x positions.
        #
        # The grid is written once at the top rather than repeated per record,
        # and the horizon with it: a model whose own domain stops short of
        # MAX_FOLLOW_UP_YEARS is held flat from there on (see curve_on_grid), and
        # the merge needs to know where that starts to keep it out of a median.
        curves = list(grade_only_results["curve"])
        for rec, sf in zip(records, curves):
            rec["curve"] = [round(float(v), GRADE_CURVE_DECIMALS)
                            for v in curve_on_grid(sf, GRADE_CURVE_TIMES)]

        meta["grade_only_results"]  = records
        meta["grade curve times"]   = GRADE_CURVE_TIMES.tolist()
        meta["grade curve horizon"] = (min(float(sf.domain[1]) for sf in curves)
                                       if curves else None)

    print(f"Saving metadata to: {meta_path}")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return str(model_path)


@beartype
def _write_oof_curves(
    curves: list[dict],
    y: np.ndarray,
    metadata: pd.DataFrame,
    global_times: np.ndarray,
    output_file: str,
    n_splits: int,
    random_state: int,
) -> None:
    """
    Merge the per-fold survival matrices into one out-of-fold cohort matrix and
    write it to *output_file* as a single ``.npz``.

    Rows cover the whole cohort in the order of *metadata*, not just the
    patients that ended up in a test fold. That keeps ``y_train`` of fold k
    derivable as "every row with ``fold != k``" even once the folds stop
    partitioning the cohort (LODO never holds out its smallest datasets, yet
    still trains on them). Samples that were never held out keep ``fold = -1``
    and an all-NaN curve; a fold that returned early leaves its samples the
    same way.

    Values outside a fold's valid time window are NaN too. That window is
    derived from the split alone, so the NaN pattern is identical for every
    model trained on the same split and the curves line up column by column in
    ``run_ensemble__time_to_event``.
    """
    n    = len(metadata)
    surv = np.full((n, len(global_times)), np.nan, dtype=np.float32)
    fold = np.full(n, -1, dtype=np.int32)
    risk = np.full(n, np.nan, dtype=np.float64)

    for c in curves:
        idx = c["test_idx"]
        # c["times"] is a contiguous slice of global_times, so its columns are
        # recoverable by exact value: both arrays hold the very same floats.
        cols = np.where(np.isin(global_times, c["times"]))[0]
        surv[np.ix_(idx, cols)] = c["surv"]
        fold[idx]               = c["fold"]
        risk[idx]               = c["risk_scores"]

    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.savez_compressed(
        output_file,
        scheme="cv",
        times=global_times,
        surv=surv,
        fold=fold,
        risk_score=risk,
        sentrix_id=metadata["Sentrix ID"].values.astype(str),
        tumor_type=metadata["Tumor Type"].values.astype(str),
        y=y,
        n_splits=n_splits,
        random_state=random_state,
    )

    n_covered = int((fold >= 0).sum())
    print(f"OOF curves exported to: {output_file}  ({n_covered}/{n} samples covered)")


@beartype
def _patient_groups(metadata: pd.DataFrame, patient_column: str | None) -> np.ndarray:
    """
    Fold-grouping labels that keep every sample of one patient together.

    Replicates and repeat biopsies share a ``Patient ID``. Splitting them over
    train and test puts near-identical methylation profiles on both sides, which
    inflates the C-index and flatters the Brier score.

    A sample carrying no identifier is linked to nothing, so it gets a group of
    its own and stays free to land in any fold. The same holds for every sample
    when the column is absent or ``patient_column`` is None: all groups are then
    unique and the grouped splitter behaves like an ordinary K-fold, which keeps
    one code path for both cases.
    """
    if patient_column is None or patient_column not in metadata.columns:
        ids = pd.Series(pd.NA, index=metadata.index, dtype="string")
    else:
        ids = metadata[patient_column].astype("string").str.strip()
        # Values that only look like an identifier: the string forms a missing
        # cell takes on after a CSV round-trip.
        ids = ids.mask(ids.isin(["", "nan", "None", "NA", "N/A", "<NA>", "-"]))

    return np.array([
        f"__unlinked_{i}" if pd.isna(pid) else str(pid)
        for i, pid in enumerate(ids)
    ], dtype=object)


@beartype
def _cv_splits(
    X: np.ndarray,
    metadata: pd.DataFrame,
    n_splits: int,
    random_state: int,
    patient_column: str | None = "Patient ID",
    stratify: np.ndarray | None = None,
    verbose: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Cross-validation splits that never separate two samples of the same patient.

    The split is always group-aware; ``_patient_groups`` decides what a group
    is, and hands out singleton groups where no patient is known - so passing
    ``patient_column=None`` degrades this to a plain K-fold without a second
    branch here. Passing *stratify* keeps the class balance of that vector in
    every fold. Group folds are inherently a little uneven in size - the
    grouping constraint wins over the balance.
    """
    groups   = _patient_groups(metadata, patient_column)
    splitter = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if stratify is not None else
        GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )

    if verbose:
        counts = pd.Series(groups).value_counts()
        linked = counts[counts > 1]
        print(
            f"Patient-grouped CV: {len(groups)} samples in {len(counts)} groups "
            f"({int(linked.sum())} samples in {len(linked)} multi-sample patients)"
        )

    return list(splitter.split(X, stratify, groups))


@beartype
def run_cross_validation__time_to_event(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    cv_output_file: str,
    tumor_types: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    ibs_output_file: str | None = None,
    curve_output_file: str | None = None,
    patient_column: str | None = "Patient ID",
    X_unlabelled: np.ndarray | None = None,
) -> dict:
    """
    Stratified K-fold cross-validation for a survival pipeline.

    All samples of one patient stay in the same fold - see ``_cv_splits``.
    Pass ``patient_column=None`` to switch the grouping off; the split is then
    an ordinary K-fold again, though not the identical one a pre-grouping run
    produced for the same *random_state*.

    ``X_unlabelled`` adds samples without an outcome to the unsupervised steps
    of the pipeline (the PCA in particular), without them ever being scored --
    they take part in no split, no metric and no export. Their feature columns
    must match ``X`` in both content and order. The model that is fitted for
    export has to be given the same matrix, or its PCA is not the one that was
    validated here.

    Returns
    -------
    dict
        ``results[tumor_type][metric]`` — per-fold lists.
        See ``_METRIC_KEYS__TIME_TO_EVENT`` for the full set of metrics.
        Regression metrics (mae, slope, r2) are computed on event-only
        patients using expected_survival (trapezoid AUC) as the predicted
        value. Truncation bias in expected_survival is not corrected here;
        use the LOWESS calibrator post-hoc instead.
    """
    if tumor_types is None:
        tumor_types = _DEFAULT_TUMOR_TYPES

    splits       = _cv_splits(X, metadata, n_splits, random_state, patient_column)
    results      = _empty_fold_results(tumor_types)
    global_times = np.linspace(y["time"].min(), y["time"].max() - 0.5, 50)
    curves       = [] if curve_output_file is not None else None

    if os.path.exists(cv_output_file):
        os.remove(cv_output_file)

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        # The sink carries the test indices in, so the writer can place each
        # fold's rows back at their cohort position without re-deriving the split.
        sink = None if curves is None else {"test_idx": test_idx}

        fold_metrics = _run_fold__time_to_event(
            pipeline=pipeline,
            X_train=X[train_idx], X_test=X[test_idx],
            y_train=y[train_idx], y_test=y[test_idx],
            m_test=metadata.iloc[test_idx],
            tumor_types=tumor_types,
            output_file=cv_output_file,
            fold_num=fold_num,
            global_times=global_times,
            verbose=True,
            curve_sink=sink,
            X_unlabelled=X_unlabelled,
        )
        if sink is not None and "surv" in sink:
            curves.append(sink)
        if fold_metrics is None:
            continue
        for tt, metrics in fold_metrics.items():
            _accumulate(results, tt, metrics)

    _print_summary__time_to_event(results, tumor_types, ibs_output_file=ibs_output_file)
    print(f"\nPredictions exported to: {cv_output_file}")
    print(f"Total samples: {len(pd.read_csv(cv_output_file, sep=chr(9)))}")

    if curves:
        _write_oof_curves(
            curves=curves, y=y, metadata=metadata, global_times=global_times,
            output_file=curve_output_file, n_splits=n_splits, random_state=random_state,
        )

    return results


@beartype
def run_ensemble__time_to_event(
    curve_files: dict[str, str],
    tumor_types: list[str] | None = None,
    weights: dict[str, float] | None = None,
    ibs_output_file: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Average the out-of-fold survival curves of several models and score the
    result with the regular time-to-event metrics.

    ``curve_files`` maps a model name to the ``.npz`` written by
    ``run_cross_validation__time_to_event(..., curve_output_file=...)``. All
    models must come from the same split -- same ``n_splits``, same
    ``random_state``, same cohort order -- which is checked on load, because a
    mismatch would silently average curves belonging to different patients.

    Averaging happens on the curves, never on the risk scores: a Cox partial
    hazard, an AFT ``-predict_median`` and a DeepHit ``-rmst`` live on
    incompatible scales, so their mean carries no meaning. The ensemble risk
    score is taken from the averaged curve as ``-expected_survival``, which
    keeps it consistent with the curve the Brier score is computed on. IBS is
    the metric of interest here; the C-index that comes along is reported but
    not the target, and a rank-based alternative is described in todo.md.

    Returns
    -------
    dict
        ``results[tumor_type][metric]`` -- per-fold lists, the same shape as
        ``run_cross_validation__time_to_event``, so the existing summary,
        export and plotting helpers apply unchanged.

    *verbose* is False by default because this routine is normally called in a
    loop -- one member at a time, or a leave-one-out sweep -- where the caller
    tabulates the returned metrics itself and the per-fold lines are noise.
    Pass True for the composition line, the per-fold IBS and the summary table.
    An ``ibs_output_file`` is written either way.
    """
    if tumor_types is None:
        tumor_types = _DEFAULT_TUMOR_TYPES

    names = list(curve_files)
    if weights is None:
        w = np.full(len(names), 1.0 / len(names))
    else:
        w = np.array([weights[n] for n in names], dtype=float)
        w = w / w.sum()

    loaded, ref = [], None
    for name in names:
        d = np.load(curve_files[name], allow_pickle=False)
        if ref is None:
            ref = d
        else:
            for key, what in (("sentrix_id", "cohort order"),
                              ("fold",       "fold assignment"),
                              ("scheme",     "validation scheme")):
                if not np.array_equal(d[key], ref[key]):
                    raise ValueError(f"'{name}' differs from '{names[0]}' in {what}")
            if not np.allclose(d["times"], ref["times"]):
                raise ValueError(f"'{name}' differs from '{names[0]}' in time grid")
        loaded.append(d)

    times      = ref["times"]
    fold       = ref["fold"]
    y          = ref["y"]
    tumor_col  = ref["tumor_type"]
    # NaN sits in the same cells for every model (the time window follows from
    # the split alone), so a plain weighted sum keeps the pattern intact.
    surv       = np.sum([wi * d["surv"].astype(np.float64) for wi, d in zip(w, loaded)], axis=0)

    if verbose:
        print(f"Ensemble    : {', '.join(f'{n} (w={wi:.3f})' for n, wi in zip(names, w))}")
        print(f"Cohort      : {len(fold)} samples, {int((fold >= 0).sum())} in a test fold")

    results = _empty_fold_results(tumor_types)

    for fold_num in sorted(int(f) for f in np.unique(fold) if f >= 0):
        rows      = np.where(fold == fold_num)[0]
        train_idx = np.where(fold != fold_num)[0]

        # Columns this fold actually covers; all its rows share one window.
        cols = np.where(np.all(np.isfinite(surv[rows]), axis=0))[0]
        if len(cols) < 2:
            if verbose:
                print(f"  ⚠ Fold {fold_num}: fewer than 2 usable time points, skipping")
            continue

        times_f = times[cols]
        surv_f  = surv[np.ix_(rows, cols)]

        _, expected_survival = _median_expected_survival(surv_f, times_f)
        surv_preds = [
            interp1d(times_f, row, kind="previous", bounds_error=False,
                     fill_value=(row[0], row[-1]))
            for row in surv_f
        ]

        for tt in tumor_types:
            mask = (
                np.ones(len(rows), dtype=bool) if tt == "Overall"
                else (tumor_col[rows].astype(str) == tt)
            )
            result = _eval_group__time_to_event(
                tt=tt, mask=mask,
                y_test=y[rows], y_train=y[train_idx],
                surv_preds=surv_preds,
                risk_scores=-expected_survival,
                expected_survival=expected_survival,
                fold_num=fold_num,
                # times_f is global_times already clipped to this fold's valid
                # range; _eval_group clips it again to a sub-range of exactly
                # that window, so it lands on the identical grid it would have
                # derived from global_times itself.
                global_times=times_f,
                verbose=verbose,
            )
            if result is not None:
                _accumulate(results, tt, result)
                if verbose:
                    print(
                        f"  [ENS{fold_num}] {tt:18s}  n={result['n_samples']:3d}  "
                        f"IBS={result['brier_pycox']:.4f}"
                    )

    _print_summary__time_to_event(
        results, tumor_types, ibs_output_file=ibs_output_file, verbose=verbose,
    )
    return results


@beartype
def _interpolated_median(surv: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    First time each row of *surv* crosses 0.5, linearly interpolated between the
    two grid points that bracket the crossing. NaN where a row never gets there.

    ``_median_expected_survival`` snaps to the nearest grid point instead, which
    is why a model with 20 time bins produces at most 20 distinct medians for a
    whole cohort and its point cloud collapses into stacks. Interpolating costs
    nothing here and makes the median continuous whatever grid it is read off.

    Deliberately not folded back into ``_median_expected_survival``: that one
    feeds the cross-validation metrics and the calibrators, and changing it
    would move every reported number for a plotting concern.
    """
    below = surv <= 0.5
    found = below.any(axis=1)

    median = np.full(surv.shape[0], np.nan)
    rows   = np.flatnonzero(found)
    if rows.size == 0:
        return median

    hi = np.argmax(below[rows], axis=1)   # first column at or under 0.5
    lo = np.maximum(hi - 1, 0)            # hi == 0: the curve opens under 0.5

    s_lo, s_hi = surv[rows, lo], surv[rows, hi]
    t_lo, t_hi = times[lo],      times[hi]

    drop = s_lo - s_hi
    # drop == 0 only where lo and hi are the same column, which is the hi == 0
    # case: there is nothing to interpolate against, so the first grid point
    # stands as the answer
    frac = np.where(drop > 0, (s_lo - 0.5) / np.where(drop > 0, drop, 1.0), 0.0)
    median[rows] = t_lo + frac * (t_hi - t_lo)
    return median


@beartype
def merge_grade_curves(members: dict[str, dict]) -> pd.DataFrame:
    """
    Average the reference-cohort survival curves of several models and read a
    median and an expected survival off the merged curve.

    *members* maps a model name to its parsed ``.info.json`` (a ``Predictor``'s
    ``.data``), of which three keys are used: ``grade_only_results`` with its
    per-sample ``curve``, ``grade curve times`` and ``grade curve horizon``. All
    members must share the time grid, the cohort order and the labels, which is
    checked here -- a mismatch would average curves belonging to different
    patients.

    The same rule as ``run_ensemble__time_to_event``: equal weights, averaged on
    the curves rather than on any per-model summary, and the median taken from
    the merged curve afterwards. Averaging the models' own medians instead would
    be a different estimator, and the report's sample marker is read the first
    way -- the reference cloud has to be read the same way, or the marker and
    the cloud it is compared against are not on one scale.

    Returns
    -------
    pd.DataFrame
        ``median_survival``, ``expected_survival`` (both in days), ``WHO Grade``
        and ``Tumor Type`` -- the columns ``outputs.plot_survival`` expects of
        its *grade_only_results*.
    """
    names = list(members)
    if not names:
        raise ValueError("no members given")

    ref, frames, horizons = None, [], []
    for name in names:
        data  = members[name]
        frame = data["grade_only_results"]
        if "curve" not in frame.columns:
            raise ValueError(f"'{name}' carries no reference curves")

        if ref is None:
            ref, times = frame, np.asarray(data["grade curve times"], dtype=float)
        else:
            if not np.array_equal(np.asarray(data["grade curve times"], dtype=float), times):
                raise ValueError(f"'{name}' differs from '{names[0]}' in time grid")
            for col, what in (("WHO Grade",  "WHO grade labels"),
                              ("Tumor Type", "tumor type labels")):
                if not frame[col].astype(str).equals(ref[col].astype(str)):
                    raise ValueError(f"'{name}' differs from '{names[0]}' in {what}")

        frames.append(np.asarray(frame["curve"].tolist(), dtype=np.float64))
        if data.get("grade curve horizon") is not None:
            horizons.append(float(data["grade curve horizon"]))

    surv = np.mean(frames, axis=0)

    # Past the shortest member's horizon the mean is part prediction and part
    # carried-forward last value (see curve_on_grid), and which part is which
    # changes from column to column. A median read there would say more about
    # who ran out first than about the patient, so the grid is cut back to where
    # every member still speaks.
    if horizons:
        t_max = min(horizons)
        keep  = times <= t_max
        if not keep.all():
            print(f"Grade curves : grid cut to {t_max:.0f} d "
                  f"({int((~keep).sum())} of {len(times)} points past the "
                  f"shortest member's horizon)")
            times, surv = times[keep], surv[:, keep]

    return pd.DataFrame({
        "median_survival":   _interpolated_median(surv, times),
        "expected_survival": np.trapezoid(surv, times, axis=1),
        "WHO Grade":         ref["WHO Grade"].astype(str).to_numpy(),
        "Tumor Type":        ref["Tumor Type"].astype(str).to_numpy(),
    })


@beartype
def run_cross_validation__time_to_event_lodo(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    cv_output_file: str,
    tumor_types: list[str] | None = None,
    min_test_samples: int = 15,
    ibs_output_file: str | None = None,
    restrict_to_lodo_datasets: list[str] | None = None,
    X_unlabelled: np.ndarray | None = None,
) -> dict:
    """
    Leave-One-Dataset-Out (LODO) cross-validation for a survival pipeline.

    For each dataset in ``metadata['Dataset']`` with at least
    ``min_test_samples`` samples, that dataset is held out as test set and
    the remaining samples are used for training.  Datasets below the
    threshold are never used as test set (they remain in the training set
    of every fold).

    ``restrict_to_lodo_datasets`` narrows that selection further: only
    datasets in this list are held out. Like the ``min_test_samples``
    threshold it only controls which datasets become a *test* set — the
    excluded ones stay in the training set of every fold, so the model is
    still fitted on all data.

    Returns
    -------
    dict
        ``results[tumor_type][metric]`` — per-fold lists, identical
        structure to ``run_cross_validation__time_to_event``.
    """
    if tumor_types is None:
        tumor_types = _DEFAULT_TUMOR_TYPES

    results      = _empty_fold_results(tumor_types)
    global_times = np.linspace(y["time"].min(), y["time"].max() - 0.5, 50)

    if os.path.exists(cv_output_file):
        os.remove(cv_output_file)

    all_datasets = metadata["Dataset"].unique()
    print("All datasets: " + ", ".join(sorted(all_datasets)))
    
    eligible     = [
        ds for ds in sorted(all_datasets)
        if (metadata["Dataset"] == ds).sum() >= min_test_samples
    ]
    print("Eligible: " + ", ".join(eligible))

    if restrict_to_lodo_datasets is not None:
        requested = set(restrict_to_lodo_datasets)

        # Fail loudly on requests that cannot be honoured: silently dropping
        # them would run a LODO with fewer folds than asked for, which is easy
        # to miss in the log and invalidates any comparison across runs.
        unknown   = requested - set(all_datasets)
        too_small = (requested & set(all_datasets)) - set(eligible)
        if unknown or too_small:
            reasons = []
            if unknown:
                reasons.append(f"not present in metadata['Dataset']: {sorted(unknown)}")
            if too_small:
                reasons.append(f"fewer than min_test_samples={min_test_samples} samples: {sorted(too_small)}")
            raise ValueError(
                "restrict_to_lodo_datasets contains datasets that cannot be held out - "
                + "; ".join(reasons)
            )

        eligible = [ds for ds in eligible if ds in requested]

    restricted_str = "" if restrict_to_lodo_datasets is None else ", restricted"
    print(
        f"LODO: {len(eligible)}/{len(all_datasets)} datasets "
        f"meet n >= {min_test_samples}{restricted_str} and will be used as test set"
    )

    if not eligible:
        print("  ⚠ No datasets left to hold out; nothing to evaluate")

    for fold_num, dataset in enumerate(eligible, start=1):
        test_mask = (metadata["Dataset"] == dataset).values
        train_idx = np.where(~test_mask)[0]
        test_idx  = np.where(test_mask)[0]

        print(
            f"\n[LODO {fold_num}/{len(eligible)}] "
            f"Test dataset: {dataset}  (n={len(test_idx)})"
        )

        fold_metrics = _run_fold__time_to_event(
            pipeline=pipeline,
            X_train=X[train_idx], X_test=X[test_idx],
            y_train=y[train_idx], y_test=y[test_idx],
            m_test=metadata.iloc[test_idx],
            tumor_types=tumor_types,
            output_file=cv_output_file,
            fold_num=fold_num,
            global_times=global_times,
            verbose=True,
            dataset_name=dataset,
            X_unlabelled=X_unlabelled,
        )
        if fold_metrics is None:
            continue
        for tt, metrics in fold_metrics.items():
            _accumulate(results, tt, metrics)

    _print_summary__time_to_event(results, tumor_types, ibs_output_file=ibs_output_file)
    print(f"\nPredictions exported to: {cv_output_file}")
    if os.path.exists(cv_output_file):
        print(f"Total samples: {len(pd.read_csv(cv_output_file, sep=chr(9)))}")
    return results


@beartype
def run_cross_validation__classification(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    cv_output_file: str,
    groups: list[str] | None = None,
    group_column: str = "Tumor Type",
    label_encoder=None,
    n_splits: int = 5,
    random_state: int = 42,
    metrics_output_file: str | None = None,
    patient_column: str | None = "Patient ID",
) -> dict:
    """
    Stratified K-fold cross-validation for a class-based (classification)
    pipeline - the counterpart of ``run_cross_validation__time_to_event``,
    for endpoints predicting a class label (sex, tumor subtype, location)
    instead of a survival time.

    ``y`` holds the integer-encoded labels; pass the fitted ``LabelEncoder``
    as ``label_encoder`` to get readable labels in the TSV and the report.

    Unlike the time-to-event version the split is stratified on ``y``, so
    every fold keeps the class balance of the full cohort. On top of that all
    samples of one patient stay in the same fold - see ``_cv_splits``; the
    grouping constraint wins where the two pull apart. Pass
    ``patient_column=None`` to switch the grouping off.

    Note that ``patient_column`` (fold grouping) and ``group_column``
    (evaluation subsets) are unrelated.

    ``groups`` are evaluated separately, "Overall" meaning all samples; other
    names are matched against ``metadata[group_column]``.

    Returns
    -------
    dict
        ``results[group][metric]`` - per-fold lists.
        See ``_METRIC_KEYS__CLASSIFICATION`` for the full set of metrics.
        The dict is JSON-serialisable and can be handed to
        ``export_classifier(performance={'cv': results})`` as-is.
    """
    if groups is None:
        groups = list(_DEFAULT_GROUPS__CLASSIFICATION)

    splits  = _cv_splits(X, metadata, n_splits, random_state, patient_column, stratify=y)
    results = _empty_fold_results(groups, metric_keys=_METRIC_KEYS__CLASSIFICATION)

    if os.path.exists(cv_output_file):
        os.remove(cv_output_file)
    if os.path.dirname(cv_output_file):
        os.makedirs(os.path.dirname(cv_output_file), exist_ok=True)

    # Out-of-fold predictions, pooled for the final classification report.
    oof_true: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_result = _run_fold__classification(
            pipeline=pipeline,
            X_train=X[train_idx], X_test=X[test_idx],
            y_train=y[train_idx], y_test=y[test_idx],
            m_test=metadata.iloc[test_idx],
            groups=groups,
            group_column=group_column,
            output_file=cv_output_file,
            fold_num=fold_num,
            label_encoder=label_encoder,
            verbose=True,
        )
        if fold_result is None:
            continue
        fold_metrics, y_pred = fold_result
        oof_true.append(y[test_idx])
        oof_pred.append(y_pred)
        for group, metrics in fold_metrics.items():
            _accumulate(results, group, metrics, metric_keys=_METRIC_KEYS__CLASSIFICATION)

    _print_summary__classification(
        results, groups,
        y_true=np.concatenate(oof_true) if oof_true else None,
        y_pred=np.concatenate(oof_pred) if oof_pred else None,
        label_encoder=label_encoder,
        metrics_output_file=metrics_output_file,
    )
    print(f"\nPredictions exported to: {cv_output_file}")
    if os.path.exists(cv_output_file):
        print(f"Total samples: {len(pd.read_csv(cv_output_file, sep=chr(9)))}")
    return results


@beartype
def run_cross_validation__classification_lodo(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    cv_output_file: str,
    groups: list[str] | None = None,
    group_column: str = "Tumor Type",
    label_encoder=None,
    min_test_samples: int = 15,
    metrics_output_file: str | None = None,
    restrict_to_lodo_datasets: list[str] | None = None,
) -> dict:
    """
    Leave-One-Dataset-Out (LODO) cross-validation for a class-based pipeline -
    the counterpart of ``run_cross_validation__time_to_event_lodo``.

    For each dataset in ``metadata['Dataset']`` with at least
    ``min_test_samples`` samples, that dataset is held out as test set and
    the remaining samples are used for training.  Datasets below the
    threshold are never used as test set (they remain in the training set
    of every fold).

    ``restrict_to_lodo_datasets`` narrows that selection further: only
    datasets in this list are held out. Like the ``min_test_samples``
    threshold it only controls which datasets become a *test* set - the
    excluded ones stay in the training set of every fold, so the model is
    still fitted on all data.

    Note that a held-out dataset can be single-class (e.g. an all-female
    cohort for the sex endpoint); balanced accuracy and ROC-AUC are then
    undefined and come out as NaN for that fold.

    Returns
    -------
    dict
        ``results[group][metric]`` - per-fold lists, identical structure to
        ``run_cross_validation__classification``.
    """
    if groups is None:
        groups = list(_DEFAULT_GROUPS__CLASSIFICATION)

    results = _empty_fold_results(groups, metric_keys=_METRIC_KEYS__CLASSIFICATION)

    if os.path.exists(cv_output_file):
        os.remove(cv_output_file)
    if os.path.dirname(cv_output_file):
        os.makedirs(os.path.dirname(cv_output_file), exist_ok=True)

    all_datasets = metadata["Dataset"].unique()
    print("All datasets: " + ", ".join(sorted(all_datasets)))

    eligible     = [
        ds for ds in sorted(all_datasets)
        if (metadata["Dataset"] == ds).sum() >= min_test_samples
    ]
    print("Eligible: " + ", ".join(eligible))

    if restrict_to_lodo_datasets is not None:
        requested = set(restrict_to_lodo_datasets)

        # Fail loudly on requests that cannot be honoured: silently dropping
        # them would run a LODO with fewer folds than asked for, which is easy
        # to miss in the log and invalidates any comparison across runs.
        unknown   = requested - set(all_datasets)
        too_small = (requested & set(all_datasets)) - set(eligible)
        if unknown or too_small:
            reasons = []
            if unknown:
                reasons.append(f"not present in metadata['Dataset']: {sorted(unknown)}")
            if too_small:
                reasons.append(f"fewer than min_test_samples={min_test_samples} samples: {sorted(too_small)}")
            raise ValueError(
                "restrict_to_lodo_datasets contains datasets that cannot be held out - "
                + "; ".join(reasons)
            )

        eligible = [ds for ds in eligible if ds in requested]

    restricted_str = "" if restrict_to_lodo_datasets is None else ", restricted"
    print(
        f"LODO: {len(eligible)}/{len(all_datasets)} datasets "
        f"meet n >= {min_test_samples}{restricted_str} and will be used as test set"
    )

    if not eligible:
        print("  ⚠ No datasets left to hold out; nothing to evaluate")

    # Out-of-fold predictions, pooled for the final classification report.
    # These cover only the held-out datasets, not the whole cohort.
    oof_true: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []

    for fold_num, dataset in enumerate(eligible, start=1):
        test_mask = (metadata["Dataset"] == dataset).values
        train_idx = np.where(~test_mask)[0]
        test_idx  = np.where(test_mask)[0]

        print(
            f"\n[LODO {fold_num}/{len(eligible)}] "
            f"Test dataset: {dataset}  (n={len(test_idx)})"
        )

        fold_result = _run_fold__classification(
            pipeline=pipeline,
            X_train=X[train_idx], X_test=X[test_idx],
            y_train=y[train_idx], y_test=y[test_idx],
            m_test=metadata.iloc[test_idx],
            groups=groups,
            group_column=group_column,
            output_file=cv_output_file,
            fold_num=fold_num,
            label_encoder=label_encoder,
            verbose=True,
            dataset_name=dataset,
        )
        if fold_result is None:
            continue
        fold_metrics, y_pred = fold_result
        oof_true.append(y[test_idx])
        oof_pred.append(y_pred)
        for group, metrics in fold_metrics.items():
            _accumulate(results, group, metrics, metric_keys=_METRIC_KEYS__CLASSIFICATION)

    _print_summary__classification(
        results, groups,
        y_true=np.concatenate(oof_true) if oof_true else None,
        y_pred=np.concatenate(oof_pred) if oof_pred else None,
        label_encoder=label_encoder,
        metrics_output_file=metrics_output_file,
    )
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
    _print_summary__time_to_event during CV). Per-fold IBS cannot be recomputed from the
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
    _print_summary__time_to_event during LODO). Per-fold IBS cannot be recomputed from the
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
    output_suffix: str | None = None,
    export_individual: bool = False,
    patient_column: str | None = "Patient ID",
    X_unlabelled: np.ndarray | None = None,
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

    The folds are the patient-grouped ones of ``_cv_splits`` and are shared by
    every combo, so the combos stay comparable among themselves and with a
    ``run_cross_validation__time_to_event`` run of the same *random_state*.

    ``output_suffix`` is appended to the filenames of the summary CSV (and of
    the individual CSVs when ``export_individual`` is set), so that separate
    runs of the same classifier (e.g. a different feature set) do not
    overwrite each other's results.

    ``export_individual`` writes the per-patient predictions of every combo to
    ``grid_search/individual/``. It is off by default: the filename embeds the
    full combo tag, which overflows the filesystem's name limit for grids with
    many parameters.

    Returns
    -------
    pd.DataFrame
        One row per (combo, tumor_type) with mean ± std for all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    if export_individual:
        os.makedirs(os.path.join(output_dir, "grid_search/individual"), exist_ok=True)

    if classifier_name is None:
        classifier_name = _derive_name(classifier_class)
    if tumor_types is None:
        tumor_types = _DEFAULT_TUMOR_TYPES

    suffix_str = f"__{output_suffix}" if output_suffix else ""

    summary_file = os.path.join(
        output_dir, f"grid_search/grid_search__{classifier_name}{suffix_str}__summary.csv"
    )

    all_combos = [
        dict(zip(param_grid.keys(), combo))
        for combo in itertools.product(*param_grid.values())
    ]

    print(f"Classifier  : {classifier_name}")
    print(f"Combos      : {len(all_combos)}")
    print(f"CV folds    : {n_splits}")
    print(f"Output dir  : {output_dir}")

    splits       = _cv_splits(X, metadata, n_splits, random_state, patient_column)
    global_times = np.linspace(y["time"].min(), y["time"].max() - 0.5, 50)
    summary_rows: list[dict] = []

    for combo_idx, params in enumerate(tqdm(all_combos, desc="Grid combos"), start=1):
        tag         = _combo_tag(params)
        output_file = os.path.join(
            output_dir,
            f"grid_search/individual/grid_search__{classifier_name}__{tag}{suffix_str}.csv",
        ) if export_individual else None
        print(f"\n[{combo_idx}/{len(all_combos)}] {tag}")

        try:
            pipeline     = build_pipeline(classifier_class, params)
            fold_results = _empty_fold_results(tumor_types)

            for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
                try:
                    fold_metrics = _run_fold__time_to_event(
                        pipeline=pipeline,
                        X_train=X[train_idx], X_test=X[test_idx],
                        y_train=y[train_idx], y_test=y[test_idx],
                        m_test=metadata.iloc[test_idx],
                        tumor_types=tumor_types,
                        output_file=output_file,
                        fold_num=fold_num,
                        global_times=global_times,
                        verbose=False,
                        X_unlabelled=X_unlabelled,
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
            for k in _METRIC_KEYS__TIME_TO_EVENT:
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
