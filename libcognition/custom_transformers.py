#!/usr/bin/env python3

# Data manipulation
import numpy as np
import pandas as pd
import random

# Scikit-learn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# Scikit-survival
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.functions import StepFunction

# Lifelines
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.statistics import logrank_test

# Progress
from tqdm.auto import tqdm

# PyTorch/Pycox
import torch
import torchtuples as tt
from torchtuples.practical import MLPVanilla
from pycox.models import CoxPH, DeepHitSingle, LogisticHazard, PCHazard
from pycox.evaluation import EvalSurv



def _to_tensor(X):
    if isinstance(X, torch.Tensor):
        return X.detach().clone().to(dtype=torch.float32)
    if isinstance(X, np.ndarray):
        return torch.tensor(X.astype("float32"))
    return torch.tensor(X.values.astype("float32"))


def _cpu_state_dict(module):
    """State dict waarvan alle tensors op CPU staan, zonder het model te verplaatsen.

    Een op GPU getraind netwerk levert CUDA-tensors op. Die zijn niet te
    unpicklen op een machine zonder GPU: torch weigert CUDA-storage te
    deserialiseren ("Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False"). Door bij export naar CPU te kopieren
    is het .joblib-bestand device-onafhankelijk.

    Gebruik dit dus niet als module.cpu().state_dict(): die variant verplaatst
    het levende model naar CPU terwijl self.device_ nog naar cuda wijst,
    waardoor voorspellen na een export/dump op een GPU-machine stukloopt.
    """
    return {
        k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
        for k, v in module.state_dict().items()
    }


def _default_device():
    """GPU als die er is, anders CPU - de fallback bij het laden van modellen."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_batch_size(n_samples, batch_size):
    """Verklein batch_size tot de laatste batch niet exact 1 sample bevat.

    BatchNorm1d kan in trainingsmodus geen variantie schatten over één sample en
    faalt dan met "Expected more than 1 value per channel when training".
    torchtuples' make_dataloader accepteert geen drop_last, dus corrigeren we de
    batchgrootte in plaats van de restbatch te laten vallen - zo blijven alle
    samples in het model. Treedt alleen op bij n_samples % batch_size == 1, wat
    per CV/LODO-fold verschilt en dus maar af en toe één fold sloopt.
    """
    if n_samples % batch_size != 1:
        return batch_size

    new_batch_size = batch_size
    while new_batch_size > 1 and n_samples % new_batch_size == 1:
        new_batch_size -= 1

    print(
        f"  ⚠ batch size has been lowered from {batch_size} to {new_batch_size} "
        f"because {n_samples} training samples leave a final batch of 1, "
        f"which BatchNorm cannot process"
    )
    return new_batch_size


def curve_on_grid(sf, times):
    """Evalueer een ``StepFunction`` op *times*, vastgehouden op zijn eigen
    domeinrand waar het rooster daar voorbij loopt.

    ``sksurv.functions.StepFunction`` gooit een ValueError voor alles buiten
    zijn domein, en dat domein is het eigen tijdsbereik van het model -- de
    langste event-tijd die het zag, het laatste van zijn ``num_durations``
    cut points. Geen enkel gedeeld rooster valt daar op vast te pinnen. De
    query afknippen draagt in plaats daarvan de laatst voorspelde waarde
    door, en dat is ook wat een overlevingscurve voorbij zijn horizon zegt:
    er is niets waargenomen dat hem nog verandert.
    """
    lo, hi = sf.domain
    return sf(np.clip(np.asarray(times, dtype=float), lo, hi))


def _reindex_survival(surv_df, times):
    """Herbemonster een survivalcurve op *times*, met S(0)=1 als ankerpunt.

    Modellen met kwantiel-tijdstippen (_make_time_cuts) beginnen pas bij het
    2%-kwantiel van de trainingstijden. Wordt er een tijdstip daarvóór
    opgevraagd, dan heeft ffill() geen voorganger en blijft de waarde NaN. Die
    propageert via np.trapezoid naar expected_survival, waarna np.polyfit in
    de regressiemetrieken afgaat met "SVD did not converge" of een DLASCL-
    klacht uit LAPACK. Het treedt alleen op in folds waar train en test
    dezelfde minimale tijd hebben, want pas dan blijft het laagste punt van
    het evaluatierooster staan - vandaar dat het maar af en toe een fold
    sloopt. S(0)=1 is per definitie waar en maakt de curve over het volledige
    bereik gedefinieerd. Cox-achtige modellen indexeren op de waargenomen
    trainingstijden en hebben dit niet nodig; daar is de guard een no-op.
    """
    if 0.0 not in surv_df.index:
        anchor  = pd.DataFrame([[1.0] * surv_df.shape[1]],
                               index=[0.0], columns=surv_df.columns)
        surv_df = pd.concat([anchor, surv_df])

    new_index = surv_df.index.union(times)
    return surv_df.reindex(new_index).ffill().loc[times]


__all__ = [
    'curve_on_grid',
    '_logrank_feature_quantiles',
    '_logrank_feature',
    'fast_logrank_score_parallel',
    'fast_logrank_score_vectorized',

    'SurvivalPipeline',
    'SurvivalPredictorExtendedBase',

    'WeibullAFTFitterWrapperExtended',
    'LogNormalAFTFitterWrapperExtended',
    'LogLogisticAFTFitterWrapperExtended',
    'CoxPHFitterWrapper',
    'CoxPHFitterWrapperExtended',
    'CoxnetSurvivalAnalysisWrapperExtended',
    'RandomSurvivalForestWrapper',
    'RandomSurvivalForestWrapperExtended',
    'GradientBoostingSurvivalAnalysisWrapper',
    'GradientBoostingSurvivalAnalysisWrapperExtended',
    'ComponentwiseGradientBoostingSurvivalAnalysisWrapper',
    'ComponentwiseGradientBoostingSurvivalAnalysisWrapperExtended',
    'XGBoostAFTWrapper',
    'XGBoostAFTWrapperExtended',
    'DeepSurvWrapper',
    'DeepSurvWrapperExtended',
    'DeepHitWrapper',
    'DeepHitWrapperExtended',
    'PCHazardWrapper',
    'PCHazardWrapperExtended',
    'LogisticHazardWrapper',
    'LogisticHazardWrapperExtended',
    'LuckSurvivalWrapperExtended',
    'DeepCoxIBrierWrapperExtended',
    'MTLRSklearnWrapper',
    'MTLRSklearnWrapperExtended',
    'MTLRDeepWrapper',
    'MTLRDeepWrapperExtended',
]


def fast_logrank_score_vectorized(X, y):
    """
    Volledig gevectoriseerde logrank over alle features tegelijk.
    Geen parallelisatie nodig — BLAS matrix multiplicatie doet het werk.
    
    Geheugen: ~300 MB bij 85k features, 1600 samples, 150 event times.
    """
    durations = y["time"]
    events = y["event"].astype(bool)

    # Mediaan split voor alle features tegelijk: (n_samples, n_features)
    medians = np.median(X, axis=0)
    group_high = (X >= medians).astype(np.float32)  # float32 voor BLAS

    # Unieke event tijden
    event_times = np.unique(durations[events])  # (n_times,)

    # (n_samples, n_times)
    at_risk   = (durations[:, None] >= event_times[None, :]).astype(np.float32)
    event_mat = ((durations[:, None] == event_times[None, :]) & events[:, None]).astype(np.float32)

    # Per tijdstip: totaal at risk en events  — (n_times,)
    n = at_risk.sum(axis=0)
    d = event_mat.sum(axis=0)

    # Per feature + tijdstip via matmul: (n_features, n_times)
    n1 = group_high.T @ at_risk    # BLAS sgemm
    d1 = group_high.T @ event_mat  # BLAS sgemm

    valid = (n > 1).astype(np.float32)
    n_safe = np.where(n > 1, n, 1.0)

    # Logrank U en V over tijdstappen
    E1 = n1 * (d / n_safe) * valid                                          # (n_features, n_times)
    U  = (d1 - E1).sum(axis=1)                                              # (n_features,)

    V_terms = (n1 * (n_safe - n1) * d * np.where(n > 1, n_safe - d, 0)
               / (n_safe ** 2 * np.where(n > 1, n_safe - 1, 1))) * valid
    V = V_terms.sum(axis=1)                                                  # (n_features,)

    scores = np.where(V > 0, U ** 2 / V, 0.0)
    pvalues = np.full_like(scores, np.nan)
    return scores, pvalues


def _logrank_feature_quantiles(i, X, durations, events, method="sqrt_sum"):
    try:
        feature = X[:, i]
        q25, q50, q75 = np.quantile(feature, [0.25, 0.5, 0.75])
        groups = np.digitize(feature, bins=[q25, q50, q75])
        stats = []

        for g1, g2 in [(0, 1), (1, 2), (2, 3), (0, 3)]:
            mask1 = groups == g1
            mask2 = groups == g2
            if np.any(mask1) and np.any(mask2):
                result = logrank_test(
                    durations[mask1], durations[mask2],
                    events[mask1], events[mask2]
                )
                stats.append(result.test_statistic)

        if not stats:
            return 0.0

        if method == "mean":
            return np.mean(stats)
        elif method == "squared_mean":
            return np.mean(np.square(stats))
        elif method == "sqrt_sum":
            return np.sum(np.sqrt(stats))
        elif method == "max":
            return np.max(stats)
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception:
        return 0.0


def _logrank_feature(i, X, durations, events):
    try:
        median_val = np.median(X[:, i])
        group_high = X[:, i] >= median_val
        results = logrank_test(
            durations[group_high], durations[~group_high],
            events[group_high], events[~group_high]
        )
        return results.test_statistic
    except Exception:
        return 0.0


def fast_logrank_score_parallel(X, y, n_jobs=-1):
    """
    Log-rank score per feature voor gebruik met SelectKBest.
    Gebruikt mediaan-split (1 test per feature, 4x sneller dan kwartiel-versie).

    Returns
    -------
    scores : ndarray, shape (n_features,)
    pvalues : ndarray, shape (n_features,)  — NaN waar niet berekend
    """
    durations = y["time"]
    events = y["event"]

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_logrank_feature)(i, X, durations, events)
        for i in tqdm(range(X.shape[1]), desc="Logrank scoring")
    )

    scores = np.array(results)
    pvalues = np.full_like(scores, np.nan)  # expliciet NaN i.p.v. misleidende zeros
    return scores, pvalues


def variance_score(X, y):
    """Variance per feature voor gebruik met SelectKBest (pickleable alternatief voor lambda)."""
    return np.var(X, axis=0), np.ones(X.shape[1])




class SurvivalPredictorExtendedBase(ABC):
    """
    Abstract base class for all Extended survival model wrappers.

    Subclasses must implement predict_risk_scores, predict_survival, and
    predict_median. The default predict_all delegates to these three methods.
    """

    @abstractmethod
    def predict_risk_scores(self, X):
        """Return numeric risk scores (higher = higher risk)."""
        ...

    @abstractmethod
    def predict_survival(self, X, times=None):
        """Return a list of StepFunction survival curves, one per sample."""
        ...

    @abstractmethod
    def predict_median(self, X):
        """Return median survival times as an ndarray."""
        ...

    def predict_all(self, X, times=None):
        """Return all prediction outputs as a dict."""
        return {
            "risk_scores": self.predict_risk_scores(X),
            "survival_functions": self.predict_survival(X, times=times),
            "median": self.predict_median(X),
        }


class CoxPHFitterWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, penalizer=0.0, l1_ratio=0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        data = pd.DataFrame(X).assign(time=y['time'], event=y['event'])
        self.model_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model_.fit(data, duration_col='time', event_col='event')
        return self

    def predict(self, X, times=None):
        check_is_fitted(self)
        df = pd.DataFrame(X)
        surv_curves = self.model_.predict_survival_function(df, times=times)
        return [StepFunction(surv_curves.index.values, surv_curves.iloc[:, i].values)
                for i in range(surv_curves.shape[1])]


class CoxPHFitterWrapperExtended(CoxPHFitterWrapper, SurvivalPredictorExtendedBase):
    """Extended wrapper with survival-specific helpers."""

    def predict_risk_scores(self, X):
        df = pd.DataFrame(X, index=np.arange(len(X)))
        return self.model_.predict_partial_hazard(df).values

    def predict_survival(self, X, times=None):
        df = pd.DataFrame(X, index=np.arange(len(X)))
        surv_curves = self.model_.predict_survival_function(df, times=times)
        return [
            StepFunction(surv_curves.index.values, surv_curves.iloc[:, i].values)
            for i in range(surv_curves.shape[1])
        ]

    def predict_median(self, X):
        df = pd.DataFrame(X, index=np.arange(len(X)))
        return self.model_.predict_median(df)


class CoxnetSurvivalAnalysisWrapperExtended(CoxnetSurvivalAnalysis, SurvivalPredictorExtendedBase):
    """Extended CoxnetSurvivalAnalysis with convenience prediction helpers."""

    def __init__(
        self,
        l1_ratio=0.5,
        alphas=None,
        # n_alphas, alpha_min_ratio en penalty_factor doen alleen iets als
        # alphas None is: sksurv bepaalt dan zelf het pad, van alpha_max
        # (kleinste alpha waarbij alle coefficienten nul zijn) tot
        # alpha_max * alpha_min_ratio, in n_alphas log-stappen. "auto" =
        # 0.01 bij n_samples < n_features, anders 0.0001.
        n_alphas=100,
        alpha_min_ratio="auto",
        # Per-feature gewicht op de penalty (array van lengte n_features);
        # None = alles 1. Een 0 haalt de penalty van die feature af, zodat hij
        # altijd in het model blijft.
        penalty_factor=None,
        # Centreren/schalen binnen Coxnet zelf. De pipeline zet al een
        # StandardScaler voor de classifier, dus dit is daar bovenop.
        normalize=False,
        fit_baseline_model=True,
        max_iter=100000,
        tol=1e-7,
        copy_X=False
    ):
        super().__init__(
            l1_ratio=l1_ratio,
            alphas=alphas,
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            penalty_factor=penalty_factor,
            normalize=normalize,
            fit_baseline_model=fit_baseline_model,
            max_iter=max_iter,
            tol=tol,
            copy_X=copy_X
        )
        if not fit_baseline_model:
            print("Warning: fit_baseline_model is False. Survival/Median predictions will fail.")

    def predict_risk_scores(self, X):
        return self.predict(X)

    def predict_survival(self, X, times=None):
        surv_funcs = self.predict_survival_function(X)
        if times is None:
            return surv_funcs
        # sksurv hands back curves on its own grid -- the unique event times of
        # the training set. Silently returning those when a grid was asked for
        # is worse than not accepting the argument at all: the caller believes
        # it got what it asked for, and only notices when its columns turn out
        # to mean different times per model.
        return [StepFunction(np.asarray(times, dtype=float),
                             curve_on_grid(fn, times))
                for fn in surv_funcs]

    def predict_median(self, X):
        surv_funcs = self.predict_survival_function(X)
        median_times = []
        for fn in surv_funcs:
            median_idx = np.where(fn.y <= 0.5)[0]
            if len(median_idx) > 0:
                median_times.append(fn.x[median_idx[0]])
            else:
                median_times.append(np.inf)
        return np.array(median_times)


class WeibullAFTFitterWrapperExtended(BaseEstimator, RegressorMixin, SurvivalPredictorExtendedBase):
    def __init__(self, penalizer=0.0, l1_ratio=0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        df = pd.DataFrame(X).copy()
        df["time"] = y["time"]
        df["event"] = y["event"]
        self.model_ = WeibullAFTFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model_._scipy_fit_method = "SLSQP"
        self.model_.fit(df, duration_col="time", event_col="event")
        return self

    def predict(self, X):
        return self.predict_risk_scores(X)

    def predict_risk_scores(self, X):
        df = pd.DataFrame(X, index=np.arange(len(X)))
        return -self.model_.predict_median(df).values

    def predict_median(self, X):
        df = pd.DataFrame(X, index=np.arange(len(X)))
        return self.model_.predict_median(df).values

    def predict_survival(self, X, times=None):
        df = pd.DataFrame(X, index=np.arange(len(X)))
        psf = self.model_.predict_survival_function(df, times=times)
        return [
            StepFunction(psf.index.values, psf.iloc[:, i].values)
            for i in range(psf.shape[1])
        ]


class LogNormalAFTFitterWrapperExtended(BaseEstimator, RegressorMixin, SurvivalPredictorExtendedBase):
    def __init__(self, penalizer=0.0, l1_ratio=0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        df = pd.DataFrame(X).assign(time=y["time"], event=y["event"])
        self.model_ = LogNormalAFTFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model_._scipy_fit_method = "SLSQP"
        self.model_.fit(df, duration_col="time", event_col="event")
        return self

    def predict(self, X):
        return self.predict_risk_scores(X)

    def predict_risk_scores(self, X):
        check_is_fitted(self)
        df = pd.DataFrame(X)
        return -self.model_.predict_median(df).values

    def predict_median(self, X):
        check_is_fitted(self)
        df = pd.DataFrame(X)
        return self.model_.predict_median(df).values

    def predict_survival(self, X, times=None):
        check_is_fitted(self)
        df = pd.DataFrame(X)
        surv_curves = self.model_.predict_survival_function(df, times=times)
        return [
            StepFunction(surv_curves.index.values, surv_curves.iloc[:, i].values)
            for i in range(surv_curves.shape[1])
        ]


class LogLogisticAFTFitterWrapperExtended(BaseEstimator, RegressorMixin, SurvivalPredictorExtendedBase):
    def __init__(self, penalizer=0.0, l1_ratio=0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        df = pd.DataFrame(X).assign(time=y["time"], event=y["event"])
        self.model_ = LogLogisticAFTFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model_._scipy_fit_method = "SLSQP"
        self.model_.fit(df, duration_col="time", event_col="event")
        return self

    def predict(self, X):
        return self.predict_risk_scores(X)

    def predict_risk_scores(self, X):
        check_is_fitted(self)
        df = pd.DataFrame(X)
        return -self.model_.predict_median(df).values

    def predict_median(self, X):
        check_is_fitted(self)
        df = pd.DataFrame(X)
        return self.model_.predict_median(df).values

    def predict_survival(self, X, times=None):
        check_is_fitted(self)
        df = pd.DataFrame(X)
        surv_curves = self.model_.predict_survival_function(df, times=times)
        return [
            StepFunction(surv_curves.index.values, surv_curves.iloc[:, i].values)
            for i in range(surv_curves.shape[1])
        ]


class RandomSurvivalForestWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 random_state=42,
                 n_jobs=28,
                 n_estimators=75,
                 max_depth=15,
                 min_samples_leaf=7,
                 min_samples_split=8,
                 max_features="sqrt",
                 max_samples=None,
                 verbose=1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        # Aantal kandidaat-features per split. Default "sqrt" komt van sksurv,
        # maar is in PCA-ruimte vaak te krap: van 1000 componenten zijn er dan
        # 31 per split in beeld, terwijl het prognostische signaal in een
        # handvol componenten zit. Een fractie (0.3-1.0) werkt hier meestal
        # beter; hoger = minder decorrelatie tussen de bomen.
        self.max_features = max_features
        # Het RSF-equivalent van subsample: fractie (0.0-1.0) of aantal samples
        # dat per boom getrokken wordt. None = n_samples, de sksurv-default.
        # Let op: dit is een bootstrap-trekking *met* teruglegging, anders dan
        # de subsample van GradientBoostingSurvivalAnalysis, die zonder
        # teruglegging trekt. Werkt alleen zolang bootstrap=True (sksurv-default).
        self.max_samples = max_samples
        self.verbose = verbose

    def fit(self, X, y):
        self.model_ = RandomSurvivalForest(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            max_samples=self.max_samples,
            verbose=self.verbose
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        # Ensemble mortality: de som van de cumulatieve hazard over alle
        # event-tijden. Continu en gebruikt de hele curve, in tegenstelling tot
        # een aflezing op één tijdstip (zoals 1 - S(t_max)), die veel ties
        # oplevert en de vroege separatie tussen curves negeert.
        return self.model_.predict(X)


class RandomSurvivalForestWrapperExtended(RandomSurvivalForestWrapper, SurvivalPredictorExtendedBase):
    """Extended wrapper for RandomSurvivalForest with survival-specific helpers."""

    def predict_risk_scores(self, X):
        return self.model_.predict(X)

    def predict_survival(self, X, times=None):
        surv_funcs = self.model_.predict_survival_function(X)
        return [
            StepFunction(fn.x, fn(fn.x)) if times is None else StepFunction(times, fn(times))
            for fn in surv_funcs
        ]

    def predict_median(self, X):
        surv_funcs = self.model_.predict_survival_function(X)
        medians = []
        for fn in surv_funcs:
            median_time = next((t for t, s in zip(fn.x, fn.y) if s <= 0.5), np.nan)
            medians.append(median_time)
        return np.array(medians)


# ---------------------------------------------------------------------------
# GradientBoostingSurvivalAnalysis
# ---------------------------------------------------------------------------

class GradientBoostingSurvivalAnalysisWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 loss="coxph",
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 dropout_rate=0.0,
                 max_depth=3,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 max_features=None,
                 random_state=42,
                 verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.dropout_rate = dropout_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.model_ = GradientBoostingSurvivalAnalysis(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            dropout_rate=self.dropout_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.model_.fit(X, y)

        # sksurv schat alleen een Breslow-baseline bij loss="coxph"; bij de
        # AFT-achtige losses ("ipcwls", "squared") is predict_survival_function()
        # niet beschikbaar en faalt met "`fit` must be called with the loss option
        # set to 'coxph'". Daar schatten we de baseline zelf op de afgeleide
        # risicoscore, zodat survival/median predictions ook dan werken.
        if self.loss == "coxph":
            self.baseline_model_ = None
            self.risk_offset_ = 0.0
        else:
            # predict() geeft dan exp(raw) = voorspelde tijd; log() brengt ons
            # terug op de raw schaal, centreren houdt exp(risk) rond 1.
            log_time = np.log(np.clip(self.model_.predict(X), 1e-12, None))
            self.risk_offset_ = float(np.mean(log_time))
            self.baseline_model_ = BreslowEstimator().fit(
                -(log_time - self.risk_offset_), y["event"], y["time"]
            )
        return self

    def _risk_scores(self, X):
        """Risicoscore op log-hazard schaal (hoger = hoger risico), voor elke loss."""
        pred = self.model_.predict(X)
        if self.loss == "coxph":
            return pred
        # Voorspelde tijd → risico: hoger = korter overleven.
        return -(np.log(np.clip(pred, 1e-12, None)) - self.risk_offset_)

    def predict(self, X):
        return self._risk_scores(X)


class GradientBoostingSurvivalAnalysisWrapperExtended(GradientBoostingSurvivalAnalysisWrapper, SurvivalPredictorExtendedBase):
    """Extended wrapper for GradientBoostingSurvivalAnalysis with survival-specific helpers."""

    def predict_risk_scores(self, X):
        return self._risk_scores(X)

    def _survival_functions(self, X):
        if self.baseline_model_ is None:
            return self.model_.predict_survival_function(X)
        return self.baseline_model_.get_survival_function(self._risk_scores(X))

    def predict_survival(self, X, times=None):
        surv_funcs = self._survival_functions(X)
        return [
            StepFunction(fn.x, fn(fn.x)) if times is None else StepFunction(times, fn(times))
            for fn in surv_funcs
        ]

    def predict_median(self, X):
        surv_funcs = self._survival_functions(X)
        medians = []
        for fn in surv_funcs:
            median_time = next((t for t, s in zip(fn.x, fn.y) if s <= 0.5), np.nan)
            medians.append(median_time)
        return np.array(medians)


# ---------------------------------------------------------------------------
# ComponentwiseGradientBoostingSurvivalAnalysis
# ---------------------------------------------------------------------------

class ComponentwiseGradientBoostingSurvivalAnalysisWrapper(BaseEstimator, RegressorMixin):
    """Boosting met een univariate least-squares base learner in plaats van bomen.

    Per iteratie wordt één feature gekozen en lineair gefit; het eindmodel is
    daarmee een lineaire predictor (coef_) met ingebouwde featureselectie. Het
    boostingpad benadert het lasso-pad, dus qua modelklasse ligt dit dicht bij
    CoxnetSurvivalAnalysis: n_estimators x learning_rate speelt de rol van de
    penalty. Geen max_depth/min_samples_*/max_features - die horen bij bomen.

    Laat dropout_rate op 0.0: > 0 is stuk in sksurv 0.27.0. De schaalfactoren
    die dropout per base learner oplevert (`_scale`) worden alleen tijdens fit
    toegepast; `_raw_predict` van de componentwise klasse telt bij het
    voorspellen alle learners ongeschaald op (de boom-variant heeft daar wel
    een `_dropout_raw_predict` voor). De linear predictor wordt daardoor ordes
    van grootte te groot, exp() ervan overflowt in de Breslow-baseline en
    iedereen krijgt dezelfde survival curve (C-index exact 0.500). Dat geldt
    ook voor de coef_-property hieronder, die diezelfde ongeschaalde som
    aggregeert.
    """

    def __init__(self,
                 loss="coxph",
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 dropout_rate=0.0,
                 random_state=42,
                 verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.model_ = ComponentwiseGradientBoostingSurvivalAnalysis(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.model_.fit(X, y)

        # Zelfde beperking als bij de boom-variant: sksurv schat alleen een
        # Breslow-baseline bij loss="coxph"; bij "ipcwls"/"squared" faalt
        # predict_survival_function(). Daar schatten we de baseline zelf op de
        # afgeleide risicoscore.
        if self.loss == "coxph":
            self.baseline_model_ = None
            self.risk_offset_ = 0.0
        else:
            # predict() geeft dan exp(raw) = voorspelde tijd; log() brengt ons
            # terug op de raw schaal, centreren houdt exp(risk) rond 1.
            log_time = np.log(np.clip(self.model_.predict(X), 1e-12, None))
            self.risk_offset_ = float(np.mean(log_time))
            self.baseline_model_ = BreslowEstimator().fit(
                -(log_time - self.risk_offset_), y["event"], y["time"]
            )
        return self

    def _risk_scores(self, X):
        """Risicoscore op log-hazard schaal (hoger = hoger risico), voor elke loss."""
        pred = self.model_.predict(X)
        if self.loss == "coxph":
            return pred
        # Voorspelde tijd → risico: hoger = korter overleven.
        return -(np.log(np.clip(pred, 1e-12, None)) - self.risk_offset_)

    def predict(self, X):
        return self._risk_scores(X)

    @property
    def coef_(self):
        """Geaggregeerde coëfficiënten; coef_[0] is de intercept (0 bij coxph)."""
        check_is_fitted(self, "model_")
        return self.model_.coef_


class ComponentwiseGradientBoostingSurvivalAnalysisWrapperExtended(ComponentwiseGradientBoostingSurvivalAnalysisWrapper, SurvivalPredictorExtendedBase):
    """Extended wrapper for ComponentwiseGradientBoostingSurvivalAnalysis with survival-specific helpers."""

    def predict_risk_scores(self, X):
        return self._risk_scores(X)

    def _survival_functions(self, X):
        if self.baseline_model_ is None:
            return self.model_.predict_survival_function(X)
        return self.baseline_model_.get_survival_function(self._risk_scores(X))

    def predict_survival(self, X, times=None):
        surv_funcs = self._survival_functions(X)
        return [
            StepFunction(fn.x, fn(fn.x)) if times is None else StepFunction(times, fn(times))
            for fn in surv_funcs
        ]

    def predict_median(self, X):
        surv_funcs = self._survival_functions(X)
        medians = []
        for fn in surv_funcs:
            median_time = next((t for t, s in zip(fn.x, fn.y) if s <= 0.5), np.nan)
            medians.append(median_time)
        return np.array(medians)


# ---------------------------------------------------------------------------
# XGBoost AFT (Accelerated Failure Time)
# ---------------------------------------------------------------------------

# Standaard verdelingen voor de AFT foutterm: log(T) = mu + sigma * Z.
# scipy levert zowel de CDF (voor de survival curve) als de ppf (voor de mediaan).
import scipy.stats as _sp_stats

_AFT_DISTRIBUTIONS = {
    "normal":   _sp_stats.norm,
    "logistic": _sp_stats.logistic,
    "extreme":  _sp_stats.gumbel_l,  # minimum extreme value → Weibull voor T
}


class XGBoostAFTWrapper(BaseEstimator, RegressorMixin):
    """
    XGBoost native Accelerated Failure Time (objective="survival:aft").

    Traint via de native xgb.train API met interval-labels
    (label_lower_bound / label_upper_bound) voor rechts-gecensureerde data.
    predict() geeft de voorspelde overlevingstijd exp(mu) (hoger = langer leven).
    """

    def __init__(self,
                 aft_loss_distribution="normal",       # normal | logistic | extreme
                 aft_loss_distribution_scale=1.0,
                 learning_rate=0.1,
                 n_estimators=100,
                 max_depth=3,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 min_child_weight=1.0,
                 reg_lambda=1.0,
                 reg_alpha=0.0,
                 random_state=42,
                 verbose=False):
        self.aft_loss_distribution = aft_loss_distribution
        self.aft_loss_distribution_scale = aft_loss_distribution_scale
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self.verbose = verbose

    def _extract_times_events(self, y):
        if hasattr(y, "values"):
            times  = y["time"].values.astype("float32")
            events = y["event"].values.astype("float32")
        elif isinstance(y, np.ndarray) and y.dtype.names:
            times  = y["time"].astype("float32")
            events = y["event"].astype("float32")
        else:
            raise ValueError("y moet een DataFrame of structured numpy array zijn met 'time' en 'event'.")
        return times, events

    def fit(self, X, y):
        import xgboost as xgb

        times, events = self._extract_times_events(y)
        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")

        # Rechts-censurering: ondergrens = tijd; bovengrens = tijd bij een event, +inf bij censurering.
        y_lower = times
        y_upper = np.where(events == 1, times, np.inf).astype("float32")

        dtrain = xgb.DMatrix(x_np)
        dtrain.set_float_info("label_lower_bound", y_lower)
        dtrain.set_float_info("label_upper_bound", y_upper)

        params = {
            "objective":                   "survival:aft",
            "eval_metric":                 "aft-nloglik",
            "aft_loss_distribution":       self.aft_loss_distribution,
            "aft_loss_distribution_scale": self.aft_loss_distribution_scale,
            "eta":              self.learning_rate,
            "max_depth":        self.max_depth,
            "subsample":        self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "lambda":           self.reg_lambda,
            "alpha":            self.reg_alpha,
            "seed":             self.random_state,
        }

        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=self.verbose,
        )
        # Default tijd-grid voor survival curves wanneer geen tijden worden meegegeven.
        self.time_grid_ = np.unique(times[times > 0])
        return self

    def predict(self, X):
        import xgboost as xgb
        check_is_fitted(self)
        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")
        return self.model_.predict(xgb.DMatrix(x_np))


class XGBoostAFTWrapperExtended(XGBoostAFTWrapper, SurvivalPredictorExtendedBase):
    """
    Extended XGBoost AFT wrapper met survival-specifieke helpers.

    XGBoost AFT geeft alleen een puntschatting van de overlevingstijd. De volledige
    survival curve wordt analytisch afgeleid uit de AFT verdeling:
        log(T) = log(pred) + sigma * Z,   Z ~ standaard verdeling
        S(t|x) = 1 - F_Z((log t - log pred) / sigma)
    """

    def _dist(self):
        try:
            return _AFT_DISTRIBUTIONS[self.aft_loss_distribution]
        except KeyError:
            raise ValueError(
                f"Onbekende aft_loss_distribution '{self.aft_loss_distribution}'. "
                f"Kies uit: {sorted(_AFT_DISTRIBUTIONS)}."
            )

    def predict_risk_scores(self, X):
        # pred = voorspelde overlevingstijd (hoger = langer leven), dus risico = -pred.
        return -self.predict(X)

    def predict_survival(self, X, times=None):
        pred = self.predict(X)
        t = self.time_grid_ if times is None else np.asarray(times, dtype="float64")

        sigma = self.aft_loss_distribution_scale
        cdf   = self._dist().cdf
        # clip tegen log(0) = -inf; S(0)=1 blijft numeriek correct.
        log_t = np.log(np.clip(t, 1e-8, None))       # (n_times,)
        mu    = np.log(np.clip(pred, 1e-8, None))    # (n_samples,)

        # z[i, k] = (log t_k - mu_i) / sigma  →  S = 1 - F_Z(z)
        z    = (log_t[None, :] - mu[:, None]) / sigma
        surv = 1.0 - cdf(z)                           # (n_samples, n_times)

        return [StepFunction(t, surv[i, :]) for i in range(surv.shape[0])]

    def predict_median(self, X):
        pred = self.predict(X)
        # Mediaan van T = exp(mu + sigma * ppf_Z(0.5)); voor normal/logistic is ppf(0.5)=0 → pred.
        z_median = self._dist().ppf(0.5)
        return pred * np.exp(self.aft_loss_distribution_scale * z_median)


# ---------------------------------------------------------------------------
# DeepSurv
# ---------------------------------------------------------------------------

class DeepSurvWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 num_nodes=[128, 64],
                 dropout=0.34,
                 batch_norm=True,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 batch_size=512,
                 epochs=1500,
                 patience=35,
                 val_fraction=0.0,  # EXPERIMENTEEL, zie fit(); 0.0 = oude gedrag
                 activation=torch.nn.ReLU,  # torch.nn.ELU | torch.nn.Tanh | torch.nn.SELU
                 verbose=False,
                 seed=None):
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.activation = activation
        self.verbose = verbose
        self.seed = seed

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(self, X, y):
        self._set_seed()

        times = y['time'].astype('float32')
        events = y['event'].astype('float32')

        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = (torch.tensor(times, dtype=torch.float32),
               torch.tensor(events, dtype=torch.bool))

        # EXPERIMENTEEL - val_fraction > 0 houdt een deel van de trainingsdata
        # apart voor early stopping. Met val_fraction=0.0 is val_data gelijk aan
        # de trainingsdata: EarlyStopping monitort dan de trainloss, die vrijwel
        # altijd blijft dalen, waardoor de patience nooit afgaat en er in de
        # praktijk exact self.epochs epochs gedraaid worden. Dat is het gedrag
        # waarop alle bestaande CV-resultaten zijn gebaseerd, vandaar dat 0.0
        # de default blijft. De split is gestratificeerd op event, anders kan de
        # val-set bijna alleen censoring bevatten en is de val-loss een slecht
        # stopsignaal.
        if self.val_fraction:
            idx_tr, idx_val = train_test_split(
                np.arange(len(times)), test_size=self.val_fraction,
                stratify=events, random_state=self.seed)
            fit_data = (x_t[idx_tr],  (y_t[0][idx_tr],  y_t[1][idx_tr]))
            val_data = (x_t[idx_val], (y_t[0][idx_val], y_t[1][idx_val]))
        else:
            fit_data = val_data = (x_t, y_t)

        n_covariates = X.shape[1]
        net = MLPVanilla(n_covariates, self.num_nodes, 1,
                         self.batch_norm, self.dropout,
                         activation=self.activation)
        optimizer = tt.optim.Adam(lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        self.model_ = CoxPH(net, optimizer)

        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        batch_size = _safe_batch_size(len(fit_data[0]), self.batch_size)
        log = self.model_.fit(*fit_data, batch_size, self.epochs,
                              callbacks, self.verbose, val_data=val_data)

        # de baseline hazard is geen geleerde parameter maar een aparte
        # schatting, dus die mag over alle data - anders lever je de
        # val-samples ook daar in
        self.model_.compute_baseline_hazards(input=x_t, target=y_t)

        # EXPERIMENTEEL - alleen de getallen bewaren, niet de logger zelf: die
        # refereert het model en de callbacks en overleeft __getstate__ niet.
        # best_epoch_ is met val_fraction=0.0 het optimum op de trainingsdata
        # en zegt dan weinig; pas met een echte val-split is het bruikbaar om
        # epochs mee te tunen.
        log_df = log.to_pandas()
        self.n_epochs_ran_ = len(log_df)
        self.best_epoch_ = int(log_df['val_loss'].idxmin()) + 1

        return self

    def predict(self, X, times=None):
        check_is_fitted(self)
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        if times is not None:
            new_index = surv.index.union(times)
            surv = surv.reindex(new_index).ffill().loc[times]

        return [StepFunction(surv.index.values, surv.iloc[:, i].values)
                for i in range(surv.shape[1])]

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'model_' in state:
            state['model_state_'] = {
                'n_covariates': self.model_.net.net[0].linear.weight.shape[1],
                'net_state_dict': _cpu_state_dict(self.model_.net),
                'baseline_hazards': self.model_.baseline_hazards_,
                'baseline_cumulative_hazards': self.model_.baseline_cumulative_hazards_,
            }
            del state['model_']
        return state

    def __setstate__(self, state):
        model_state = state.pop('model_state_', None)
        self.__dict__.update(state)

        if model_state is not None:
            net = MLPVanilla(
                model_state['n_covariates'],
                self.num_nodes, 1, self.batch_norm, self.dropout,
                activation=self.activation
            )
            net.load_state_dict(model_state['net_state_dict'])
            net = net.to(_default_device())
            optimizer = tt.optim.Adam(lr=self.learning_rate, weight_decay=self.weight_decay)
            self.model_ = CoxPH(net, optimizer)
            self.model_.baseline_hazards_ = model_state['baseline_hazards']
            self.model_.baseline_cumulative_hazards_ = model_state['baseline_cumulative_hazards']


class DeepSurvWrapperExtended(DeepSurvWrapper, SurvivalPredictorExtendedBase):
    """Extended wrapper with survival-specific helpers."""

    def predict_risk_scores(self, X):
        x_t = _to_tensor(X)
        # numpy=True: pycox mirrors the input type by default, so a tensor input
        # would yield a tensor here instead of the ndarray the base class promises.
        return self.model_.predict(x_t, numpy=True).flatten()

    def predict_survival(self, X, times=None):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        if times is not None:
            new_index = surv.index.union(times)
            surv = surv.reindex(new_index).ffill().loc[times]

        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def predict_median(self, X):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        medians = []
        for i in range(surv.shape[1]):
            surv_col = surv.iloc[:, i]
            below_half = surv_col[surv_col <= 0.5]
            medians.append(below_half.index[0] if len(below_half) > 0 else np.inf)
        return np.array(medians)


# ---------------------------------------------------------------------------
# DeepHit
# ---------------------------------------------------------------------------

class DeepHitWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 num_nodes=[128, 64],
                 dropout=0.34,
                 batch_norm=True,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 batch_size=512,
                 epochs=1500,
                 patience=35,
                 num_durations=100,
                 alpha=0.2,
                 sigma=0.1,
                 scheme="equidistant",  # equidistant | quantiles
                 activation=torch.nn.ReLU,  # torch.nn.ELU | torch.nn.Tanh | torch.nn.SELU
                 verbose=False,
                 seed=None):
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_durations = num_durations
        self.alpha = alpha
        self.sigma = sigma
        self.scheme = scheme
        self.activation = activation
        self.verbose = verbose
        self.seed = seed

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(self, X, y):
        self._set_seed()

        if hasattr(y, 'values'):
            times  = y['time'].values.astype('float32')
            events = y['event'].values.astype('float32')
        elif isinstance(y, np.ndarray) and y.dtype.names:
            times  = y['time'].astype('float32')
            events = y['event'].astype('float32')
        else:
            raise ValueError("y must be a DataFrame or structured numpy array with 'time' and 'event' fields.")

        self.labtrans_ = DeepHitSingle.label_transform(self.num_durations, scheme=self.scheme)
        y_transformed = self.labtrans_.fit_transform(times, events)

        x_np = X.astype('float32') if isinstance(X, np.ndarray) else X.values.astype('float32')

        n_covariates = x_np.shape[1]
        net = MLPVanilla(n_covariates, self.num_nodes, self.labtrans_.out_features,
                         self.batch_norm, self.dropout,
                         activation=self.activation)
        optimizer = tt.optim.Adam(lr=self.learning_rate, weight_decay=self.weight_decay)

        self.model_ = DeepHitSingle(net, optimizer,
                                    alpha=self.alpha,
                                    sigma=self.sigma,
                                    duration_index=self.labtrans_.cuts)

        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        batch_size = _safe_batch_size(len(x_np), self.batch_size)
        self.model_.fit(x_np, y_transformed, batch_size, self.epochs,
                        callbacks, self.verbose,
                        val_data=(x_np, y_transformed))

        return self

    def predict(self, X, times=None):
        check_is_fitted(self)

        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        if times is not None:
            new_index = surv.index.union(times)
            surv = surv.reindex(new_index).ffill().loc[times]

        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'model_' in state:
            state['model_state_'] = {
                'net_state_dict': _cpu_state_dict(self.model_.net),
                'duration_index': self.model_.duration_index,
            }
            del state['model_']
        return state

    def __setstate__(self, state):
        model_state = state.pop('model_state_', None)
        self.__dict__.update(state)

        if model_state is not None:
            net_sd = model_state['net_state_dict']
            if 'net.0.linear.weight' in net_sd:
                n_in = net_sd['net.0.linear.weight'].shape[1]
            elif '0.weight' in net_sd:
                n_in = net_sd['0.weight'].shape[1]
            else:
                raise KeyError(f"Cannot infer n_in from state dict keys: {list(net_sd.keys())}")

            net = MLPVanilla(n_in, self.num_nodes,
                             len(model_state['duration_index']),
                             self.batch_norm, self.dropout,
                             activation=self.activation)
            net.load_state_dict(model_state['net_state_dict'])
            optimizer = tt.optim.Adam(lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
            self.model_ = DeepHitSingle(net, optimizer,
                                        alpha=self.alpha,
                                        sigma=self.sigma,
                                        duration_index=model_state['duration_index'])


class DeepHitWrapperExtended(DeepHitWrapper, SurvivalPredictorExtendedBase):
    """Extended wrapper met survival-specifieke helpers."""

    def predict_risk_scores(self, X):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)
        times = surv.index.values
        rmst = np.trapezoid(surv.values, x=times, axis=0)
        return -rmst  # hogere RMST = langer leven = lager risico

    def predict_survival(self, X, times=None):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        if times is not None:
            new_index = surv.index.union(times)
            surv = surv.reindex(new_index).ffill().loc[times]

        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def predict_median(self, X):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        medians = []
        for i in range(surv.shape[1]):
            surv_col = surv.iloc[:, i]
            below_half = surv_col[surv_col <= 0.5]
            medians.append(below_half.index[0] if len(below_half) > 0 else np.inf)
        return np.array(medians)


# ---------------------------------------------------------------------------
# PCHazard
# ---------------------------------------------------------------------------

class PCHazardWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 num_nodes=[128, 64],
                 dropout=0.34,
                 batch_norm=True,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 batch_size=512,
                 epochs=1500,
                 patience=35,
                 num_durations=100,
                 scheme='equidistant',  # equidistant | quantiles
                 sub=10,
                 activation=torch.nn.ReLU,  # torch.nn.ELU | torch.nn.Tanh | torch.nn.SELU
                 verbose=False,
                 seed=None):
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_durations = num_durations
        self.scheme = scheme
        self.sub = sub
        self.activation = activation
        self.verbose = verbose
        self.seed = seed

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(self, X, y):
        self._set_seed()

        if hasattr(y, 'values'):
            times  = y['time'].values.astype('float32')
            events = y['event'].values.astype('float32')
        elif isinstance(y, np.ndarray) and y.dtype.names:
            times  = y['time'].astype('float32')
            events = y['event'].astype('float32')
        else:
            raise ValueError("y must be a DataFrame or structured numpy array with 'time' and 'event' fields.")

        self.labtrans_ = PCHazard.label_transform(self.num_durations, scheme=self.scheme)
        y_transformed = self.labtrans_.fit_transform(times, events)

        x_np = X.astype('float32') if isinstance(X, np.ndarray) else X.values.astype('float32')

        n_covariates = x_np.shape[1]
        net = MLPVanilla(n_covariates, self.num_nodes, self.labtrans_.out_features,
                         self.batch_norm, self.dropout,
                         activation=self.activation)
        optimizer = tt.optim.Adam(lr=self.learning_rate, weight_decay=self.weight_decay)

        self.model_ = PCHazard(net, optimizer,
                               duration_index=self.labtrans_.cuts,
                               sub=self.sub)

        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        batch_size = _safe_batch_size(len(x_np), self.batch_size)
        self.model_.fit(x_np, y_transformed, batch_size, self.epochs,
                        callbacks, self.verbose,
                        val_data=(x_np, y_transformed))

        return self

    def predict(self, X, times=None):
        check_is_fitted(self)

        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        if times is not None:
            new_index = surv.index.union(times)
            surv = surv.reindex(new_index).ffill().loc[times]

        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'model_' in state:
            state['model_state_'] = {
                'net_state_dict': _cpu_state_dict(self.model_.net),
                'duration_index': self.model_.duration_index,
            }
            del state['model_']
        return state

    def __setstate__(self, state):
        model_state = state.pop('model_state_', None)
        self.__dict__.update(state)

        if model_state is not None:
            net_state = model_state['net_state_dict']
            weight_keys = [k for k in net_state if k.endswith('.weight')]
            n_in = net_state[weight_keys[0]].shape[1]
            # PCHazard telt intervallen, niet cut points: labtrans.out_features is
            # len(cuts) - 1, terwijl duration_index alle cuts bevat. De output-laag
            # wordt daarom uit het checkpoint zelf afgeleid in plaats van uit
            # len(duration_index), wat er steevast één te veel opleverde.
            n_out = net_state[weight_keys[-1]].shape[0]

            net = MLPVanilla(n_in, self.num_nodes,
                             n_out,
                             self.batch_norm, self.dropout,
                             activation=self.activation)
            net.load_state_dict(net_state)
            optimizer = tt.optim.Adam(lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
            self.model_ = PCHazard(net, optimizer,
                                   duration_index=model_state['duration_index'],
                                   sub=self.sub)


class PCHazardWrapperExtended(PCHazardWrapper, SurvivalPredictorExtendedBase):
    """Extended wrapper met survival-specifieke helpers."""

    def predict_risk_scores(self, X):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)
        times = surv.index.values
        rmst = np.trapezoid(surv.values, x=times, axis=0)
        return -rmst  # negatief: hogere RMST = langer leven = lager risico

    def predict_survival(self, X, times=None):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        if times is not None:
            new_index = surv.index.union(times)
            surv = surv.reindex(new_index).ffill().loc[times]

        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def predict_median(self, X):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        medians = []
        for i in range(surv.shape[1]):
            surv_col = surv.iloc[:, i]
            below_half = surv_col[surv_col <= 0.5]
            medians.append(below_half.index[0] if len(below_half) > 0 else np.inf)
        return np.array(medians)


# ---------------------------------------------------------------------------
# LogisticHazard
# ---------------------------------------------------------------------------

class LogisticHazardWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 num_nodes=[128, 64],
                 dropout=0.34,
                 batch_norm=True,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 batch_size=512,
                 epochs=1500,
                 patience=35,
                 num_durations=100,
                 scheme='equidistant',  # equidistant | quantiles
                 activation=torch.nn.ReLU,  # torch.nn.ELU | torch.nn.Tanh | torch.nn.SELU
                 verbose=False,
                 seed=None):
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_durations = num_durations
        self.scheme = scheme
        self.activation = activation
        self.verbose = verbose
        self.seed = seed

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(self, X, y):
        self._set_seed()

        if hasattr(y, 'values'):
            times  = y['time'].values.astype('float32')
            events = y['event'].values.astype('float32')
        elif isinstance(y, np.ndarray) and y.dtype.names:
            times  = y['time'].astype('float32')
            events = y['event'].astype('float32')
        else:
            raise ValueError("y must be a DataFrame or structured numpy array with 'time' and 'event' fields.")

        self.labtrans_ = LogisticHazard.label_transform(self.num_durations, scheme=self.scheme)
        y_transformed = self.labtrans_.fit_transform(times, events)

        x_np = X.astype('float32') if isinstance(X, np.ndarray) else X.values.astype('float32')

        n_covariates = x_np.shape[1]
        net = MLPVanilla(n_covariates, 
                         self.num_nodes, 
                         self.labtrans_.out_features,
                         batch_norm=self.batch_norm,
                         dropout= self.dropout,
                         activation=self.activation)
        optimizer = tt.optim.Adam(lr=self.learning_rate, 
                                  weight_decay=self.weight_decay)

        self.model_ = LogisticHazard(net, optimizer,
                                     duration_index=self.labtrans_.cuts)

        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        batch_size = _safe_batch_size(len(x_np), self.batch_size)
        self.model_.fit(x_np, y_transformed, batch_size, self.epochs,
                        callbacks, self.verbose,
                        val_data=(x_np, y_transformed)
                        )

        return self

    def predict(self, X, times=None):
        check_is_fitted(self)

        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        if times is not None:
            new_index = surv.index.union(times)
            surv = surv.reindex(new_index).ffill().loc[times]

        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'model_' in state:
            state['model_state_'] = {
                'net_state_dict': _cpu_state_dict(self.model_.net),
                'duration_index': self.model_.duration_index,
            }
            del state['model_']
        return state

    def __setstate__(self, state):
        model_state = state.pop('model_state_', None)
        self.__dict__.update(state)

        if model_state is not None:
            net_state = model_state['net_state_dict']
            first_weight_key = next(k for k in net_state if k.endswith('.weight'))
            n_in = net_state[first_weight_key].shape[1]

            net = MLPVanilla(n_in, self.num_nodes,
                             len(model_state['duration_index']),
                             self.batch_norm, self.dropout,
                             activation=self.activation)
            net.load_state_dict(net_state)
            optimizer = tt.optim.Adam(lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
            self.model_ = LogisticHazard(net, optimizer,
                                         duration_index=model_state['duration_index'])


class LogisticHazardWrapperExtended(LogisticHazardWrapper, SurvivalPredictorExtendedBase):
    """Extended wrapper met survival-specifieke helpers."""

    def predict_risk_scores(self, X):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)
        times = surv.index.values
        rmst = np.trapezoid(surv.values, x=times, axis=0)
        return -rmst  # negatief: hogere RMST = langer leven = lager risico

    def predict_survival(self, X, times=None):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        if times is not None:
            new_index = surv.index.union(times)
            surv = surv.reindex(new_index).ffill().loc[times]

        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def predict_median(self, X):
        x_t = _to_tensor(X)
        surv = self.model_.predict_surv_df(x_t)

        medians = []
        for i in range(surv.shape[1]):
            surv_col = surv.iloc[:, i]
            below_half = surv_col[surv_col <= 0.5]
            medians.append(below_half.index[0] if len(below_half) > 0 else np.inf)
        return np.array(medians)


# ---------------------------------------------------------------------------
# LuckSurvival (Luck et al. 2017)
# ---------------------------------------------------------------------------

"""
Implementatie van de Luck et al. (2017) survival architectuur.
Paper: https://arxiv.org/pdf/1705.10245

Architectuur:
    Input → [gedeeld MLP] → 1 bottleneck node (lineair) → T output nodes (survival per tijdstip)

Twee losses tegelijk:
    L1: Cox partial log-likelihood op de bottleneck node  → leert rangorde (wie gaat eerder dood?)
    L2: Binary cross-entropy op de T output nodes         → leert absolute survival kansen per tijdstip
"""

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LuckNet(nn.Module):
    """
    Gedeeld MLP → 1 bottleneck node → T output nodes.

    bottleneck_value:  s¹ in het paper — lineaire activatie, risicomaat
    output:            T waarden — survival kans per tijdstip
    """

    def __init__(self, n_in, num_nodes, num_durations, batch_norm, dropout):
        super().__init__()

        layers = []
        in_features = n_in
        for out_features in num_nodes:
            layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        self.shared = nn.Sequential(*layers)
        self.bottleneck = nn.Linear(in_features, 1)
        self.time_layer = nn.Linear(1, num_durations)

    def forward(self, x):
        shared_out = self.shared(x)
        s1 = self.bottleneck(shared_out)
        survival_logits = self.time_layer(s1)
        survival = torch.sigmoid(survival_logits)
        return s1.squeeze(1), survival


def cox_partial_log_likelihood(risk_scores, times, events):
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    times = times[order]
    events = events[order]
    log_cumsum_exp = torch.logcumsumexp(risk_scores, dim=0)
    loss = -torch.mean((risk_scores - log_cumsum_exp) * events)
    return loss


def survival_bce_loss(survival_pred, times, events, time_cuts):
    batch_size = survival_pred.shape[0]
    T = len(time_cuts)
    time_cuts_t = torch.tensor(time_cuts, dtype=torch.float32, device=survival_pred.device)
    times_expanded = times.unsqueeze(1).expand(batch_size, T)
    events_expanded = events.unsqueeze(1).expand(batch_size, T)
    cuts_expanded = time_cuts_t.unsqueeze(0).expand(batch_size, T)
    targets = (times_expanded > cuts_expanded).float()
    mask = (times_expanded > cuts_expanded) | (events_expanded == 1)
    bce = nn.functional.binary_cross_entropy(survival_pred, targets, reduction="none")
    loss = (bce * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return loss


class LuckSurvivalWrapperExtended(BaseEstimator, RegressorMixin, SurvivalPredictorExtendedBase):
    def __init__(self,
                 num_nodes=[128, 64],
                 dropout=0.34,
                 batch_norm=True,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 batch_size=256,
                 epochs=1500,
                 patience=35,
                 num_durations=50,
                 alpha=0.5,
                 scheme="quantiles",
                 verbose=False,
                 seed=None):
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_durations = num_durations
        self.alpha = alpha
        self.scheme = scheme
        self.verbose = verbose
        self.seed = seed

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _make_time_cuts(self, times):
        if self.scheme == "quantiles":
            quantiles = np.linspace(0, 1, self.num_durations + 1)[1:]
            return np.quantile(times, quantiles).astype("float32")
        else:
            return np.linspace(times.min(), times.max(), self.num_durations).astype("float32")

    def fit(self, X, y):
        self._set_seed()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(y, "values"):
            times = y["time"].values.astype("float32")
            events = y["event"].values.astype("float32")
        elif isinstance(y, np.ndarray) and y.dtype.names:
            times = y["time"].astype("float32")
            events = y["event"].astype("float32")
        else:
            raise ValueError("y moet een DataFrame of structured numpy array zijn met 'time' en 'event'.")

        self.time_cuts_ = self._make_time_cuts(times)

        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")
        X_t = torch.tensor(x_np, device=device)
        t_t = torch.tensor(times, device=device)
        e_t = torch.tensor(events, device=device)

        n_covariates = x_np.shape[1]
        self.net_ = LuckNet(n_covariates, self.num_nodes, self.num_durations,
                            self.batch_norm, self.dropout).to(device)

        optimizer = optim.Adam(self.net_.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        dataset = TensorDataset(X_t, t_t, e_t)
        # Alleen een kleine staart-batch weggooien: te weinig samples geeft instabiele
        # BatchNorm-statistiek en een lege Cox risk set (die wordt binnen de batch opgebouwd).
        # De ondergrens van 2 is hard nodig: BatchNorm1d crasht op een batch van 1 sample.
        # De len-check voorkomt dat de loader leeg raakt bij datasets kleiner dan een batch.
        min_tail = max(2, int(0.15 * self.batch_size))
        drop_last = (len(dataset) > self.batch_size
                   and len(dataset) % self.batch_size < min_tail)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.net_.train()
            epoch_loss = 0.0

            for xb, tb, eb in loader:
                optimizer.zero_grad()
                s1, surv = self.net_(xb)
                l1 = cox_partial_log_likelihood(s1, tb, eb)
                l2 = survival_bce_loss(surv, tb, eb, self.time_cuts_)
                loss = self.alpha * l1 + (1 - self.alpha) * l2
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs}  loss={avg_loss:.4f}")

            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping op epoch {epoch+1}")
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

        self.device_ = device
        return self

    def _predict_surv_df(self, X):
        import pandas as pd
        check_is_fitted(self)
        self.net_.eval()
        with torch.no_grad():
            x_t = _to_tensor(X).to(self.device_)
            _, surv = self.net_(x_t)
            surv_np = surv.cpu().numpy()
        surv_np = np.minimum.accumulate(surv_np, axis=1)
        return pd.DataFrame(surv_np.T, index=self.time_cuts_)

    def predict_risk_scores(self, X):
        check_is_fitted(self)
        self.net_.eval()
        with torch.no_grad():
            x_t = _to_tensor(X).to(self.device_)
            s1, _ = self.net_(x_t)
        return s1.cpu().numpy()

    def predict(self, X, times=None):
        surv = self._predict_surv_df(X)
        if times is not None:
            surv = _reindex_survival(surv, times)
        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def predict_survival(self, X, times=None):
        return self.predict(X, times=times)

    def predict_median(self, X):
        surv = self._predict_surv_df(X)
        medians = []
        for i in range(surv.shape[1]):
            col = surv.iloc[:, i]
            below = col[col <= 0.5]
            medians.append(below.index[0] if len(below) > 0 else np.inf)
        return np.array(medians)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "net_" in state:
            state["net_state_"] = {
                "state_dict": _cpu_state_dict(self.net_),
                "n_in": next(self.net_.shared.parameters()).shape[1],
            }
            del state["net_"]
        return state

    def __setstate__(self, state):
        net_state = state.pop("net_state_", None)
        self.__dict__.update(state)
        if net_state is not None:
            net = LuckNet(
                n_in=net_state["n_in"],
                num_nodes=self.num_nodes,
                num_durations=self.num_durations,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
            )
            # de gewichten staan altijd op CPU in de export; hier weer naar GPU
            # als die er is, anders draait alles gewoon op CPU verder
            device = _default_device()
            net.load_state_dict(net_state["state_dict"])
            self.net_ = net.to(device)
            self.device_ = device


# ---------------------------------------------------------------------------
# DeepCoxIBrier
# ---------------------------------------------------------------------------

"""
DeepCoxIBrier — Deep Cox netwerk met Integrated Brier kalibratie

Architectuur:
    Input → [gedeeld MLP] → s¹ (bottleneck, 1 node, lineair)
                                 ↓
              S(t_k|x) = sigmoid(-(s¹ · w_k + b_k))   ← geleerde tijdsparameters
                                 ↓
              L_cox (rangorde) + L_ibrier (kalibratie)
"""


class DeepCoxIBrierNet(nn.Module):
    def __init__(self, n_in, num_nodes, num_durations, batch_norm, dropout):
        super().__init__()

        layers = []
        in_features = n_in
        for out_features in num_nodes:
            layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        self.shared = nn.Sequential(*layers)
        self.bottleneck = nn.Linear(in_features, 1)
        self.w_raw = nn.Parameter(torch.zeros(num_durations))
        self.b = nn.Parameter(torch.zeros(num_durations))

    def forward(self, x):
        shared_out = self.shared(x)
        s1 = self.bottleneck(shared_out).squeeze(1)
        w = nn.functional.softplus(self.w_raw)
        logits = -(s1.unsqueeze(1) * w.unsqueeze(0) + self.b.unsqueeze(0))
        survival = torch.sigmoid(logits)
        survival = torch.cummin(survival, dim=1).values
        return s1, survival


def integrated_brier_score_loss(survival_pred, times, events, time_cuts):
    device = survival_pred.device
    batch_size = survival_pred.shape[0]
    T = len(time_cuts)

    time_cuts_t = torch.tensor(time_cuts, dtype=torch.float32, device=device)
    times_np = times.detach().cpu().numpy()
    events_np = events.detach().cpu().numpy()

    order = np.argsort(times_np)
    sorted_times = times_np[order]
    sorted_censored = 1 - events_np[order]

    n = len(sorted_times)
    G = np.ones(T, dtype=np.float32)
    g_current = 1.0
    j = 0
    for k, t_k in enumerate(time_cuts):
        while j < n and sorted_times[j] <= t_k:
            if sorted_censored[j] == 1:
                g_current *= (1 - 1.0 / max(n - j, 1))
            j += 1
        G[k] = max(g_current, 1e-4)

    G_t = torch.tensor(G, dtype=torch.float32, device=device)

    times_exp = times.unsqueeze(1).expand(batch_size, T)
    events_exp = events.unsqueeze(1).expand(batch_size, T)
    cuts_exp = time_cuts_t.unsqueeze(0).expand(batch_size, T)
    G_exp = G_t.unsqueeze(0).expand(batch_size, T)

    target = (times_exp > cuts_exp).float()

    G_i = torch.tensor(
        np.interp(times_np, time_cuts, G),
        dtype=torch.float32, device=device
    ).unsqueeze(1).expand(batch_size, T)

    alive_weight = target / G_exp.clamp(min=1e-4)
    dead_weight = (1 - target) * events_exp / G_i.clamp(min=1e-4)
    ipcw_weight = alive_weight + dead_weight

    squared_error = (survival_pred - target) ** 2
    brier_per_t = (squared_error * ipcw_weight).mean(dim=0)

    dt = torch.diff(time_cuts_t, prepend=time_cuts_t[:1])
    ibrier = (brier_per_t * dt).sum() / (time_cuts_t[-1] - time_cuts_t[0]).clamp(min=1e-4)

    return ibrier


class DeepCoxIBrierWrapperExtended(BaseEstimator, RegressorMixin, SurvivalPredictorExtendedBase):
    def __init__(self,
                 num_nodes=[128, 64],
                 dropout=0.34,
                 batch_norm=True,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 batch_size=256,
                 epochs=1500,
                 patience=35,
                 val_fraction=0.0,  # EXPERIMENTEEL, zie fit(); 0.0 = oude gedrag
                 num_durations=50,
                 alpha=0.4,
                 scheme="quantiles",
                 verbose=False,
                 seed=None):
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.num_durations = num_durations
        self.alpha = alpha
        self.scheme = scheme
        self.verbose = verbose
        self.seed = seed

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _make_time_cuts(self, times):
        if self.scheme == "quantiles":
            quantiles = np.linspace(0, 1, self.num_durations + 1)[1:]
            return np.quantile(times, quantiles).astype("float32")
        else:
            return np.linspace(times.min(), times.max(), self.num_durations).astype("float32")

    def fit(self, X, y):
        self._set_seed()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(y, "values"):
            times = y["time"].values.astype("float32")
            events = y["event"].values.astype("float32")
        elif isinstance(y, np.ndarray) and y.dtype.names:
            times = y["time"].astype("float32")
            events = y["event"].astype("float32")
        else:
            raise ValueError("y moet een DataFrame of structured numpy array zijn met 'time' en 'event'.")

        self.time_cuts_ = self._make_time_cuts(times)

        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")
        X_t = torch.tensor(x_np, device=device)
        t_t = torch.tensor(times, device=device)
        e_t = torch.tensor(events, device=device)

        n_covariates = x_np.shape[1]
        self.net_ = DeepCoxIBrierNet(n_covariates, self.num_nodes, self.num_durations,
                                     self.batch_norm, self.dropout).to(device)

        optimizer = optim.Adam(self.net_.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        # EXPERIMENTEEL - val_fraction > 0 houdt een deel van de trainingsdata
        # apart voor early stopping. Met val_fraction=0.0 wordt de trainloss
        # gemonitord, die vrijwel altijd blijft dalen, waardoor de patience
        # nooit afgaat en er in de praktijk exact self.epochs epochs gedraaid
        # worden. Dat is het gedrag waarop alle bestaande CV-resultaten zijn
        # gebaseerd, vandaar dat 0.0 de default blijft. De split is
        # gestratificeerd op event, anders kan de val-set bijna alleen
        # censoring bevatten en is de val-loss een slecht stopsignaal.
        if self.val_fraction:
            idx_tr, idx_val = train_test_split(
                np.arange(len(times)), test_size=self.val_fraction,
                stratify=events, random_state=self.seed)
            dataset  = TensorDataset(X_t[idx_tr], t_t[idx_tr], e_t[idx_tr])
            val_data = (X_t[idx_val], t_t[idx_val], e_t[idx_val])
        else:
            dataset  = TensorDataset(X_t, t_t, e_t)
            val_data = None

        # Alleen een kleine staart-batch weggooien: te weinig samples geeft instabiele
        # BatchNorm-statistiek en een lege Cox risk set (die wordt binnen de batch opgebouwd).
        # De ondergrens van 2 is hard nodig: BatchNorm1d crasht op een batch van 1 sample.
        # De len-check voorkomt dat de loader leeg raakt bij datasets kleiner dan een batch.
        min_tail = max(2, int(0.15 * self.batch_size))
        drop_last = (len(dataset) > self.batch_size
                   and len(dataset) % self.batch_size < min_tail)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.net_.train()
            epoch_loss = 0.0

            for xb, tb, eb in loader:
                optimizer.zero_grad()
                s1, surv = self.net_(xb)
                l_cox = cox_partial_log_likelihood(s1, tb, eb)
                l_ibrier = integrated_brier_score_loss(surv, tb, eb, self.time_cuts_)
                loss = self.alpha * l_cox + (1 - self.alpha) * l_ibrier
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            # Zonder validatiesplit is de trainloss het stopsignaal - identiek
            # aan het gedrag van voor val_fraction. De val-loss gebruikt
            # dezelfde gewogen som van beide termen, maar over de hele val-set
            # in een keer, zodat de IPCW-gewichten niet per batch geschat
            # worden.
            if val_data is not None:
                self.net_.eval()
                with torch.no_grad():
                    xv, tv, ev = val_data
                    s1_v, surv_v = self.net_(xv)
                    monitor_loss = (
                        self.alpha * cox_partial_log_likelihood(s1_v, tv, ev)
                        + (1 - self.alpha) * integrated_brier_score_loss(
                            surv_v, tv, ev, self.time_cuts_)
                    ).item()
            else:
                monitor_loss = avg_loss

            if self.verbose and (epoch + 1) % 50 == 0:
                val_str = "" if val_data is None else f"  val={monitor_loss:.4f}"
                print(f"Epoch {epoch+1:4d}/{self.epochs}  "
                      f"loss={avg_loss:.4f}  "
                      f"cox={l_cox.item():.4f}  "
                      f"ibrier={l_ibrier.item():.4f}"
                      f"{val_str}")

            if monitor_loss < best_loss - 1e-5:
                best_loss = monitor_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping op epoch {epoch+1}")
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

        self.device_ = device
        return self

    def _predict_surv_df(self, X):
        import pandas as pd
        check_is_fitted(self)
        self.net_.eval()
        with torch.no_grad():
            x_t = _to_tensor(X).to(self.device_)
            _, surv = self.net_(x_t)
            surv_np = surv.cpu().numpy()
        return pd.DataFrame(surv_np.T, index=self.time_cuts_)

    def predict_risk_scores(self, X):
        check_is_fitted(self)
        self.net_.eval()
        with torch.no_grad():
            x_t = _to_tensor(X).to(self.device_)
            s1, _ = self.net_(x_t)
        return s1.cpu().numpy()

    def predict(self, X, times=None):
        surv = self._predict_surv_df(X)
        if times is not None:
            surv = _reindex_survival(surv, times)
        return [
            StepFunction(surv.index.values, surv.iloc[:, i].values)
            for i in range(surv.shape[1])
        ]

    def predict_survival(self, X, times=None):
        return self.predict(X, times=times)

    def predict_median(self, X):
        surv = self._predict_surv_df(X)
        medians = []
        for i in range(surv.shape[1]):
            col = surv.iloc[:, i]
            below = col[col <= 0.5]
            medians.append(below.index[0] if len(below) > 0 else np.inf)
        return np.array(medians)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "net_" in state:
            state["net_state_"] = {
                "state_dict": _cpu_state_dict(self.net_),
                "n_in": next(self.net_.shared.parameters()).shape[1],
            }
            del state["net_"]
        return state

    def __setstate__(self, state):
        net_state = state.pop("net_state_", None)
        self.__dict__.update(state)
        if net_state is not None:
            net = DeepCoxIBrierNet(
                n_in=net_state["n_in"],
                num_nodes=self.num_nodes,
                num_durations=self.num_durations,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
            )
            # de gewichten staan altijd op CPU in de export; hier weer naar GPU
            # als die er is, anders draait alles gewoon op CPU verder
            device = _default_device()
            net.load_state_dict(net_state["state_dict"])
            self.net_ = net.to(device)
            self.device_ = device


# ---------------------------------------------------------------------------
# MTLR (Multi-Task Logistic Regression) - Sklearn variant
# ---------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression


class MTLRSklearnWrapper(BaseEstimator, RegressorMixin):
    """
    Multi-Task Logistic Regression met sklearn.

    Discretiseert overlevingstijden in bins en traint een LogisticRegression
    per bin om P(survive to t_k | X) te voorspellen.

    Voordelen:
    - Zeer snel
    - Goed gekalibreerd (directe kansvoorspelingen)
    - Geen GPU nodig
    - Interpreteerbaar (lineaire gewichten per feature per bin)
    """

    def __init__(self, num_durations=50, scheme="quantiles", C=1.0, penalty="l2",
                 solver="lbfgs", max_iter=1000, verbose=False):
        self.num_durations = num_durations
        self.scheme = scheme
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose

    def _make_time_cuts(self, times):
        if self.scheme == "quantiles":
            quantiles = np.linspace(0, 1, self.num_durations + 1)[1:]
            return np.quantile(times, quantiles).astype("float32")
        else:
            return np.linspace(times.min(), times.max(), self.num_durations).astype("float32")

    def fit(self, X, y):
        if hasattr(y, "values"):
            times = y["time"].values.astype("float32")
            events = y["event"].values.astype("float32")
        elif isinstance(y, np.ndarray) and y.dtype.names:
            times = y["time"].astype("float32")
            events = y["event"].astype("float32")
        else:
            raise ValueError("y moet een DataFrame of structured numpy array zijn met 'time' en 'event'.")

        self.time_cuts_ = self._make_time_cuts(times)
        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")

        self.models_ = []
        self.class_probs_ = []

        for k, t_k in enumerate(self.time_cuts_):
            # Wie op t_k nog gevolgd wordt, telt mee; wie vóór t_k gecensureerd
            # is heeft op t_k een onbekende status en valt uit déze bin. Zonder
            # dit masker krijgt een patiënt die op dag 500 uit follow-up gaat
            # vanaf bin 500 hetzelfde label als iemand die daar overleed, wat de
            # voorspelde overleving structureel omlaagtrekt naarmate t groeit.
            at_risk = (times > t_k) | (events == 1)
            target  = (times[at_risk] > t_k).astype(int)
            x_bin   = x_np[at_risk]
            n_classes = len(np.unique(target))

            if len(target) < 2 or n_classes < 2:
                self.models_.append(None)
                # Lege bin (geen enkele bekende status meer) -> geen overleving;
                # de cummin in predict houdt de curve daarna monotoon.
                self.class_probs_.append(float(target.mean()) if len(target) else 0.0)
            else:
                model = LogisticRegression(C=self.C, penalty=self.penalty,
                                          max_iter=self.max_iter, solver=self.solver, random_state=42)
                model.fit(x_bin, target)
                self.models_.append(model)
                self.class_probs_.append(None)

            if self.verbose and (k + 1) % 10 == 0:
                print(f"Trained {k+1}/{len(self.time_cuts_)} bins  (n at risk={len(target)})")

        return self

    def predict(self, X, times=None):
        check_is_fitted(self)
        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")

        surv_probs = []
        for model, class_prob in zip(self.models_, self.class_probs_):
            if model is None:
                probs = np.full(len(x_np), class_prob)
            else:
                probs = model.predict_proba(x_np)[:, 1]
            surv_probs.append(probs)

        surv_probs = np.array(surv_probs).T
        surv_probs = np.minimum.accumulate(surv_probs, axis=1)

        time_index = self.time_cuts_
        if times is not None:
            import pandas as pd
            surv_df = pd.DataFrame(surv_probs.T, index=self.time_cuts_)
            surv_df = _reindex_survival(surv_df, times)
            surv_probs = surv_df.T.values
            time_index = times

        return [
            StepFunction(time_index, surv_probs[i, :])
            for i in range(surv_probs.shape[0])
        ]


class MTLRSklearnWrapperExtended(MTLRSklearnWrapper, SurvivalPredictorExtendedBase):
    """Extended MTLR Sklearn wrapper met survival-specifieke helpers."""

    def predict_risk_scores(self, X):
        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")

        surv_probs = []
        for model, class_prob in zip(self.models_, self.class_probs_):
            if model is None:
                probs = np.full(len(x_np), class_prob)
            else:
                probs = model.predict_proba(x_np)[:, 1]
            surv_probs.append(probs)

        surv_probs = np.array(surv_probs).T
        surv_probs = np.minimum.accumulate(surv_probs, axis=1)

        times = self.time_cuts_
        rmst = np.trapezoid(surv_probs, x=times, axis=1)
        return -rmst

    def predict_survival(self, X, times=None):
        return self.predict(X, times=times)

    def predict_median(self, X):
        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")

        surv_probs = []
        for model, class_prob in zip(self.models_, self.class_probs_):
            if model is None:
                probs = np.full(len(x_np), class_prob)
            else:
                probs = model.predict_proba(x_np)[:, 1]
            surv_probs.append(probs)

        surv_probs = np.array(surv_probs).T
        surv_probs = np.minimum.accumulate(surv_probs, axis=1)

        medians = []
        for i in range(surv_probs.shape[0]):
            surv_col = surv_probs[i, :]
            below_half = np.where(surv_col <= 0.5)[0]
            if len(below_half) > 0:
                medians.append(self.time_cuts_[below_half[0]])
            else:
                medians.append(np.inf)

        return np.array(medians)


# ---------------------------------------------------------------------------
# MTLR (Multi-Task Logistic Regression) - Deep variant
# ---------------------------------------------------------------------------

class MTLRDeepNet(nn.Module):
    """
    Diep neuraal netwerk voor MTLR.

    Gedeeld MLP → T outputs (één sigmoid per time bin).
    """

    def __init__(self, n_in, num_nodes, num_durations, batch_norm, dropout):
        super().__init__()

        layers = []
        in_features = n_in
        for out_features in num_nodes:
            layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        self.shared = nn.Sequential(*layers)
        self.time_layer = nn.Linear(in_features, num_durations)

    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.time_layer(shared_out)
        survival = torch.sigmoid(logits)
        survival = torch.cummin(survival, dim=1)[0]
        return survival


def mtlr_bce_loss(survival_pred, times, events, time_cuts):
    """Binary cross-entropy loss voor MTLR."""
    batch_size = survival_pred.shape[0]
    T = len(time_cuts)

    time_cuts_t = torch.tensor(time_cuts, dtype=torch.float32, device=survival_pred.device)
    times_expanded = times.unsqueeze(1).expand(batch_size, T)
    events_expanded = events.unsqueeze(1).expand(batch_size, T)
    cuts_expanded = time_cuts_t.unsqueeze(0).expand(batch_size, T)

    targets = (times_expanded > cuts_expanded).float()

    # Zelfde maskering als in de sklearn-variant: een gecensureerde patiënt
    # levert alleen bins op tot aan zijn laatste controle. Daarna is onbekend
    # of hij nog leeft, dus die cellen tellen niet mee in de loss in plaats van
    # als sterfte te worden gescoord.
    observed = (times_expanded > cuts_expanded) | (events_expanded == 1)
    observed = observed.float()

    bce = nn.functional.binary_cross_entropy(survival_pred, targets, reduction="none")
    loss = (bce * observed).sum() / observed.sum().clamp(min=1.0)

    return loss


class MTLRDeepWrapper(BaseEstimator, RegressorMixin):
    """
    Multi-Task Logistic Regression met diep netwerk.

    Voordelen:
    - Flexibeler dan sklearn variant (non-lineaire transformaties)
    - Goed gekalibreerd (directe survival kansen)
    - GPU ondersteund
    """

    def __init__(self,
                 num_nodes=[128, 64],
                 dropout=0.34,
                 batch_norm=True,
                 learning_rate=1e-4,
                 weight_decay=1e-4,
                 batch_size=512,
                 epochs=1500,
                 patience=35,
                 num_durations=50,
                 scheme='equidistant',
                 verbose=False,
                 seed=None):
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_durations = num_durations
        self.scheme = scheme
        self.verbose = verbose
        self.seed = seed

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _make_time_cuts(self, times):
        if self.scheme == "quantiles":
            quantiles = np.linspace(0, 1, self.num_durations + 1)[1:]
            return np.quantile(times, quantiles).astype("float32")
        else:
            return np.linspace(times.min(), times.max(), self.num_durations).astype("float32")

    def fit(self, X, y):
        self._set_seed()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(y, "values"):
            times = y["time"].values.astype("float32")
            events = y["event"].values.astype("float32")
        elif isinstance(y, np.ndarray) and y.dtype.names:
            times = y["time"].astype("float32")
            events = y["event"].astype("float32")
        else:
            raise ValueError("y moet een DataFrame of structured numpy array zijn met 'time' en 'event'.")

        self.time_cuts_ = self._make_time_cuts(times)

        x_np = X.astype("float32") if isinstance(X, np.ndarray) else X.values.astype("float32")
        X_t = torch.tensor(x_np, device=device)
        t_t = torch.tensor(times, device=device)
        e_t = torch.tensor(events, device=device)

        n_covariates = x_np.shape[1]
        self.net_ = MTLRDeepNet(n_covariates, self.num_nodes, self.num_durations,
                               self.batch_norm, self.dropout).to(device)

        optimizer = optim.Adam(self.net_.parameters(),
                              lr=self.learning_rate,
                              weight_decay=self.weight_decay)

        dataset = TensorDataset(X_t, t_t, e_t)
        # Alleen een kleine staart-batch weggooien: te weinig samples geeft instabiele
        # BatchNorm-statistiek en een lege Cox risk set (die wordt binnen de batch opgebouwd).
        # De ondergrens van 2 is hard nodig: BatchNorm1d crasht op een batch van 1 sample.
        # De len-check voorkomt dat de loader leeg raakt bij datasets kleiner dan een batch.
        min_tail = max(2, int(0.15 * self.batch_size))
        drop_last = (len(dataset) > self.batch_size
                   and len(dataset) % self.batch_size < min_tail)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.net_.train()
            epoch_loss = 0.0

            for xb, tb, eb in loader:
                optimizer.zero_grad()
                surv = self.net_(xb)
                loss = mtlr_bce_loss(surv, tb, eb, self.time_cuts_)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs}  loss={avg_loss:.4f}")

            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping op epoch {epoch+1}")
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

        self.device_ = device
        return self

    def predict(self, X, times=None):
        check_is_fitted(self)
        self.net_.eval()
        with torch.no_grad():
            x_t = _to_tensor(X).to(self.device_)
            surv = self.net_(x_t)
            surv_np = surv.cpu().numpy()

        time_index = self.time_cuts_
        if times is not None:
            import pandas as pd
            surv_df = pd.DataFrame(surv_np.T, index=self.time_cuts_)
            surv_df = _reindex_survival(surv_df, times)
            surv_np = surv_df.T.values
            time_index = times

        return [
            StepFunction(time_index, surv_np[i, :])
            for i in range(surv_np.shape[0])
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        if "net_" in state:
            state["net_state_"] = {
                "state_dict": _cpu_state_dict(self.net_),
                "n_in": next(self.net_.shared.parameters()).shape[1],
            }
            del state["net_"]
        return state

    def __setstate__(self, state):
        net_state = state.pop("net_state_", None)
        self.__dict__.update(state)
        if net_state is not None:
            net = MTLRDeepNet(
                n_in=net_state["n_in"],
                num_nodes=self.num_nodes,
                num_durations=self.num_durations,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
            )
            # de gewichten staan altijd op CPU in de export; hier weer naar GPU
            # als die er is, anders draait alles gewoon op CPU verder
            device = _default_device()
            net.load_state_dict(net_state["state_dict"])
            self.net_ = net.to(device)
            self.device_ = device


class MTLRDeepWrapperExtended(MTLRDeepWrapper, SurvivalPredictorExtendedBase):
    """Extended MTLR Deep wrapper met survival-specifieke helpers."""

    def predict_risk_scores(self, X):
        self.net_.eval()
        with torch.no_grad():
            x_t = _to_tensor(X).to(self.device_)
            surv = self.net_(x_t)
            surv_np = surv.cpu().numpy()

        times = self.time_cuts_
        rmst = np.trapezoid(surv_np, x=times, axis=1)
        return -rmst

    def predict_survival(self, X, times=None):
        return self.predict(X, times=times)

    def predict_median(self, X):
        self.net_.eval()
        with torch.no_grad():
            x_t = _to_tensor(X).to(self.device_)
            surv = self.net_(x_t)
            surv_np = surv.cpu().numpy()

        medians = []
        for i in range(surv_np.shape[0]):
            surv_col = surv_np[i, :]
            below_half = np.where(surv_col <= 0.5)[0]
            if len(below_half) > 0:
                medians.append(self.time_cuts_[below_half[0]])
            else:
                medians.append(np.inf)

        return np.array(medians)


# ---------------------------------------------------------------------------
# SurvivalPipeline
# ---------------------------------------------------------------------------

def _pad_y_for_unlabelled(y, n_extra: int):
    """Extend *y* with n_extra dummy entries, keeping its dtype.

    Only needed to satisfy length checks: the steps this padded y is handed to
    ignore it. A structured survival y keeps its ('event', 'time') dtype, so
    the concatenation stays valid.
    """
    if y is None or n_extra == 0:
        return y
    y = np.asarray(y)
    return np.concatenate([y, np.zeros(n_extra, dtype=y.dtype)])


class SurvivalPipeline(Pipeline):
    """Pipeline wrapper to expose survival-specific predictions."""

    # Steps that do not consume y and may therefore also be fitted on the
    # unlabelled samples; the names are the ones build_pipeline assigns.
    # Anything not listed counts as supervised, so a step added later is never
    # silently handed extra rows -- it keeps the labelled-only behaviour until
    # it is explicitly declared here.
    _UNSUPERVISED_STEPS = frozenset({"var_filter", "scaler", "pca", "scaler_pca"})

    def fit(self, X, y=None, X_unlabelled=None, **fit_params):
        """Fit the pipeline, optionally with extra samples for the PCA.

        *X_unlabelled* holds samples without an outcome -- same features, same
        column order as X. They join the fit of the steps in
        _UNSUPERVISED_STEPS (the PCA above all: more samples give a better
        estimate of the components) and are transformed along with X so they
        reach the later steps in the same space. They never reach a step that
        consumes y, nor the final estimator.

        In cross-validation this stays leakage-free as long as the caller keeps
        the test fold out of X_unlabelled: the supervised steps see the
        training fold only, exactly as before.

        Without X_unlabelled this is the stock Pipeline.fit, including its
        `memory` caching -- which the semi-supervised branch does not use,
        since it fits the steps itself.
        """
        if X_unlabelled is None:
            return super().fit(X, y, **fit_params)

        if fit_params:
            raise ValueError(
                "fit_params are not supported together with X_unlabelled: the "
                "steps are fitted directly here, not through Pipeline.fit."
            )

        X_labelled, X_unlab = X, X_unlabelled

        for name, step in self.steps[:-1]:
            if step is None or step == "passthrough":
                continue

            if name in self._UNSUPERVISED_STEPS:
                # y is padded instead of dropped: var_filter is a SelectKBest,
                # which rejects a y whose length does not match X even though
                # its score function (_variance_score) ignores y entirely.
                step.fit(np.vstack([X_labelled, X_unlab]),
                         _pad_y_for_unlabelled(y, len(X_unlab)))
            else:
                step.fit(X_labelled, y)

            X_labelled = step.transform(X_labelled)
            X_unlab    = step.transform(X_unlab)

        self.steps[-1][1].fit(X_labelled, y)
        return self

    def _transform_X(self, X):
        return self[:-1].transform(X) if len(self.steps) > 1 else X

    def predict_survival(self, X, times=None):
        Xt = self._transform_X(X)
        return self.steps[-1][1].predict_survival(Xt, times=times)

    def predict_median(self, X):
        Xt = self._transform_X(X)
        return self.steps[-1][1].predict_median(Xt)

    def predict_all(self, X, times=None):
        Xt = self._transform_X(X)
        return self.steps[-1][1].predict_all(Xt, times=times)


