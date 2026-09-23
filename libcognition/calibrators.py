#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

ASSETS_PATH = Path(__file__).resolve().parent.parent / "assets"


class SurvivalCalibrator:

    def __init__(self, method="lowess", frac=0.75):
        if method not in ("linear", "lowess"):
            raise ValueError(f"method must be 'linear' or 'lowess', got '{method}'")
        self.method = method
        self.frac   = frac
        self._params = {}

    def fit(self, predicted_times, actual_times, actual_events):
        predicted_times = np.asarray(predicted_times, float)

        # handle string "True"/"False" as well as numeric 0/1
        actual_events = np.asarray(actual_events)
        if actual_events.dtype.kind in ('U', 'S', 'O'):
            events_bool = actual_events == "True"
        else:
            events_bool = actual_events.astype(bool)

        mask   = events_bool & np.isfinite(predicted_times)
        pred   = np.log1p(predicted_times[mask])
        actual = np.log1p(np.asarray(actual_times, float)[mask])

        if self.method == "linear":
            s, i, *_ = stats.linregress(actual, pred)   # pred ~ actual
            self._params = {"slope": s, "intercept": i}

        elif self.method == "lowess":
            sort_idx = np.argsort(actual)
            smoothed = lowess(pred[sort_idx], actual[sort_idx], frac=self.frac)  # pred ~ actual
            self._params = {"x": smoothed[:, 0].tolist(),
                            "y": smoothed[:, 1].tolist()}
            self._build_interp()

        return self

    def transform(self, predicted_times):
        pred = np.log1p(np.asarray(predicted_times, float))

        if self.method == "linear":
            corrected = (pred - self._params["intercept"]) / self._params["slope"]
        elif self.method == "lowess":
            corrected = self._interp(pred)

        result = np.expm1(corrected)
        return np.clip(result, 0, None)

    def _build_interp(self):
        x = np.array(self._params["x"])
        y = np.array(self._params["y"])

        self._interp = interp1d(y, x, kind="linear",
                                bounds_error=False,
                                fill_value="extrapolate")

    def save(self, path: str):
        payload = {"method": self.method, "frac": self.frac, "params": self._params}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(f"{ASSETS_PATH}/models/{path}") as f:
            payload = json.load(f)
        cal = cls(method=payload["method"], frac=payload["frac"])
        cal._params = payload["params"]
        if cal.method == "lowess":
            cal._build_interp()
        return cal

    def __str__(self):
        if not self._params:
            return f"{self.__class__.__name__} — not yet fitted"

        if self.method == "linear":
            return (
                f"{self.__class__.__name__} — linear\n"
                f"  slope     : {self._params['slope']:.4f}\n"
                f"  intercept : {self._params['intercept']:.4f}"
            )

        if self.method == "lowess":
            x = np.array(self._params["x"])
            y = np.array(self._params["y"])
            factors = np.expm1(y) / np.where(np.expm1(x) == 0, np.nan, np.expm1(x))
            median_factor = np.nanmedian(factors)
            p25, p75 = np.nanpercentile(factors, [25, 75])
            return (
                f"{self.__class__.__name__} — lowess (frac={self.frac})\n"
                f"  knots          : {len(x)}\n"
                f"  correction factor (calibrated/raw) in original space:\n"
                f"    median : {median_factor:.4f}\n"
                f"    IQR    : [{p25:.4f}, {p75:.4f}]"
            )

