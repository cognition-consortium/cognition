#!/usr/bin/env python3

import json
import joblib
import re

import numpy as np
import pandas as pd

import copy

from pathlib import Path

from enum import Enum
from colorama import Fore, Back, Style, init
from sksurv.functions import StepFunction

from libcognition.calibrators import *

# This is important for cross-platform compatibility
init(autoreset=True)


from . import ASSETS_PATH, DAYS_PER_YEAR, MAX_FOLLOW_UP_YEARS


def load_model_bundle(path):
    """joblib.load met CPU-fallback voor op GPU getrainde modellen.

    De wrappers exporteren hun gewichten sinds v1.0.4 op CPU, maar oudere
    .joblib-builds (en modellen van derden) kunnen CUDA-tensors bevatten. torch
    weigert die te deserialiseren zodra er geen GPU is: "Attempting to
    deserialize object on a CUDA device but torch.cuda.is_available() is
    False". torch.storage._load_from_bytes roept torch.load aan zonder
    map_location, dus zetten we die default hier tijdelijk op 'cpu' - alleen op
    machines zonder GPU, zodat op de GPU-server niets verandert.
    """
    import torch

    if torch.cuda.is_available():
        return joblib.load(path)

    original_load = torch.load

    def load_on_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original_load(*args, **kwargs)

    torch.load = load_on_cpu
    try:
        return joblib.load(path)
    finally:
        torch.load = original_load


class BaseEnum(Enum):
    """A custom Enum that allows for a 'color' attribute."""
    def __new__(cls, value, color):
        # Create a new instance of the enum member
        obj = object.__new__(cls)
        
        # Set the 'value' and 'color' attributes
        obj._value_ = value
        obj.color = color
        
        return obj
    
    def __str__(self):
        # Use the stored color attribute to format the string
        return f"{self.color}{self.value}{Style.RESET_ALL}"


class EndPoint(BaseEnum):
    """Enum for a status to ensure type safety."""
    OverallSurvival = "overall survival", Fore.CYAN
    ProgressionFreeSurvival = "progression free survival", Fore.CYAN
    Sex = "sex", Fore.GREEN
    TumorSubtype = "tumor subtype", Fore.YELLOW
    CopyNumber = "CNV", Fore.BLUE,
    TumorPurity = "tumor purity", Fore.MAGENTA

    def __lt__(self, other):
        if not isinstance(other, EndPoint):
            return NotImplemented
        return self.value[0] < other.value[0]

    def __le__(self, other):
        if not isinstance(other, EndPoint):
            return NotImplemented
        return self.value[0] <= other.value[0]

    @classmethod
    def from_string(cls, value):
        """Find enum member by string value."""
        alternative_mappings = {'survival': 'overall survival'}
        value = alternative_mappings.get(value, value)
        
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No EndPoint with value '{value}'")


class PredictorType(BaseEnum):
    """Enum for predictor types."""
    # deterministic
    CoxnetSurvivalAnalysis = 'CoxnetSurvivalAnalysis', Fore.GREEN
    CoxPHFitter = 'CoxPHFitter', Fore.GREEN
    LogLogisticAFTFitter = 'LogLogisticAFTFitter', Fore.GREEN
    LogNormalAFTFitter = 'LogNormalAFTFitter', Fore.GREEN
    WeibullPHFitter = 'WeibullAFTFitter', Fore.GREEN
    MTLRSklearnWrapper = 'MTLRSklearn', Fore.GREEN

    LinearDiscriminantAnalysis = 'LinearDiscriminantAnalysis', Fore.GREEN
    LogisticRegression = 'LogisticRegression', Fore.GREEN

    # non deterministic
    DeepSurv = 'DeepSurv', Fore.CYAN
    DeepHit = 'DeepHit', Fore.CYAN
    LogisticHazardWrapper = "LogisticHazardWrapper", Fore.CYAN
    DeepCoxIBrierWrapper = "DeepCoxIBrier", Fore.CYAN
    LuckSurvivalWrapper = "LuckSurvival", Fore.CYAN
    MTLRDeepWrapper = 'MTLRDeep', Fore.CYAN
    PCHazard = 'PCHazard', Fore.CYAN
    RandomForest = 'RandomForest', Fore.CYAN
    RandomSurvivalForest = 'RandomSurvivalForest', Fore.CYAN
    GradientBoostingSurvivalAnalysis = 'GradientBoostingSurvivalAnalysis', Fore.CYAN
    ComponentwiseGradientBoostingSurvivalAnalysis = 'ComponentwiseGradientBoostingSurvivalAnalysis', Fore.CYAN
    XGBoostAFT = 'XGBoostAFT', Fore.CYAN

    @classmethod
    def from_string(cls, value):
        """Find enum member by string value."""
        alternative_mappings = {'LDA': 'LinearDiscriminantAnalysis'}
        value = alternative_mappings.get(value, value)

        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No PredictorType with value '{value}'")


class Predictor:
    """
    A class to handle predictor logic based on a JSON configuration file.
    """
    def __init__(self, json_file: str):
        self.json_file = json_file

        try:
            # Use a 'with' statement for safe file handling
            with open(json_file, 'r') as json_data:
                # Use json.load() for file objects
                data = json.load(json_data)
                
                # Parse the model string
                model_parts = data['model'].split("__", 3)
                if len(model_parts) < 3:
                    raise ValueError("Invalid model format")
                
                # Convert strings to enums safely
                data['predictor type'] = PredictorType.from_string(data['predictiontype'])
                data['endpoint'] = EndPoint.from_string(data['endpoint']  )

                if 'grade_only_results' in data:
                    data['grade_only_results'] = pd.DataFrame(data['grade_only_results'])

                self.data = data


        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file '{json_file}' not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required key in JSON: {e}")
        
        # The bundle is loaded on first use, not here. The .joblib files run up
        # to 1.2 GB apiece, and `cognition list` reads nothing but this JSON:
        # eagerly loading them pulled the entire model directory (9.4 GB for
        # v1.0.3) off the filesystem just to print a table.
        self._compiled = None
        #self._compiled = torch.load(path + "/" + self.data['model'].replace('.joblib','.pt')) # still has issues

    @property
    def compiled(self) -> dict:
        """The joblib bundle (pipeline, label encoder, ...), loaded on demand."""
        if self._compiled is None:
            path = Path(f"{ASSETS_PATH}/models/{self.data['model']}")
            # Announced because this is where the wait is: a 1.2 GB bundle off a
            # network filesystem takes long enough that silence looks like a hang.
            size = path.stat().st_size / 1024 ** 3 if path.exists() else 0.0
            print(f"Loading model bundle ({size:.1f} GB): {self.data['model']}", flush=True)
            self._compiled = load_model_bundle(str(path))
        return self._compiled


    def predict_discrete(self, unseen_data):
        self.out = self.compiled['pipeline'].predict_proba(unseen_data)[0]

    def print_outcome_discrete(self):
        class_names = self.compiled['label_encoder'].classes_
        class_and_prob = list(zip(class_names, self.out))
        class_and_prob.sort(key=lambda x: x[1], reverse=True)

        for class_name, prob in class_and_prob:
            #print(f"{class_name[:32]:<32} | Probability: {prob:.4f}")
            print(f"{(class_name[:30] + '..') if len(class_name) > 32 else class_name:<32} | Probability: {prob:.4f}")  


    def predict_right_censored(self, unseen_data):
        self.out = self.compiled['pipeline'].predict_survival(unseen_data)[0]

        if 'calibrations' in self.data:
            cal = SurvivalCalibrator.load(self.data['calibrations']["expected_survival"]["linear"]["path"])
            calibrated_x = cal.transform(self.out.x)
            self.out_cal = StepFunction(x=calibrated_x, y=self.out.y)# @todo move this into transform?


    def print_outcome_right_censored(self):
        # plot outcome to stdout
        times_df = pd.DataFrame({
            "days": [
                DAYS_PER_YEAR * (1/4),
                DAYS_PER_YEAR * (2/4),
                DAYS_PER_YEAR * (3/4),
                DAYS_PER_YEAR,
                DAYS_PER_YEAR * 1.5,
                DAYS_PER_YEAR * 2,
                DAYS_PER_YEAR * 3,
                DAYS_PER_YEAR * 4,
                DAYS_PER_YEAR * 5,
                DAYS_PER_YEAR * 7.5,
                DAYS_PER_YEAR * 10,
                DAYS_PER_YEAR * 12.5,
                DAYS_PER_YEAR * 15.0,
                DAYS_PER_YEAR * 20.0,
                DAYS_PER_YEAR * 25.0,
                DAYS_PER_YEAR * MAX_FOLLOW_UP_YEARS
                ]
            })
        times_df['label'] = ""
        
        median_time_idx = np.argmax(self.out.y <= 0.5)
        if median_time_idx > 0 and self.out.y[median_time_idx] <= 0.5:
            new_row = pd.DataFrame([{"days": self.out.x[median_time_idx], 'label': "*median"}])
            times_df = pd.concat([times_df, new_row], ignore_index=True)
            times_df = times_df.sort_values("days").reset_index(drop=True)


        times_df["time_normalised"] = np.where(
            times_df["days"] < DAYS_PER_YEAR,
            (times_df["days"] / (DAYS_PER_YEAR/12)).round(0),
            (times_df["days"] / DAYS_PER_YEAR).round(1)
        )
        times_df["time_unit"] = np.where(
            times_df["days"] < DAYS_PER_YEAR,
            "months:",
            "years: "
        )

        def format_time(x):
            # cast floats that are whole numbers into int
            return str(int(x)) if float(x).is_integer() else str(x)
        
        times_df['survival_p'] = self.out(np.array(times_df['days'].tolist()))
        times_df["survival %"] = (times_df["survival_p"] * 100)

        print("    time     survival")
        for _, row in times_df.iterrows():
            time_str = format_time(row["time_normalised"])
            unit_str = row["time_unit"]
            
            # fixed width, 1 decimal, right-aligned
            surv_str = f"{row['survival %']:5.1f}%"
            
            print(f"{time_str:>4} {unit_str:<6} {surv_str} {row.get('label','')}")


    
    def predict(self, unseen_data):
        s = str(self)
        s_len = len(re.sub(r'\033\[[0-9;]*m', '', s))
        print("-" * (s_len + 12))
        print("::::: " + s + " :::::")
        print("-" * (s_len + 12))
        
        # Discrete end points
        if self.data['endpoint'] in [EndPoint.TumorSubtype, EndPoint.Sex]:
            self.predict_discrete(unseen_data)
            # @todo: self.export_outcome_discrete() # to file
            # @todo: self.plot_outcome_discrete() # make a plot

        # Right censored end points
        elif self.data['endpoint'] in [EndPoint.OverallSurvival, EndPoint.ProgressionFreeSurvival]:
            self.predict_right_censored(unseen_data)
            # @todo: self.export_outcome_right_censored() # to file
            # @todo: self.plot_outcome_right_censored() # make a plot
        
        else:
            print(f"not implemented type of endpoint: {self.data['endpoint']}")

    
    def print_outcome(self):
        if not hasattr(self, 'out'):
            raise RuntimeError("No prediction available. Call predict() before print_outcome().")

        if self.data['endpoint'] in [EndPoint.TumorSubtype, EndPoint.Sex]:
            self.print_outcome_discrete() # to stdout

        # Right censored end points
        elif self.data['endpoint'] in [EndPoint.OverallSurvival, EndPoint.ProgressionFreeSurvival]:
            self.print_outcome_right_censored()

    def __str__(self):
        applicable = ' & '.join([key for key, value in self.data['applicable on'].items() if value])
        
        return f"{self.data['date']}: {self.data['predictor type']} [{self.data['endpoint']}; {applicable}] (n={self.data['trained on']['total']})"

__all__ = [
    'EndPoint',
    'PredictorType',
    'Predictor'
]


