#!/usr/bin/env python3

"""Display names for the classifier classes.

Its own stdlib-only leaf module rather than a home next to the enums in
predictors.py: scripts/update_stats.py runs from a git hook and must not drag
in pandas or torch, while libcognition/outputs.py needs the very same names for
its plot legends. Keeping it here is what stops a figure and the README from
disagreeing about what a model is called.
"""

# Display names only: the *.info.json "predictiontype" values and the paths
# under assets/models/ are left untouched. Suffixes that carry no information
# are stripped generically; anything that stays unwieldy gets an override.
STRIP_SUFFIXES = ("SurvivalAnalysis", "Fitter", "Wrapper")
DISPLAY_NAMES = {
    # Componentwise boosting with linear base learners is glmboost in mboost.
    "ComponentwiseGradientBoostingSurvivalAnalysis": "GLMBoost",
}


def display_name(model: str) -> str:
    if model in DISPLAY_NAMES:
        return DISPLAY_NAMES[model]
    for suffix in STRIP_SUFFIXES:
        if model.endswith(suffix) and len(model) > len(suffix):
            return model[: -len(suffix)]
    return model
