#!/usr/bin/env python

from __future__ import annotations

import atexit
import click
import glob
from typing import TYPE_CHECKING
from pathlib import Path

from . import __version__, ASSETS_PATH, DISCLAIMER, HUGGINGFACE_URL

if TYPE_CHECKING:
    import numpy as np
    from .predictors import Predictor



def load_predictor(json_file: str) -> Predictor:
    from .predictors import Predictor
    try:
        return Predictor(json_file)
    except Exception as e:
        raise Exception(f"Failed to load predictor from '{json_file}': {e}") from e


def scan_for_local_models() -> list[str]:
    """Glob the predictor '.info.json' files under ASSETS_PATH."""
    out = {}

    glob_path = f"{ASSETS_PATH}/models/v{__version__}/"
    print(f"Scanning for models files under: {glob_path}")

    # sorted, because glob follows the filesystem: without it the order the
    # predictors come back in differs per machine, and anything downstream that
    # depends on which one comes last silently differs with it
    json_files = sorted(glob.glob(f"{glob_path}/*/*/*.info.json"))
    print(f"Found {len(json_files)} predictor(s)\n")

    for json in json_files:
        pred = load_predictor(json)
        if pred.data['endpoint'] not in out:
            out[pred.data['endpoint']] = []
        
        out[pred.data['endpoint']].append(pred)
    
    return out

        



# Printed on the way out. That covers every subcommand, a run that crashes, and
# `--help` -- which click renders and exits on its own, before any code here
# gets a turn.
atexit.register(print, "\n" + DISCLAIMER)


@click.group(invoke_without_command=True)
@click.version_option(__version__)
@click.pass_context
def main(ctx):
    if ctx.invoked_subcommand is not None:
        return
    click.echo(f"""
   A--T
   A--T       COGNITION v{__version__}
  *C--G
   G--C*     Objective DNA methylation based overall
   T--A       survival prediction of IDH mutant gliomas
   A--T

Based on Illumina DNA methylation arrays, this application can predict:

  - prognosis of several types of brain tumor(s)
  - the sex of the sample
  - the tumor subtype needed for plotting appropriate reference data
    or warn if for the predicted subtype no survival data was fitted""")


@main.command(name="list")
def list_():
    """Lists available predictors"""
    import numpy as np

    predictors_by_endpoint = scan_for_local_models()
    predictors = [pred for preds in predictors_by_endpoint.values() for pred in preds]
    predictors.sort(key=lambda p: (p.data['endpoint'], p.data['date']))

    n = len(predictors)
    width = len(str(n))
    prev_endpoint = None

    performance_map = {
        'c_index_sksurv': 'C-idx',
        'brier_pycox': 'IBS',

        # emitted by run_cross_validation__classification
        'accuracy': 'Acc.',
        'balanced_accuracy': 'Acc. (Bal)',
        'roc_auc': 'AUC',

        # hand-computed in older notebooks; kept so already exported models
        # keep rendering their performance line
        'mean accuracy': 'Acc.',
        'std_accuracy': 'sd Acc.',
        'mean balanced accuracy': 'Acc. (Bal)',
        'std_balanced_accuracy': 'sd Acc. (Bal)'
    }

    for i, pred in enumerate(predictors, start=1):
        endpoint = pred.data['endpoint']
        if endpoint != prev_endpoint:
            if prev_endpoint is not None:
                print()
            print(f"{'=' * 40} {endpoint} {'=' * 40}")
            prev_endpoint = endpoint
        
        suffix = ""

        if 'performance' in pred.data:
            for key in ['cv', 'lodo']:
                if key in pred.data['performance']:
                    suffix += f"\n    - {key.upper()}:"
                    for metric, value in pred.data['performance'][key]['Overall'].items():
                        if metric in performance_map:
                            metric_name = performance_map[metric]
                            # nanmean: a per-fold metric can legitimately be
                            # NaN (e.g. ROC-AUC on a single-class LODO fold)
                            suffix += f" {metric_name}: {np.nanmean(value):.3f};"

        if 'calibrations' in pred.data:
            suffix += f"\n    - calibrations: {', '.join(pred.data['calibrations']['expected_survival'].keys())}"
        
        if 'grade_only_results' in pred.data:
            suffix += f"\n    - Stats on WHO Grade: "
            suffix += ', '.join(     f"{k}: {v}" for k, v in pred.data['grade_only_results']['Tumor Type'].value_counts().items() )

        print(f"[{i:{width}}/{n}] {pred}{suffix}")



@main.command(name="pull")
def pull_():
    """Pull latest model builds from Hugging Face"""
    from .utils import pull_from_huggingface

    print(f"Pull latest model builds from Hugging Face: {HUGGINGFACE_URL} for version v{__version__}")
    pull_from_huggingface(
        HUGGINGFACE_URL,
        ASSETS_PATH,
        remote_subfolder=f"models/v{__version__}/",
    )



@main.command()
@click.argument("idat_grn")
@click.argument("idat_red")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Print diagnostic information (assets path, discovered model files).")
def run(idat_grn, idat_red, verbose):
    """Runs all classifiers on idat_grn and idat_red"""
    from .predictors import EndPoint
    from .utils import idat_to_mvalues
    from .outputs import plot_survival, export_survival
    from .notebook_functions import merge_grade_curves

    import pandas as pd

    mvalues, array_type, sentrix_id = idat_to_mvalues(idat_grn, idat_red)

    default_stem = f"{sentrix_id}.cognition-v{__version__}"
    out = {}

    grade_only_results = None
    grade_curve_members = {}
    sex_prediction = None
    subtype_prediction = None
    warnings = []

    # CNV
    from libcognition.mepylome_helpers import NORMALS_BASE, mepylome_idat_to_cnv_cli
    cnv_base = Path(idat_grn).name.removesuffix(".gz").removesuffix(".idat").removesuffix("_Red").removesuffix("_Grn")
    try:
        mepylome_idat_to_cnv_cli(
                    idat_grn,
                    idat_red,
                    NORMALS_BASE,
                    cnv_base
                     )
    except Exception as e:
        # a failed CNV run must not take the survival report down with it: the
        # panel drops out by itself further down, since its files are absent
        warnings.append(f"CNV analysis failed ({type(e).__name__}: {e}). "
                        f"The report is written without the CNV panel.")

    # same names mepylome_idat_to_cnv_cli writes them under; plot_survival
    # skips the CNV panel by itself when they are not there
    cnv_bins_file = Path(f"{cnv_base}.cognition-v{__version__}.cnv_bins.hg38.csv")
    cnv_detail_file = Path(f"{cnv_base}.cognition-v{__version__}.cnv_detail.hg38.csv")


    # other stuff
    predictors = scan_for_local_models()

    if verbose:
        print(f"Assets path : {ASSETS_PATH}")
        print(f"Model files : {predictors}")

    for endpoint in predictors:
        for pred in predictors[endpoint]:
            # Flagged at export time (export_classifier). Skipped before it is
            # run at all: it must stay out of `out`, which is what the report
            # averages, and out of `grade_curve_members`, so the predicted
            # curve and the reference cloud keep averaging the same members.
            if pred.data.get('exclude from running', False):
                print(f"Model found but flagged as discarded: {pred}")
                print("")
            elif not pred.data['applicable on'][array_type]:
                print(f"Classifier not suited for array type: {array_type}")
            else:
                if 'features' not in pred.data:
                    raise Exception("Invalid format -- feature names not defined")

                mvalues_index_set = set(mvalues.index)
                missing_features = [f for f in pred.data['features'] if f not in mvalues_index_set]
                if missing_features:
                    raise Exception(f"Missing features: {missing_features[:10]}")

                subset = mvalues.loc[pred.data['features']]
                pred.predict(subset.values.transpose())
                pred.print_outcome()
                print("")

                # Handle Sex endpoint
                if pred.data['endpoint'] == EndPoint.from_string("sex"):
                    class_names = pred.compiled['label_encoder'].classes_
                    sex_prediction = dict(zip(class_names, pred.out))

                # Handle Tumor Subtype endpoint
                if pred.data['endpoint'] == EndPoint.from_string("tumor subtype"):
                    class_names = pred.compiled['label_encoder'].classes_
                    class_and_prob = list(zip(class_names, pred.out))
                    class_and_prob.sort(key=lambda x: x[1], reverse=True)

                    df = pd.DataFrame(class_and_prob, columns=['Tumor Subtype', 'Prediction Probability'])

                    subtype_prediction = dict(class_and_prob)

                    # 1. Check if Oligo/Astro in classifier training
                    required_classes = {'Oligodendroglioma', 'Astrocytoma'}
                    available_classes = set(df['Tumor Subtype'])

                    if not required_classes.issubset(available_classes):
                        missing = required_classes - available_classes
                        warnings.append(
                            f"Unclear if specimen could be actually Astrocytoma or Oligodendroglioma, "
                            f"as classifier was not trained on these: {missing}"
                        )

                    # 2. Check if max probability belongs to Astro or Oligo
                    max_prob_idx = df['Prediction Probability'].idxmax()
                    max_tumor_type = df.loc[max_prob_idx, 'Tumor Subtype']
                    max_probability = df.loc[max_prob_idx, 'Prediction Probability']

                    if max_tumor_type not in {'Oligodendroglioma', 'Astrocytoma'}:
                        warnings.append(
                            f"Specimen was classified as {max_tumor_type} with p={max_probability:.4f}. "
                            f"Overall survival prediction is specifically designed for Oligodendroglioma and Astrocytoma."
                        )

                if pred.data['endpoint'] not in out:
                    out[pred.data['endpoint']] = {}
                out[pred.data['endpoint']][pred.data['model']] = pred.out

                # The reference cloud in the report is the ensemble's, not one
                # model's: every member that carries its reference curves
                # contributes, and they are averaged below. The scalar frame is
                # kept only as a fallback for models exported before the curves
                # were, and then it really is whichever predictor comes last.
                if 'grade curve times' in pred.data:
                    grade_curve_members[pred.data['model']] = pred.data
                elif 'grade_only_results' in pred.data:
                    grade_only_results = pred.data['grade_only_results']
                else:
                    print(f"'grade_only_results' not available for classifier {str(pred)}")

                if hasattr(pred, 'out_cal'):
                    out[pred.data['endpoint']][pred.data['model'] + "__calibrated_time"] = pred.out_cal

    # outside the endpoint loop: every endpoint has to have been seen before the
    # report is drawn, or the panels of whichever ones came last are missing
    survival_data = out.get(EndPoint("overall survival"))

    if grade_curve_members:
        grade_only_results = merge_grade_curves(grade_curve_members)
        print(f"Reference cloud: ensemble over {len(grade_curve_members)} models "
              f"({len(grade_only_results)} graded samples)")

        # the sample's own curve is the mean over every applicable model, so a
        # reference cloud built from fewer of them is a different ensemble and
        # the two panels stop being comparable
        n_members = len([k for k in (survival_data or {})
                         if "__calibrated" not in k])
        if n_members and len(grade_curve_members) < n_members:
            warnings.append(
                f"The reference cloud averages {len(grade_curve_members)} models "
                f"while the predicted curve averages {n_members}: the models "
                f"without exported reference curves are missing from the WHO grade "
                f"panels. Rebuild them so both are the same ensemble."
            )
    elif grade_only_results is not None:
        warnings.append(
            "No model exported its reference-cohort curves, so "
            "the WHO grade panels fall back on a single model's medians. Rebuild "
            "the overall survival models to get the ensemble reference cloud."
        )

    if survival_data is not None:
        export_path = f"{default_stem}.overall_survival.txt"
        survival_df = export_survival(survival_data, export_path)

        plot_path = f"{default_stem}.overall_survival.pdf"
        plot_survival(survival_df, plot_path, sentrix_id, grade_only_results,
                      cnv_bins_file=cnv_bins_file,
                      cnv_detail_file=cnv_detail_file,
                      sex_prediction=sex_prediction,
                      subtype_prediction=subtype_prediction)

    # Print all warnings at the end in red
    if warnings:
        print("\n" + "="*80)
        for warning in warnings:
            print(f"\033[91mWARNING: {warning}\033[0m")
        print("="*80)

    return warnings

