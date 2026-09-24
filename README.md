<p align="center">
  <img src="assets/logo.png" alt="COGNITION - isoCitrate dehydrOGenase mutaNt glIoma objecTIve tumOr gradiNg" width="380">
</p>

<p align="center">
  <b>isoCitrate dehydrOGenase mutaNt glIoma objecTIve tumOr gradiNg</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+"></a>
</p>

A Python package for methylation-based tumor classification and survival prediction using Illumina DNA methylation arrays. COGNITION predicts prognosis of brain tumor subtypes, sample sex, and includes survival analysis using various statistical and deep learning models.

## Features

- **Survival prediction** for multiple tumor types using:
  - Statistical models: Cox PH, CoxNet, Weibull AFT, LogNormal AFT, LogLogistic AFT
  - Deep learning: DeepSurv, DeepHit, PCHazard, Random Survival Forest
  - Outputs: Risk scores, survival curves, median survival time

- **Tumor classification**: Predicts brain tumor subtypes using LDA and logistic regression

- **Sex prediction**: Classifies sample sex from methylation profiles

- **Copy number variation (CNV)**: Detects and visualizes CNV from methylation data

- **Array support**: EPIC, EPICv2, and 450K Illumina DNA methylation arrays

- **Pre-trained models** available via HuggingFace Hub

## Requirements

- **Python**: 3.12 or higher
- **PyTorch**: For deep learning models (CPU or GPU)
  - GPU (optional): NVIDIA GPU with CUDA for accelerated deep learning
- **Dependencies**: See `pyproject.toml` for full list (lifelines, scikit-learn, scikit-survival, pycox, etc.)

## Installation

### Quick Installation

```bash
# Clone and install in editable mode
git clone https://github.com/cognition-consortium/cognition.git
cd cognition
pip install --editable .
```

### Manual Installation (with specific PyTorch build)

For GPU support, install PyTorch with CUDA first:

```bash
# Example: PyTorch with CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Then install cognition
pip install --editable .
```

### Download latest prediction models

```bash
cognition pull  # Download latest models and references
```

### Verify Installation

```bash
cognition list  # Lists available predictors
```


<!-- STATS:START -->

### Performance (v1.0.3)

Cross-validated performance of the models packaged with COGNITION v1.0.3. Bars are scaled 0.50–1.00 and longer is better; the IBS bar spans the full 0–1 range, with IBS light on the left and 1 − IBS dark on the right. See [docs/MODELS-v1.0.3.md](docs/MODELS-v1.0.3.md) for the per-tumour-type and leave-one-dataset-out breakdowns.

#### Overall survival

| Model | IBS (CV) | C-index (CV) |
|---|---|---|
| **LogLogisticAFT**<br><sub>2026-08-26</sub> | `░▐██████████` 0.140 ± 0.021 | `██████▋░░░░░` 0.778 ± 0.006 |
| **LogNormalAFT**<br><sub>2026-08-26</sub> | `░▐██████████` 0.140 ± 0.018 | `██████▌░░░░░` 0.772 ± 0.009 |
| **CoxPH**<br><sub>2026-08-26</sub> | `░▐██████████` 0.142 ± 0.019 | `██████▌░░░░░` 0.771 ± 0.013 |
| **WeibullAFT**<br><sub>2026-08-26</sub> | `░▐██████████` 0.144 ± 0.018 | `██████▌░░░░░` 0.773 ± 0.008 |
| **MTLRDeep**<br><sub>2026-08-27</sub> | `░▐██████████` 0.145 ± 0.016 | `██████▍░░░░░` 0.768 ± 0.009 |
| **Coxnet**<br><sub>2026-08-26</sub> | `░░██████████` 0.147 ± 0.018 | `██████▌░░░░░` 0.771 ± 0.009 |
| **LuckSurvival**<br><sub>2026-08-26</sub> | `░░██████████` 0.148 ± 0.022 | `██████▊░░░░░` 0.780 ± 0.015 |
| **GLMBoost**<br><sub>2026-08-26</sub> | `░░██████████` 0.148 ± 0.018 | `██████▍░░░░░` 0.768 ± 0.010 |
| **DeepCoxIBrier**<br><sub>2026-08-26</sub> | `░░██████████` 0.149 ± 0.014 | `██████▉░░░░░` 0.785 ± 0.008 |
| **DeepSurv**<br><sub>2026-08-26</sub> | `░░██████████` 0.163 ± 0.035 | `██████▋░░░░░` 0.775 ± 0.011 |
| **DeepHit**<br><sub>2026-08-26</sub> | `░░██████████` 0.165 ± 0.016 | `██████▊░░░░░` 0.780 ± 0.010 |
| **LogisticHazard**<br><sub>2026-08-26</sub> | `░░██████████` 0.165 ± 0.031 | `██████▍░░░░░` 0.764 ± 0.016 |
| **MTLRSklearn**<br><sub>2026-08-27</sub> | `░░██████████` 0.171 ± 0.020 | `██████▏░░░░░` 0.755 ± 0.007 |
| **GradientBoosting**<br><sub>2026-08-26</sub> | `░░██████████` 0.174 ± 0.027 | `██████░░░░░░` 0.752 ± 0.012 |
| **PCHazard**<br><sub>2026-08-26</sub> | `░░██████████` 0.178 ± 0.017 | `██████▍░░░░░` 0.767 ± 0.012 |
| **XGBoostAFT**<br><sub>2026-08-27</sub> | `░░██████████` 0.186 ± 0.008 | `██████▏░░░░░` 0.755 ± 0.009 |

#### Tumor subtype

| Model | Performance |
|---|---|
| **LinearDiscriminantAnalysis**<br><sub>2026-08-25</sub> | not exported |

#### Sex

| Model | Performance |
|---|---|
| **LogisticRegression**<br><sub>2026-08-25</sub> | not exported |

Other releases: [v1.0.2](docs/MODELS-v1.0.2.md)

<sub>5-fold cross-validation (training cohort), mean ± SD over folds; c-index via `c_index_sksurv`, IBS via `brier_pycox`. COGNITION v1.0.3, newest model 2026-08-27. Generated from `assets/models/**/*.info.json` by `scripts/update_stats.py` — do not edit by hand.</sub>

<!-- STATS:END -->


## Usage

Pre-trained models are available on HuggingFace:
- **[ErasmusMC-Neuro-Oncology/cognition](https://huggingface.co/ErasmusMC-Neuro-Oncology/cognition/tree/main)**



### Command-Line Interface (CLI)

```bash
# List available predictors
cognition list

# Run prediction on IDAT files
cognition predict path/to/sample_Grn.idat path/to/sample_Red.idat

# Run with specific genome build
cognition predict --hg19 path/to/sample_Grn.idat path/to/sample_Red.idat
```

### Python API

```python
from libcognition import Predictor, EndPoint
from libcognition.utils import idat_to_data_container_mepylome

# Load and convert IDAT files
data = idat_to_data_container_mepylome("sample_Grn.idat", "sample_Red.idat")

# Create predictor (model auto-loads from assets/models/)
predictor = Predictor()

# Predict overall survival risk
os_scores = predictor.predict(data, endpoint=EndPoint.OverallSurvival)

# Predict progression-free survival with survival curves
pfs_pred = predictor.predict(data, endpoint=EndPoint.ProgressionFreeSurvival)
risk_scores = pfs_pred.risk_scores
survival_curves = pfs_pred.survival_curves
median_survival = pfs_pred.median_survival

# Predict tumor subtype
tumor_type = predictor.predict(data, endpoint=EndPoint.TumorType)

# Predict sample sex
sex = predictor.predict(data, endpoint=EndPoint.Sex)

# Predict copy number variation
cnv = predictor.predict(data, endpoint=EndPoint.CopyNumber)
```

## Pre-trained Models

Pre-trained models are available on HuggingFace:

- **[ErasmusMC-Neuro-Oncology/cognition](https://huggingface.co/ErasmusMC-Neuro-Oncology/cognition/tree/main)** — Model zoo with multiple tumor types and prediction targets

Models are automatically downloaded and cached in `assets/models/` on first use.

## Development

### Setup

```bash
# Install in editable mode with development dependencies
pip install --editable .

# Run tests
pytest tests/test_custom_transformers.py -v

# Run a specific model test
pytest tests/test_custom_transformers.py::test_cox -v
```

### Project Structure

- **`libcognition/`** - Core library
  - `predictors.py` - Main `Predictor` class and `EndPoint` enum
  - `custom_transformers.py` - Model wrappers (Cox, DeepSurv, DeepHit, etc.)
  - `calibrators.py` - Risk score calibration
  - `utils.py` - IDAT file conversion, array utilities
  - `cli.py` - Command-line interface
  - `notebook_functions.py` - Analysis utilities for Jupyter notebooks

- **`assets/models/`** - Pre-trained model files (`.pkl` + `.info.json` metadata)

- **`tests/`** - Unit tests for model wrappers

- **`notebooks/`** - Analysis and training notebooks

### Architecture

COGNITION uses a **wrapper pattern** to standardize the API across heterogeneous survival models (statistical, tree-based, deep learning). All model wrappers inherit from `SurvivalPredictorExtendedBase` and expose four prediction methods:

- `predict_risk_scores()` - Normalized risk scores [0-1]
- `predict_survival()` - Survival probabilities at time points
- `predict_median()` - Median survival time
- `predict_all()` - Combined output (scores, curves, median)

### Adding a New Survival Model

1. Create a wrapper class in `libcognition/custom_transformers.py` inheriting from `SurvivalPredictorExtendedBase`
2. Implement `fit(X, y, event_observed)` and the four prediction methods
3. Train your model using a notebook in `notebooks/predict_survival_*.ipynb`
4. Serialize to `assets/models/{model_name}.pkl` and create `{model_name}.info.json` metadata
5. Add test coverage in `tests/test_custom_transformers.py`

## Citation

If you use COGNITION in your research, please cite:

```bibtex
@software{cognition2024,
  title={COGNITION: Methylation-based tumor classification and survival prediction},
  author={Hoogstrate, Youri and Schoonhoven, Richard},
  year={2024},
  url={https://github.com/cognition-consortium/cognition}
}
```

## License

[License information to be added - check repository for details]

## Contributing

Contributions are welcome! Please:

1. Follow the existing code style and architecture
2. Add tests for new features
3. Update documentation as needed
4. Run tests before submitting PRs: `pytest tests/`

## Support & Issues

For bugs, feature requests, or questions:
- Create an issue on GitHub

---

