# Insurance Risk Analytics

This repository contains a DVC-tracked insurance analytics workflow with a cleaning stage, EDA notebook, and reproducible pipeline outputs.

## Project Layout

- `data/MachineLearningRating_v3.txt.dvc` tracks the raw insurance dataset through DVC.
- `src/data_loader.py` handles dataset loading, separator detection, and validation.
- `src/clean_data.py` prepares the cleaned dataset for analysis.
- `notebooks/01_eda.ipynb` contains descriptive statistics, missing-value analysis, visualizations, and loss-ratio exploration.
- `.github/workflows/ci.yml` runs repository checks in CI.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

## DVC Workflow

Pull the tracked dataset and reproduce the pipeline end to end:

```bash
dvc pull
dvc repro
```

The pipeline defined in `dvc.yaml` reads `data/MachineLearningRating_v3.txt`, writes the cleaned output to `data/cleaned_insurance_data.txt`, and updates `dvc.lock`.

## EDA Notebook

Open `notebooks/01_eda.ipynb` to review:

- Dataset shape, column types, and descriptive statistics.
- Missing-value checks and the handling strategy used in cleaning.
- Claims and premium distributions.
- Loss-ratio views by province, vehicle type, gender, and time.
- Correlation and seasonal-style visualizations to support pricing review.

## CI

The GitHub Actions workflow installs the project dependencies and runs the test/lint checks defined for the repository. This keeps the repo verifiable even when the DVC data is not present in a fresh clone.
