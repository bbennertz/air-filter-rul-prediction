# Remaining Useful Life Prediction for Air Filters

Final project for **IND8217E (Analytics for Faults)** — exchange semester.
Authors: Bruno Albino Bennertz and Rodrigo Fortunato Carvalho.

The project moves an air-filter maintenance policy from **preventive** (replace after a fixed interval) to **predictive** (replace based on the estimated Remaining Useful Life, RUL), using the *Preventive to Predictive Maintenance* dataset (Hochschule Esslingen).

## Problem

A test rig feeds dust into an air filter. As dust accumulates, the differential pressure Δp across the filter rises, and the filter is considered **failed when Δp > 600 Pa**.

- Preventive maintenance replaces filters at a fixed interval, which discards a lot of useful life: on the test set it uses only **72.4 %** of the filter life.
- Training runs are almost never observed to failure (only 5 of 50), so the data are **right-censored**.

Goal: predict RUL (in seconds) on the test runs from the Δp history and operating conditions, handling the censoring properly.

## Approach

| Model | Idea |
|---|---|
| M1 Weibull AFT | Classical survival/reliability model with covariates (`lifelines`) |
| M2 Random forest | Regression on engineered features with pseudo-labels for censored runs |
| M3 XGBoost | Same features and pseudo-labels, gradient boosting |
| M4 Autoregressive | Recursive Δp forecasting rolled out to the 600 Pa threshold — needs no RUL labels |

Pipeline follows the instructor's *Project Workflow*: problem definition → data definition → variable categorization → data processing → tools → execution plan → execution → analysis → documentation.

## Results (test set, last observed point, n = 50)

| Model | RMSE [s] | MAE [s] | Within ±10 s | PHM score |
|---|---|---|---|---|
| M1 Weibull AFT | 24.3 | 17.1 | 46 % | 790.8 |
| M2 Random forest | 19.5 | 13.8 | 52 % | 683.0 |
| M3 XGBoost | 18.8 | 13.3 | 54 % | 320.8 |
| **M4 Autoregressive** | **15.2** | **10.9** | **62 %** | **161.0** |

Maintenance-policy comparison:

| Policy | Unplanned failures | Life used |
|---|---|---|
| Preventive (fixed interval) | 5 | 72.4 % |
| Predictive – M1 | 11 | 89.5 % |
| Predictive – M2 | 19 | 92.5 % |
| Predictive – M3 | 10 | 90.1 % |
| Predictive – M4 | 10 | 91.5 % |

Predictive policies recover roughly 17–20 percentage points of useful life, at the cost of more unplanned failures unless a safety margin is added. Full discussion in [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md) (Portuguese) and [docs/Project_Report.pdf](docs/Project_Report.pdf).

## Repository layout

```
.
├── Filter_RUL_Project.ipynb   # full analysis; regenerates figures/ and results/
├── data/                      # Train/Test CSVs (+ original Data.mat)
├── figures/                   # fig01 … fig15 produced by the notebook
├── results/                   # metrics, predictions, numbers.json
├── docs/
│   ├── Project_Report.pdf/.docx
│   ├── PROJECT_EXPLAINED.md   # step-by-step walkthrough (PT-BR)
│   └── Preventive to Predictive Maintenance dataset.pdf
└── requirements.txt
```

## Reproducing

Python 3.12:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Filter_RUL_Project.ipynb
```

Run from the repository root — the notebook reads `data/*.csv` and writes to `figures/` and `results/`.

## Data

*Preventive to Predictive Maintenance* dataset, Hochschule Esslingen; see the dataset description PDF in [docs/](docs/) for source and terms of use.
