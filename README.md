# Career Growth Analytics MVP

AI Career Platform lifecycle analytics MVP for synthetic event generation, growth analytics, churn prediction, and Next Best Action recommendations.

This repository contains the MVP and modeling portions of the Deepmanifold take-home project. It focuses on local, reproducible data science workflows: data generation, validation, funnel analysis, cohort retention, A/B experiment analysis, churn label construction, churn modeling, model evaluation, explainability, and notebook-based presentation.

No cloud services, external APIs, backend server, or frontend application are required for this MVP repository.

## Scope

The MVP includes:

- Synthetic user and event data generation.
- Data schema validation and quality checks.
- Growth funnel analysis.
- Cohort retention analysis.
- A/B experiment analysis with sample ratio mismatch detection.
- Churn label construction without data leakage.
- Churn prediction model training and evaluation.
- Model explainability and subgroup metrics.
- Rule-based Next Best Action recommendations.
- End-to-end Jupyter notebooks.
- Automated pytest coverage.

The Enterprise full-stack system is intentionally excluded from this MVP export. The Enterprise version lives in the separate full repository and contains the FastAPI backend, SQLite data layer, and React frontend.

## Business Context

The simulated product helps students and early-career professionals explore career paths, upload resumes, receive job recommendations, complete growth tasks, and generate career reports. The growth platform measures and optimizes the user journey from signup through activation, retention, churn risk, and recommended interventions.

Core user journey:

```text
signup
-> onboarding_complete
-> profile_complete
-> resume_upload
-> job_recommendation_view
-> job_save
-> growth_task_complete
-> career_report_generate
-> retained / churned
```

## Repository Structure

```text
career-growth-analytics-mvp/
|-- artifacts/                  # Formal model artifacts and plots
|-- data/
|   |-- sample/                 # 1,000-user sample CSV data
|   `-- processed/              # Derived outputs such as labels.csv
|-- docs/
|   |-- data_schema.md          # Data schema reference
|   |-- methodology.md          # Synthetic data and label methodology
|   `-- model_card.md           # Churn model card
|-- notebooks/
|   |-- lifecycle_analysis.ipynb
|   `-- churn_modeling.ipynb
|-- scripts/
|   |-- generate_data.py
|   |-- run_analysis.py
|   |-- compute_summary.py
|   |-- build_notebook.py
|   `-- train_churn_model.py
|-- src/career_growth/
|   |-- analytics/
|   |-- data_generation/
|   |-- decisions/
|   |-- features/
|   |-- modeling/
|   |-- validation/
|   |-- config.py
|   `-- schemas.py
|-- tests/
|-- PHASE2_MODELING_REPORT.md
|-- pyproject.toml
|-- README.md
`-- LICENSE
```

## Technology Stack

- Python 3.10+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib / seaborn
- pydantic
- pytest
- Jupyter

## Installation

Create and activate a virtual environment in the repository root using a real CPython interpreter.

On Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On macOS or Linux:

```bash
/path/to/real/python -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -e ".[dev]"
```

The `.venv/` directory is ignored by Git.

## Generate Data

The repository includes a 1,000-user sample dataset under `data/sample/`. To regenerate it:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts/generate_data.py --count 1000 --seed 42
```

To generate the full 5,000-user training dataset without overwriting the committed sample data:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts/train_churn_model.py --count 5000 --seed 42
```

The training script writes temporary training data to `data/training/`, which is ignored by Git.

## Run Analytics

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts/run_analysis.py
```

This runs validation, funnel analysis, retention analysis, experiment analysis, and Next Best Action examples.

## Train Churn Model

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts/train_churn_model.py --count 5000 --seed 42
```

The modeling pipeline:

- Builds pre-cutoff features using only events before signup day 7.
- Attaches churn labels from the day 8-21 label window.
- Splits users chronologically into train, validation, and test sets.
- Trains Logistic Regression and HistGradientBoostingClassifier models.
- Selects the best model by validation PR-AUC.
- Selects the operating threshold on validation data.
- Evaluates once on the held-out test set.
- Saves metrics, metadata, model artifacts, plots, subgroup metrics, user explanations, and NBA examples under `artifacts/`.

## Model Artifacts

Key committed artifacts:

- `artifacts/churn_model.joblib`
- `artifacts/model_metadata.json`
- `artifacts/metrics.json`
- `artifacts/feature_schema.json`
- `artifacts/explainability.json`
- `artifacts/user_explanations.json`
- `artifacts/subgroup_metrics.csv`
- `artifacts/subgroup_metrics.json`
- `artifacts/nba_examples.csv`
- `artifacts/nba_examples.json`
- `artifacts/plots/*.png`

## Run Tests

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest tests -q
```

## Open Notebooks

```powershell
.venv\Scripts\jupyter.exe notebook notebooks/lifecycle_analysis.ipynb
.venv\Scripts\jupyter.exe notebook notebooks/churn_modeling.ipynb
```

To rebuild the lifecycle notebook non-interactively:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts/build_notebook.py
```

## Churn Label Definition

- Prediction cutoff: signup timestamp + 7 days.
- Label window: day 8 through day 21 after signup.
- `is_churned = 1` if the user has no `event_source == "user_action"` events in the label window.
- Only users with a complete 21-day observation window are included.

## A/B Experiment

`exp_onboarding_v1` compares three onboarding flows:

- `control` -- standard five-step onboarding (40%)
- `personalized` -- adaptive onboarding (30%)
- `simplified` -- two-step onboarding (30%)

Primary metrics:

- onboarding completion rate
- profile completion rate
- D7 retention rate

The treatment effects are injected into the synthetic data through a causal mechanism. The results demonstrate the analytics pipeline and should not be interpreted as real product performance.

## Reports

- `PHASE2_MODELING_REPORT.md`
- `docs/data_schema.md`
- `docs/methodology.md`
- `docs/model_card.md`

## License

MIT License -- see `LICENSE`.
