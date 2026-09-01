# LendSight — AI-Powered Loan Default Risk Prediction

LendSight is a portfolio-ready Streamlit application for exploring a loan book, scoring individual and bulk applications, explaining risk signals, and maintaining a local decision audit trail. It is an educational decision-support project and is not intended to automate lending decisions.

## Features

- Automatically creates a realistic, reproducible synthetic portfolio of 15,000 applicants with a 20% default rate.
- Interactive portfolio explorer, executive dashboard, filters, Plotly visualisations, and correlation matrix.
- Single applicant form with probability gauge, affordability/fraud warning, plain-English explanation, JSON export, and PDF report.
- Batch CSV scoring with validation, high-risk table, risk pie chart, and results download.
- Local SQLite prediction history with search, filters, CSV export, and deletion.
- Training lab comparing Logistic Regression, Decision Tree, Random Forest, and XGBoost, with holdout metrics and ROC curves.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

On first run the app builds `data/loan_default_dataset.csv`, trains and saves a model in `models/loan_model.pkl`, and initializes `database/history.db`. These artifacts are excluded from Git and can be recreated in Settings.

## Layout

```text
app.py                  Streamlit dashboard and navigation
src/data.py             Synthetic data generation
src/ml.py               Feature engineering, training, prediction
src/database.py         SQLite history repository
src/reports.py          PDF assessment report
src/ui.py               Fintech dashboard styling
```

The included Dockerfile exposes Streamlit on port 8501 and can be used for Render or any container runtime.
