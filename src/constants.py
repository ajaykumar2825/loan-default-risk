from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, MODEL_DIR, DATABASE_DIR = (ROOT / "data", ROOT / "models", ROOT / "database")
DATA_PATH, MODEL_PATH = DATA_DIR / "loan_default_dataset.csv", MODEL_DIR / "loan_model.pkl"
DB_PATH = DATABASE_DIR / "history.db"
NUMERIC_COLUMNS = ["Age", "Annual_Income", "Monthly_Income", "Loan_Amount", "Loan_Term", "Interest_Rate", "Credit_Score", "Existing_EMI", "Debt_To_Income_Ratio", "Number_of_Open_Loans", "Years_at_Job", "Savings_Balance", "Missed_Payments", "Previous_Default"]
CATEGORICAL_COLUMNS = ["Gender", "Marital_Status", "Education", "Employment_Type", "Loan_Purpose", "Property_Area"]
RAW_FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
