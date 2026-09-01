from __future__ import annotations
import numpy as np
import pandas as pd
from .constants import DATA_DIR, DATA_PATH

def generate_dataset(n: int = 15000, force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    if DATA_PATH.exists() and not force: return pd.read_csv(DATA_PATH)
    r = np.random.default_rng(2026)
    age = r.integers(21, 67, n); income = np.clip(r.lognormal(13.25, .58, n), 120000, 4200000).round()
    monthly = (income / 12 * r.normal(1, .035, n)).clip(10000).round()
    employment = r.choice(["Salaried", "Self-employed", "Business", "Unemployed"], n, p=[.54,.2,.21,.05])
    credit = np.clip(r.normal(685, 75, n) - (employment == "Unemployed") * 45, 300, 900).round().astype(int)
    amount = np.clip(income * r.lognormal(-1.3, .62, n), 30000, 3500000).round()
    dti = np.clip(r.beta(2.1, 5, n) + (income < 300000)*.08, .03, .89).round(2)
    missed = r.poisson(.7 + (credit < 600)*.8, n).clip(0, 12)
    previous = r.binomial(1, np.clip(.04 + missed*.025 + (credit < 570)*.08, .02, .55))
    term = r.choice([12,24,36,48,60,84], n, p=[.05,.12,.35,.15,.27,.06])
    rate = np.clip(8 + (700-credit)*.045 + dti*8 + previous*3 + r.normal(0,1.2,n), 6, 32).round(2)
    emi = (amount / term * (1 + rate/100*.5) + r.normal(0, 1200, n)).clip(0).round()
    df = pd.DataFrame({"Loan_ID":[f"LN{202600000+i:06d}" for i in range(n)], "Age":age, "Gender":r.choice(["Female","Male","Non-binary"],n,p=[.47,.51,.02]), "Marital_Status":r.choice(["Single","Married","Divorced"],n,p=[.36,.55,.09]), "Education":r.choice(["High School","Graduate","Postgraduate"],n,p=[.28,.53,.19]), "Employment_Type":employment, "Annual_Income":income, "Monthly_Income":monthly, "Loan_Amount":amount, "Loan_Term":term, "Interest_Rate":rate, "Credit_Score":credit, "Existing_EMI":emi, "Debt_To_Income_Ratio":dti, "Number_of_Open_Loans":r.poisson(2.1,n).clip(0,12), "Loan_Purpose":r.choice(["Home","Vehicle","Education","Business","Personal","Medical"],n,p=[.25,.19,.13,.15,.21,.07]), "Property_Area":r.choice(["Urban","Semiurban","Rural"],n,p=[.48,.31,.21]), "Years_at_Job":np.minimum(r.poisson(6,n),42), "Savings_Balance":np.clip(income*r.lognormal(-2.25,.75,n),0,2000000).round(), "Missed_Payments":missed, "Previous_Default":previous})
    risk = -2.2 + (680-credit)/105 + dti*1.8 + missed*.22 + previous*1.1 + (amount/income)*.28 + (employment == "Unemployed")*.7 + (df.Number_of_Open_Loans > 5)*.25
    df["Default"] = (risk >= np.quantile(risk, .80)).astype(int)
    df.to_csv(DATA_PATH, index=False)
    return df

def ensure_data() -> pd.DataFrame: return generate_dataset()
