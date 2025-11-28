import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# -----------------------------
# Load and train model once
# -----------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("data/cs-training.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
    df["NumberOfDependents"].fillna(df["NumberOfDependents"].mode()[0], inplace=True)

    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]

    model = XGBClassifier(eval_metric="logloss", scale_pos_weight=13)
    model.fit(X, y)
    return model, X

model, X = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Loan Default Risk Predictor")

uploaded_file = st.file_uploader("Upload borrower data (CSV)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # ✅ Align uploaded CSV with training features
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Predict default probability
    prediction = model.predict_proba(input_df)[:, 1]
    st.write("Default Risk Probability:", prediction)

    # SHAP explanation
    explainer = shap.Explainer(model, X)
    shap_values = explainer(input_df)

    st.subheader("SHAP Feature Impact")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)