from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from .constants import CATEGORICAL_COLUMNS, MODEL_DIR, MODEL_PATH, MODEL_SCHEMA_VERSION, NUMERIC_COLUMNS, RAW_FEATURES

BINARY_VALUES = {"yes": 1, "y": 1, "true": 1, "1": 1, "default": 1, "previous default": 1, "no": 0, "n": 0, "false": 0, "0": 0, "none": 0, "no default": 0}

def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready raw inputs while accepting common CSV spellings safely."""
    missing = [column for column in RAW_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    clean = frame.loc[:, RAW_FEATURES].copy()
    for column in NUMERIC_COLUMNS:
        values = clean[column]
        if column == "Previous_Default":
            text = values.astype("string").str.strip().str.lower()
            values = text.map(BINARY_VALUES).where(text.isin(BINARY_VALUES), values)
        clean[column] = pd.to_numeric(values, errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        clean[column] = clean[column].astype("string").str.strip().fillna("Unknown")
    return clean

def engineer(frame):
    x = normalize_features(frame)
    x["Income_to_Loan_Ratio"] = x.Annual_Income / x.Loan_Amount.clip(lower=1)
    x["EMI_Burden"] = x.Existing_EMI / x.Monthly_Income.clip(lower=1)
    x["Loan_Utilization"] = x.Loan_Amount / x.Annual_Income.clip(lower=1)
    x["Financial_Stability_Score"] = (x.Credit_Score / 900 * 55 + (1-x.Debt_To_Income_Ratio)*25 + np.clip(x.Savings_Balance/x.Annual_Income,0,2)*10 + np.clip(x.Years_at_Job/20,0,1)*10)
    x["Risk_Category"] = pd.cut(x.Credit_Score, [0,580,680,1000], labels=["High","Medium","Low"]).astype(str)
    return x

def build_pipeline(kind="Random Forest"):
    engineered_num = NUMERIC_COLUMNS + ["Income_to_Loan_Ratio","EMI_Burden","Loan_Utilization","Financial_Stability_Score"]
    cats = CATEGORICAL_COLUMNS + ["Risk_Category"]
    prep = ColumnTransformer([("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), engineered_num), ("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cats)])
    choices = {"Logistic Regression":LogisticRegression(max_iter=800, class_weight="balanced"), "Decision Tree":DecisionTreeClassifier(max_depth=9, min_samples_leaf=15, class_weight="balanced", random_state=42), "Random Forest":RandomForestClassifier(n_estimators=180, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=42)}
    if kind == "XGBoost":
        try:
            from xgboost import XGBClassifier
            choices[kind] = XGBClassifier(n_estimators=220,max_depth=5,learning_rate=.06,subsample=.85,colsample_bytree=.85,eval_metric="logloss",random_state=42,n_jobs=2)
        except ImportError: choices[kind] = RandomForestClassifier(n_estimators=180, class_weight="balanced", n_jobs=-1, random_state=42)
    # SMOTE is fitted only on the training partition when this pipeline is fit.
    # Keeping it after preprocessing prevents test-set leakage during evaluation.
    return ImbPipeline([("prep",prep),("smote",SMOTE(random_state=42)),("model",choices[kind])])

def train_models(df, compare_all=False):
    if "Default" not in df:
        raise ValueError("Training data must include the Default target column.")
    X, y = engineer(df), pd.to_numeric(df.Default, errors="raise").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X,y,stratify=y,test_size=.2,random_state=42)
    names = ["Logistic Regression","Decision Tree","Random Forest","XGBoost"] if compare_all else ["Random Forest"]
    results=[]; curves={}; best=None; best_auc=-1
    for name in names:
        pipe=build_pipeline(name); pipe.fit(Xtr,ytr); p=pipe.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
        metrics={"model":name,"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred),"f1":f1_score(yte,pred),"roc_auc":roc_auc_score(yte,p)}; results.append(metrics)
        fpr,tpr,_=roc_curve(yte,p); curves[name]={"fpr":fpr,"tpr":tpr}
        if metrics["roc_auc"] > best_auc: best_auc,best=metrics["roc_auc"],{"pipeline":pipe,"name":name,"metrics":metrics}
    best.update({"schema_version": MODEL_SCHEMA_VERSION, "sklearn_version": sklearn.__version__, "raw_features": RAW_FEATURES})
    MODEL_DIR.mkdir(exist_ok=True); joblib.dump(best, MODEL_PATH)
    return best, pd.DataFrame(results).sort_values("roc_auc",ascending=False), curves

def ensure_model(df):
    """Load only a compatible artifact; retrain stale pickles automatically."""
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if (bundle.get("schema_version") == MODEL_SCHEMA_VERSION and bundle.get("sklearn_version") == sklearn.__version__ and bundle.get("raw_features") == RAW_FEATURES):
                return bundle
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError, EOFError):
            pass
    return train_models(df, False)[0]
def predict(bundle, frame): return bundle["pipeline"].predict_proba(engineer(frame))[:,1]
def feature_importance(bundle):
    pipe=bundle["pipeline"]; model=pipe.named_steps["model"]; names=pipe.named_steps["prep"].get_feature_names_out(); values=getattr(model,"feature_importances_", np.abs(getattr(model,"coef_",np.zeros((1,len(names))))).ravel())
    return pd.DataFrame({"feature":[str(n).replace("num__","").replace("cat__","") for n in names],"importance":values}).sort_values("importance",ascending=False)
def explain_prediction(bundle, applicant):
    a=normalize_features(pd.DataFrame([applicant])).iloc[0]; messages=[]
    if a["Credit_Score"] < 620: messages.append(f"Credit score of {a['Credit_Score']} is below the preferred range and increases risk.")
    if a["Debt_To_Income_Ratio"] > .42: messages.append(f"Debt-to-income ratio of {a['Debt_To_Income_Ratio']:.0%} indicates a heavy repayment burden.")
    if a["Missed_Payments"] > 1: messages.append(f"{a['Missed_Payments']} missed payments signal recent repayment stress.")
    if a["Previous_Default"]: messages.append("A previous default is a material adverse credit-history signal.")
    if a["Loan_Amount"] / max(a["Annual_Income"],1) > .75: messages.append("The requested amount is high relative to annual income.")
    if not messages: messages.append("Strong credit score, manageable debt burden, and stable affordability reduce risk.")
    return messages[:4]
