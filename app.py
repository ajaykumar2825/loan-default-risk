"""LendSight: synthetic, explainable loan default-risk workspace."""
from __future__ import annotations
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.constants import NUMERIC_COLUMNS, RAW_FEATURES
from src.data import ensure_data, generate_dataset
from src.database import clear_history, delete_prediction, get_history, init_database, save_prediction
from src.ml import ensure_model, explain_prediction, feature_importance, predict, train_models
from src.reports import build_pdf_report
from src.ui import apply_theme, metric_card, risk_badge

st.set_page_config(page_title="LendSight | Credit intelligence", page_icon="◆", layout="wide")
apply_theme(); init_database()
@st.cache_data(show_spinner=False)
def dataset(): return ensure_data()
@st.cache_resource(show_spinner="Preparing credit-risk model…")
def model_bundle(): return ensure_model(dataset())
def money(v): return f"₹{v:,.0f}"
def kpis(df):
    values=[("Applicants",f"{len(df):,}","portfolio records"),("Default rate",f"{df.Default.mean():.1%}","observed risk"),("Average loan",money(df.Loan_Amount.mean()),"requested capital"),("Credit score",f"{df.Credit_Score.mean():.0f}","portfolio average")]
    for col,(t,v,s) in zip(st.columns(4),values):
        with col: metric_card(t,v,s)

def home(df):
    st.markdown("""<section class='hero'><p class='eyebrow'>LENDSIGHT / DECISION INTELLIGENCE</p><h1>Make confident<br><span>credit decisions.</span></h1><p>Explainable default-risk scoring for faster, fairer lending. Explore a realistic 15,000-applicant portfolio, assess applicants, and keep an auditable decision history.</p></section>""",unsafe_allow_html=True)
    kpis(df); a,b=st.columns((1.25,1))
    with a:
        x=df.groupby("Loan_Purpose",as_index=False).Default.mean().sort_values("Default")
        st.plotly_chart(px.bar(x,x="Default",y="Loan_Purpose",orientation="h",color="Default",color_continuous_scale=["#5eead4","#f59e0b","#fb7185"],title="Default rate by loan purpose").update_layout(coloraxis_showscale=False,height=330),use_container_width=True)
    with b:
        st.subheader("Workflow")
        st.markdown("""<div class='workflow'><b>01 &nbsp; Explore</b><br><small>Review portfolio quality and demographic patterns.</small><br><br><b>02 &nbsp; Score</b><br><small>Assess one applicant or upload a batch.</small><br><br><b>03 &nbsp; Explain</b><br><small>See the signals that moved each decision.</small></div>""",unsafe_allow_html=True)
        st.caption("Demo data is synthetic and must not be used for real lending decisions.")

def explorer(df):
    st.title("Portfolio explorer")
    with st.expander("Filters",expanded=True):
        a,b,c=st.columns(3); gender=a.multiselect("Gender",df.Gender.unique(),default=df.Gender.unique()); education=b.multiselect("Education",df.Education.unique(),default=df.Education.unique()); purpose=c.multiselect("Loan purpose",df.Loan_Purpose.unique(),default=df.Loan_Purpose.unique())
        a,b,c=st.columns(3); income=a.slider("Annual income",int(df.Annual_Income.min()),int(df.Annual_Income.max()),(int(df.Annual_Income.quantile(.02)),int(df.Annual_Income.quantile(.98)))); score=b.slider("Credit score",int(df.Credit_Score.min()),int(df.Credit_Score.max()),(int(df.Credit_Score.quantile(.02)),int(df.Credit_Score.quantile(.98)))); area=c.multiselect("Property area",df.Property_Area.unique(),default=df.Property_Area.unique())
    view=df[df.Gender.isin(gender)&df.Education.isin(education)&df.Loan_Purpose.isin(purpose)&df.Property_Area.isin(area)&df.Annual_Income.between(*income)&df.Credit_Score.between(*score)]
    st.caption(f"{len(view):,} matching applicants"); a,b=st.columns(2)
    with a: st.plotly_chart(px.histogram(view,x="Credit_Score",color="Default",nbins=35,barmode="overlay",title="Credit-score distribution",color_discrete_map={0:"#5eead4",1:"#fb7185"}),use_container_width=True)
    with b: st.plotly_chart(px.box(view,x="Default",y="Loan_Amount",color="Default",title="Loan amount by outcome",color_discrete_map={0:"#5eead4",1:"#fb7185"}),use_container_width=True)
    a,b=st.columns(2)
    with a: st.plotly_chart(px.scatter(view.sample(min(1800,len(view)),random_state=1),x="Annual_Income",y="Loan_Amount",color="Default",size="Savings_Balance",hover_data=["Credit_Score","Loan_Purpose"],title="Income versus loan amount",color_discrete_map={0:"#5eead4",1:"#fb7185"}),use_container_width=True)
    with b: st.plotly_chart(px.violin(view,x="Employment_Type",y="Debt_To_Income_Ratio",color="Default",box=True,title="Debt burden by employment",color_discrete_map={0:"#5eead4",1:"#fb7185"}),use_container_width=True)
    st.plotly_chart(px.imshow(view[NUMERIC_COLUMNS+["Default"]].corr(),color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Numeric correlation matrix"),use_container_width=True)
    st.dataframe(view.head(300),use_container_width=True,hide_index=True)

def applicant_form(df):
    example=df.sample(1).iloc[0].to_dict()
    if st.button("Use a random demo applicant",use_container_width=True): st.session_state.demo=example
    s=st.session_state.get("demo",example); r={}
    with st.form("applicant_form"):
        st.subheader("Personal profile"); a,b,c,d=st.columns(4)
        r["Age"]=a.number_input("Age",18,80,int(s["Age"])); r["Gender"]=b.selectbox("Gender",["Female","Male","Non-binary"],index=["Female","Male","Non-binary"].index(s["Gender"])); r["Marital_Status"]=c.selectbox("Marital status",["Single","Married","Divorced"],index=["Single","Married","Divorced"].index(s["Marital_Status"])); r["Education"]=d.selectbox("Education",["High School","Graduate","Postgraduate"],index=["High School","Graduate","Postgraduate"].index(s["Education"]))
        a,b,c=st.columns(3); r["Employment_Type"]=a.selectbox("Employment type",["Salaried","Self-employed","Business","Unemployed"],index=["Salaried","Self-employed","Business","Unemployed"].index(s["Employment_Type"])); r["Years_at_Job"]=b.number_input("Years at job",0,45,int(s["Years_at_Job"])); r["Property_Area"]=c.selectbox("Property area",["Urban","Semiurban","Rural"],index=["Urban","Semiurban","Rural"].index(s["Property_Area"]))
        st.subheader("Financial profile"); a,b,c,d=st.columns(4)
        r["Annual_Income"]=a.number_input("Annual income (₹)",60000,10000000,int(s["Annual_Income"]),step=10000); r["Monthly_Income"]=b.number_input("Monthly income (₹)",5000,1000000,int(s["Monthly_Income"]),step=1000); r["Credit_Score"]=c.number_input("Credit score",300,900,int(s["Credit_Score"])); r["Savings_Balance"]=d.number_input("Savings balance (₹)",0,5000000,int(s["Savings_Balance"]),step=5000)
        a,b,c,d=st.columns(4); r["Existing_EMI"]=a.number_input("Existing EMI (₹)",0,500000,int(s["Existing_EMI"]),step=1000); r["Debt_To_Income_Ratio"]=b.slider("Debt-to-income ratio",0.,1.,float(s["Debt_To_Income_Ratio"]),.01); r["Number_of_Open_Loans"]=c.number_input("Open loans",0,20,int(s["Number_of_Open_Loans"])); r["Missed_Payments"]=d.number_input("Missed payments",0,20,int(s["Missed_Payments"]))
        st.subheader("Loan request"); a,b,c,d=st.columns(4)
        r["Loan_Amount"]=a.number_input("Loan amount (₹)",10000,10000000,int(s["Loan_Amount"]),step=10000); r["Loan_Term"]=b.selectbox("Term (months)",[12,24,36,48,60,84],index=[12,24,36,48,60,84].index(int(s["Loan_Term"]))); r["Interest_Rate"]=c.slider("Interest rate (%)",5.,36.,float(s["Interest_Rate"]),.1); r["Loan_Purpose"]=d.selectbox("Loan purpose",["Home","Vehicle","Education","Business","Personal","Medical"],index=["Home","Vehicle","Education","Business","Personal","Medical"].index(s["Loan_Purpose"])); r["Previous_Default"]=st.radio("Previous default",[0,1],horizontal=True,index=int(s["Previous_Default"]))
        return r if st.form_submit_button("Assess default risk",type="primary",use_container_width=True) else None

def show_result(applicant,bundle):
    probability=float(predict(bundle,pd.DataFrame([applicant]))[0]); level="High" if probability>=.55 else "Medium" if probability>=.28 else "Low"; reasons=explain_prediction(bundle,applicant)
    a,b=st.columns((.8,1.2))
    with a:
        fig=go.Figure(go.Indicator(mode="gauge+number",value=probability*100,number={"suffix":"%"},title={"text":"Default risk"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#fb7185" if level=="High" else "#f59e0b" if level=="Medium" else "#5eead4"},"steps":[{"range":[0,28],"color":"#e6fffb"},{"range":[28,55],"color":"#fef3c7"},{"range":[55,100],"color":"#ffe4e6"}]})); fig.update_layout(height=300,margin=dict(l=20,r=20,t=50,b=10)); st.plotly_chart(fig,use_container_width=True)
    with b:
        st.markdown(risk_badge(level),unsafe_allow_html=True); st.subheader("Decision recommendation"); st.write("Manual underwriting required" if level=="High" else "Request additional documentation" if level=="Medium" else "Eligible for standard review"); st.subheader("Primary drivers")
        for x in reasons: st.write(f"• {x}")
    if applicant["Annual_Income"]<applicant["Loan_Amount"]*.25 or applicant["Existing_EMI"]>applicant["Monthly_Income"]*.65: st.warning("Fraud / affordability warning: values indicate a potentially unsustainable request. Verify supporting documents.")
    payload={"default_probability":round(probability,4),"risk_level":level,"applicant":applicant,"reasons":reasons}; st.download_button("Download decision JSON",json.dumps(payload,indent=2),"loan_risk_decision.json","application/json"); st.download_button("Download PDF report",build_pdf_report(applicant,probability,level,reasons,bundle.get("name","Model")),"loan_risk_report.pdf","application/pdf")
    save_prediction(applicant,probability,level,reasons)

def single_prediction(df,bundle):
    st.title("Applicant assessment"); st.caption("Assess affordability and risk signals. This is an educational decision-support demo, not an automated lending decision.")
    x=applicant_form(df)
    if x: show_result(x,bundle)

def bulk_prediction(bundle):
    st.title("Bulk risk assessment"); st.caption("CSV must contain the 20 applicant fields used by the single-assessment form. Loan_ID is optional.")
    st.download_button("Download sample CSV",dataset().drop(columns="Default").head(25).to_csv(index=False).encode(),"sample_upload.csv","text/csv")
    upload=st.file_uploader("Upload applicants CSV",type="csv")
    if upload:
        frame=pd.read_csv(upload); missing=[c for c in RAW_FEATURES if c not in frame.columns]
        if missing: st.error("Missing columns: "+", ".join(missing)); return
        out=frame.copy(); prob=predict(bundle,out); out["Default_Probability"]=prob.round(4); out["Risk_Level"]=pd.cut(prob,[-.01,.28,.55,1],labels=["Low","Medium","High"])
        a,b=st.columns(2)
        with a: st.plotly_chart(px.pie(out,names="Risk_Level",title="Risk distribution",color="Risk_Level",color_discrete_map={"Low":"#5eead4","Medium":"#f59e0b","High":"#fb7185"}),use_container_width=True)
        with b: st.dataframe(out[out.Risk_Level=="High"].sort_values("Default_Probability",ascending=False),use_container_width=True,hide_index=True)
        st.download_button("Download scored results",out.to_csv(index=False).encode(),"bulk_risk_results.csv","text/csv")

def dashboard(df,bundle):
    st.title("Executive analytics"); kpis(df); a,b=st.columns(2)
    with a: st.plotly_chart(px.histogram(df,x="Loan_Amount",color="Default",nbins=35,title="Loan amount distribution",color_discrete_map={0:"#5eead4",1:"#fb7185"}),use_container_width=True)
    with b: st.plotly_chart(px.bar(df.groupby("Education",as_index=False).Default.mean(),x="Education",y="Default",title="Default rate by education",color="Default",color_continuous_scale="Reds"),use_container_width=True)
    a,b=st.columns(2)
    with a: st.plotly_chart(px.scatter(df.sample(2000,random_state=4),x="Credit_Score",y="Debt_To_Income_Ratio",color="Default",title="Credit score vs debt burden",color_discrete_map={0:"#5eead4",1:"#fb7185"}),use_container_width=True)
    with b: st.plotly_chart(px.bar(feature_importance(bundle).head(12).sort_values("importance"),x="importance",y="feature",orientation="h",title="Model feature importance",color="importance",color_continuous_scale="Teal"),use_container_width=True)

def training(df):
    st.title("Model training lab"); st.write("Compare four supervised models using a stratified holdout set. The model with the strongest ROC AUC becomes the saved model.")
    if st.button("Train and compare models",type="primary"):
        with st.spinner("Training models and evaluating the holdout set…"):
            bundle,scores,curves=train_models(df,compare_all=True); st.cache_resource.clear(); st.success(f"Selected {bundle['name']} as the best model and saved it locally."); st.dataframe(scores.style.format({"accuracy":"{:.3f}","precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}","roc_auc":"{:.3f}"}),use_container_width=True,hide_index=True)
            st.plotly_chart(go.Figure([go.Scatter(x=v["fpr"],y=v["tpr"],mode="lines",name=k) for k,v in curves.items()]).update_layout(title="ROC comparison",xaxis_title="False-positive rate",yaxis_title="True-positive rate"),use_container_width=True)

def explainability(df,bundle):
    st.title("Explainable AI"); st.caption("Global importance reflects how strongly each feature influences the saved model. Local explanations identify risk-increasing applicant signals.")
    st.plotly_chart(px.bar(feature_importance(bundle).head(15).sort_values("importance"),x="importance",y="feature",orientation="h",title="Global feature importance",color="importance",color_continuous_scale="Teal"),use_container_width=True)
    loan_id=st.selectbox("Select an applicant",df.Loan_ID.head(200).tolist()); applicant=df.loc[df.Loan_ID==loan_id].iloc[0][RAW_FEATURES].to_dict(); probability=float(predict(bundle,pd.DataFrame([applicant]))[0]); st.metric("Predicted default probability",f"{probability:.1%}")
    for reason in explain_prediction(bundle,applicant): st.write(f"• {reason}")

def history():
    st.title("Decision history"); hist=get_history()
    if hist.empty: st.info("No assessments have been recorded yet."); return
    q=st.text_input("Search loan ID, risk, or applicant data"); levels=st.multiselect("Risk level",["Low","Medium","High"],default=["Low","Medium","High"]); view=hist[hist.Risk_Level.isin(levels)]
    if q: view=view[view.astype(str).apply(lambda c:c.str.contains(q,case=False,na=False)).any(axis=1)]
    st.dataframe(view,use_container_width=True,hide_index=True); st.download_button("Download history CSV",view.to_csv(index=False).encode(),"prediction_history.csv","text/csv")
    if len(view):
        record=st.selectbox("History record to delete",view.ID.tolist())
        if st.button("Delete selected history record"): delete_prediction(record); st.rerun()

def settings(df):
    st.title("Settings & maintenance"); a,b,c=st.columns(3)
    if a.button("Retrain saved model",use_container_width=True):
        with st.spinner("Retraining…"): train_models(df,compare_all=False); st.cache_resource.clear(); st.success("Model retrained.")
    if b.button("Generate new data",use_container_width=True): generate_dataset(force=True); st.cache_data.clear(); st.cache_resource.clear(); st.success("Generated a new synthetic 15,000-applicant portfolio.")
    if c.button("Reset history",use_container_width=True): clear_history(); st.success("Prediction history cleared.")
    st.divider(); st.download_button("Export dataset",df.to_csv(index=False).encode(),"loan_default_dataset.csv","text/csv"); st.caption("Imported model artifacts are disabled in this demo: accept model files only from trusted, validated sources.")

PAGES={"Overview":home,"Data explorer":explorer,"Applicant assessment":single_prediction,"Bulk assessment":bulk_prediction,"Analytics dashboard":dashboard,"Model training":training,"Model explainability":explainability,"Prediction history":history,"Settings":settings}
with st.sidebar:
    st.markdown("<h2 class='brand'>◆ LendSight</h2><p class='sidebar-copy'>Credit intelligence workspace</p>",unsafe_allow_html=True); selected=st.radio("Navigation",list(PAGES),label_visibility="collapsed"); st.divider(); st.caption("Synthetic demo • v1.0")
df=dataset(); bundle=model_bundle(); page=PAGES[selected]
if selected in {"Overview","Data explorer","Model training","Settings"}: page(df)
elif selected in {"Applicant assessment","Analytics dashboard","Model explainability"}: page(df,bundle)
elif selected=="Bulk assessment": page(bundle)
else: page()
