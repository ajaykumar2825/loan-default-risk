import streamlit as st
def apply_theme():
    st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');.stApp{background:#f7faf9;color:#102a2c;font-family:'DM Sans',sans-serif}.block-container{padding-top:2.4rem;padding-bottom:3rem;max-width:1380px}.hero{background:linear-gradient(120deg,#073b3a,#0f766e 62%,#14b8a6);border-radius:25px;padding:3.2rem 3.4rem;color:white;margin:0 0 1.5rem}.hero h1{font-family:'Playfair Display',serif;font-size:3.6rem;line-height:1.02;margin:.4rem 0 1rem}.hero h1 span{color:#a7f3d0}.hero p{max-width:650px;color:#d5f5ee;font-size:1.05rem}.eyebrow{letter-spacing:.15em!important;font-size:.72rem!important;font-weight:700}.metric-card{background:white;border:1px solid #e2ece9;border-radius:17px;padding:1.15rem 1.2rem;box-shadow:0 4px 18px #0f766e0b}.metric-label{color:#52706d;font-size:.8rem}.metric-value{font-size:1.75rem;font-weight:700;margin:.15rem 0}.metric-sub{color:#78918e;font-size:.75rem}.brand{color:#0f766e!important;margin-bottom:0}.sidebar-copy{color:#6f8885;font-size:.8rem;margin-top:0}.workflow{background:white;border:1px solid #e2ece9;border-radius:17px;padding:1.3rem;line-height:1.55}.risk{display:inline-block;border-radius:999px;padding:.42rem 1rem;font-weight:700}.risk-low{background:#ccfbf1;color:#115e59}.risk-medium{background:#fef3c7;color:#92400e}.risk-high{background:#ffe4e6;color:#9f1239}div[data-testid='stSidebar']{background:#ffffff;border-right:1px solid #e2ece9}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>
    #MainMenu, footer {visibility:hidden}
    h1,h2,h3 {letter-spacing:-.02em}
    .metric-label {font-weight:600;text-transform:uppercase;letter-spacing:.05em}
    div[data-testid='stButton'] button, div[data-testid='stDownloadButton'] button {border-radius:10px;font-weight:600;border-color:#99d8cf}
    div[data-testid='stButton'] button[kind='primary'] {background:#0f766e;border-color:#0f766e}
    div[data-testid='stExpander'] {border:1px solid #dce9e6;border-radius:13px;background:#fff}
    </style>""", unsafe_allow_html=True)
def metric_card(title,value,sub): st.markdown(f"<div class='metric-card'><div class='metric-label'>{title}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>",unsafe_allow_html=True)
def risk_badge(level): return f"<span class='risk risk-{level.lower()}'>● {level} risk</span>"
