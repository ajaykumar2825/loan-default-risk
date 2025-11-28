## 📊 Loan Default Risk Prediction App

An end-to-end machine learning project that predicts the likelihood of loan default using historical credit data.  
Built with **XGBoost** for performance and **SHAP/LIME** for explainability, deployed as an interactive **Streamlit app**.

---

## 🚀 Project Overview
Financial institutions face challenges in identifying high-risk borrowers.  
This project provides a transparent, explainable solution to predict default risk and highlight the key factors driving each prediction.

---

## ✨ Features
- 📂 Upload borrower data (CSV)
- 🔮 Predict default risk probability
- 📊 Visualize global feature importance (SHAP)
- 🔎 Explain individual predictions (LIME)
- 🌐 Interactive Streamlit app

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn)
- **XGBoost** (imbalanced classification handling)
- **SHAP & LIME** (model explainability)
- **Streamlit** (interactive UI)
- **Matplotlib/Seaborn** (visualizations)

---

---

## live demo
<a href = https://loan-default-risk-fp3mpbhgpz38rnk5fvdd2f.streamlit.app/> click here </a>

## 📈 Model Performance
| Model                  | ROC-AUC | F1 (Default) | Recall (Default) |
|-------------------------|---------|--------------|------------------|
| Logistic Regression     | 0.71    | 0.10         | 0.05             |
| XGBoost (weighted)      | 0.85    | 0.35         | 0.68             |

---

## 📸 Screenshots
<img width="1045" height="568" alt="pic 1" src="https://github.com/user-attachments/assets/a5c16881-2a2f-401c-991b-5c63b75d16dc" />
<img width="1038" height="497" alt="pic 2" src="https://github.com/user-attachments/assets/1dab737c-6f0b-44f0-9c72-c83875c6a87f" />
<img width="1160" height="600" alt="pic 3" src="https://github.com/user-attachments/assets/2883b75f-7030-47c7-b7eb-7a73f03f51da" />
<img width="1298" height="384" alt="pic 4" src="https://github.com/user-attachments/assets/2d9b9a3a-a929-405a-9b5e-4ec3f60f1237" />
---

## ⚙️ How to Run Locally
Clone the repository and install dependencies:

```bash
git clone https://github.com/ajaykumar2825/loan-default-risk.git
cd loan-default-risk
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📂 Folder Structure
```
loan-default-risk/
├── app.py                # Streamlit app
├── data/                 # Dataset
├── notebooks/            # Jupyter notebooks
├── screenshots/          # SHAP & LIME visuals
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 💡 Business Impact
- Helps lenders identify high-risk applicants early
- Improves transparency in credit decisions
- Supports compliance with explainable AI regulations

---

## 📜 License
This project is licensed under the MIT License.  
Feel free to use and adapt it for your own work.

---

## 👨‍💻 Author
**Ajay** — B.Tech Data Science (3rd Year)  
Passionate about building explainable, business-relevant ML solutions.  
📌 [LinkedIn](https://www.linkedin.com/in/k-ajay-kumar-a32810286) | [GitHub](https://github.com/ajaykumar2825)
```
