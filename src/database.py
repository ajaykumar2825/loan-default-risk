import json, sqlite3
import pandas as pd
from .constants import DATABASE_DIR, DB_PATH
def connect(): DATABASE_DIR.mkdir(exist_ok=True); return sqlite3.connect(DB_PATH)
def init_database():
    with connect() as c: c.execute("CREATE TABLE IF NOT EXISTS prediction_history (ID INTEGER PRIMARY KEY AUTOINCREMENT, Timestamp TEXT DEFAULT CURRENT_TIMESTAMP, Risk_Level TEXT, Default_Probability REAL, Applicant_JSON TEXT, Reasons_JSON TEXT)")
def save_prediction(applicant, probability, level, reasons):
    with connect() as c: c.execute("INSERT INTO prediction_history (Risk_Level,Default_Probability,Applicant_JSON,Reasons_JSON) VALUES (?,?,?,?)", (level, probability, json.dumps(applicant), json.dumps(reasons)))
def get_history():
    with connect() as c: data = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY ID DESC", c)
    if not data.empty: data = pd.concat([data.drop(columns="Applicant_JSON"), data.Applicant_JSON.map(json.loads).apply(pd.Series)], axis=1)
    return data
def delete_prediction(record_id):
    with connect() as c: c.execute("DELETE FROM prediction_history WHERE ID=?", (int(record_id),))
def clear_history():
    with connect() as c: c.execute("DELETE FROM prediction_history")
