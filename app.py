import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("Employee Attrition Predictor")

PIPE_PATH = "models/attrition_pipeline.joblib"
DEFAULTS_PATH = "models/defaults_row.csv"

def load_pipeline_with_shims(path):
    try:
        return joblib.load(path)
    except AttributeError as e:
        # Known sklearn internal rename between versions: _RemainderColsList
        try:
            from sklearn.compose import _column_transformer
            if not hasattr(_column_transformer, "_RemainderColsList"):
                class _RemainderColsList(list):
                    pass
                _column_transformer._RemainderColsList = _RemainderColsList
            return joblib.load(path)
        except Exception as e2:
            raise e2

if not os.path.exists(PIPE_PATH):
    st.error("Missing models/attrition_pipeline.joblib. Upload the pipeline file from Colab.")
    st.stop()

if not os.path.exists(DEFAULTS_PATH):
    st.error("Missing models/defaults_row.csv. Upload the defaults row saved from Colab.")
    st.stop()

# Try load with shim fallback
pipe = load_pipeline_with_shims(PIPE_PATH)

defaults = pd.read_csv(DEFAULTS_PATH, dtype="object").iloc[0].to_dict()


# ---------------------------------------------------------------------
# Minimal inputs (you can expand these later). We will start from defaults,
# then update a few key fields based on the UI below, so ALL training
# columns exist for the pipeline.
# ---------------------------------------------------------------------
age_default = int(_safe_float(defaults.get("Age"), 35) or 35)
income_default = int(_safe_float(defaults.get("MonthlyIncome"), 5000) or 5000)
dist_default = int(_safe_float(defaults.get("DistanceFromHome"), 5) or 5)
wlb_default = int(_safe_float(defaults.get("WorkLifeBalance"), 3) or 3)

# For categoricals, fall back to safe choices if not in defaults
overtime_default = str(defaults.get("OverTime") or "No")
dept_default = str(defaults.get("Department") or "Research & Development")

age = st.slider("Age", 18, 65, age_default)
income = st.number_input("MonthlyIncome", min_value=500, max_value=100000, value=income_default, step=100)
distance = st.slider("DistanceFromHome", 0, 50, dist_default)
worklife = st.slider("WorkLifeBalance (1-4)", 1, 4, wlb_default)
overtime = st.selectbox("OverTime", ["No", "Yes"], index=0 if overtime_default not in ["Yes", "No"] else (0 if overtime_default == "No" else 1))
department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"],
                          index=0 if dept_default not in ["Research & Development","Sales","Human Resources"]
                          else ["Research & Development","Sales","Human Resources"].index(dept_default))

# ---------------------------------------------------------------------
# Build a full input row: start from defaults so every column from training exists
# ---------------------------------------------------------------------
row = defaults.copy()
row.update({
    "Age": age,
    "MonthlyIncome": income,
    "DistanceFromHome": distance,
    "WorkLifeBalance": worklife,
    "OverTime": overtime,
    "Department": department,
})

# Convert everything to a single-row DataFrame
X = pd.DataFrame([row])

# Strongly recommend: make sure numeric columns that were numeric in training are numeric here
# If you later add more UI fields, coerce them as needed.
for col in X.columns:
    # Try to coerce obvious numerics; leave categoricals as strings
    if col.lower() in ["age","monthlyincome","distancefromhome","worklifebalance",
                       "yearsatcompany","yearsincurrentrole","yearssincelastpromotion",
                       "yearswithcurrmanager","hourlyrate","dailyrate","monthlyrate",
                       "percentsalaryhike","totalworkingyears","environmentSatisfaction".lower(),
                       "joblevel","jobinvolvement","jobSatisfaction".lower(),"relationshipSatisfaction".lower()]:
        X[col] = pd.to_numeric(X[col], errors="ignore")

# ---------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------
if st.button("Predict Attrition"):
    try:
        proba = pipe.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)
        st.metric("Probability of Attrition", f"{proba:.2%}")
        st.write("Prediction:", "May Leave" if pred == 1 else "Likely to Stay")
    except Exception as e:
        st.error(f"Inference failed. Ensure defaults_row.csv matches the pipeline's training columns. {e}")
