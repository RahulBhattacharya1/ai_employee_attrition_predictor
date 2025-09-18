import os
import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("Employee Attrition Predictor")

# ----------------------------
# Helper functions
# ----------------------------
def _safe_float(x, fallback=None):
    try:
        return float(x)
    except Exception:
        return fallback

def _safe_int(x, fallback=None):
    v = _safe_float(x, fallback)
    try:
        return int(v) if v is not None else fallback
    except Exception:
        return fallback

# ----------------------------
# Artifact paths
# ----------------------------
PIPE_PATH = "models/attrition_pipeline.joblib"
DEFAULTS_PATH = "models/defaults_row.csv"

# ----------------------------
# Load artifacts
# ----------------------------
if not os.path.exists(PIPE_PATH):
    st.error("Missing models/attrition_pipeline.joblib. Upload the pipeline file from Colab.")
    st.stop()

if not os.path.exists(DEFAULTS_PATH):
    st.error("Missing models/defaults_row.csv. Upload the defaults row saved from Colab.")
    st.stop()

# Handle occasional sklearn pickle issues with a tiny shim
def load_pipeline_with_shims(path):
    try:
        return joblib.load(path)
    except AttributeError:
        # Common sklearn internal rename between versions: _RemainderColsList
        try:
            from sklearn.compose import _column_transformer
            if not hasattr(_column_transformer, "_RemainderColsList"):
                class _RemainderColsList(list):
                    pass
                _column_transformer._RemainderColsList = _RemainderColsList
            return joblib.load(path)
        except Exception as e2:
            raise e2

pipe = load_pipeline_with_shims(PIPE_PATH)

# Read defaults as strings to avoid dtype surprises, then coerce in UI
defaults_series = pd.read_csv(DEFAULTS_PATH, dtype="object").iloc[0]
defaults = defaults_series.to_dict()

# ----------------------------
# UI defaults (fallbacks if defaults_row.csv lacks a field)
# ----------------------------
age_default = _safe_int(defaults.get("Age"), 35) or 35
income_default = _safe_int(defaults.get("MonthlyIncome"), 5000) or 5000
dist_default = _safe_int(defaults.get("DistanceFromHome"), 5) or 5
wlb_default = _safe_int(defaults.get("WorkLifeBalance"), 3) or 3

overtime_default = str(defaults.get("OverTime") or "No")
dept_default = str(defaults.get("Department") or "Research & Development")

# ----------------------------
# UI
# ----------------------------
age = st.slider("Age", 18, 65, age_default)
income = st.number_input("MonthlyIncome", min_value=500, max_value=100000, value=income_default, step=100)
distance = st.slider("DistanceFromHome", 0, 50, dist_default)
worklife = st.slider("WorkLifeBalance (1-4)", 1, 4, wlb_default)
overtime = st.selectbox("OverTime", ["No", "Yes"], index=0 if overtime_default not in ["No","Yes"] else (0 if overtime_default == "No" else 1))
department = st.selectbox(
    "Department",
    ["Research & Development", "Sales", "Human Resources"],
    index=0 if dept_default not in ["Research & Development","Sales","Human Resources"]
    else ["Research & Development","Sales","Human Resources"].index(dept_default),
)

# ----------------------------
# Build a full input row starting from defaults so every training column exists
# ----------------------------
row = defaults.copy()
row.update({
    "Age": age,
    "MonthlyIncome": income,
    "DistanceFromHome": distance,
    "WorkLifeBalance": worklife,
    "OverTime": overtime,
    "Department": department,
})

X = pd.DataFrame([row])

# Coerce common numeric columns if they exist in training schema
numeric_like = [
    "Age","MonthlyIncome","DistanceFromHome","WorkLifeBalance",
    "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion",
    "YearsWithCurrManager","HourlyRate","DailyRate","MonthlyRate",
    "PercentSalaryHike","TotalWorkingYears","EnvironmentSatisfaction",
    "JobLevel","JobInvolvement","JobSatisfaction","RelationshipSatisfaction",
    "TrainingTimesLastYear"
]
for col in numeric_like:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict Attrition"):
    try:
        proba = pipe.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)
        st.metric("Probability of Attrition", f"{proba:.2%}")
        st.write("Prediction:", "May Leave" if pred == 1 else "Likely to Stay")
    except Exception as e:
        st.error(f"Inference failed. Ensure defaults_row.csv was built from the same RAW columns used to fit the pipeline. {e}")
