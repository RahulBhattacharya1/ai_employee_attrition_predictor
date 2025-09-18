import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/attrition_model.joblib")

st.title("Employee Attrition Predictor")

# Inputs
age = st.slider("Age", 18, 60, 30)
income = st.number_input("Monthly Income", 1000, 20000, 5000)
distance = st.slider("Distance From Home", 1, 30, 5)
overtime = st.selectbox("OverTime", ["Yes", "No"])
worklife = st.slider("WorkLifeBalance (1-4)", 1, 4, 3)

# Convert inputs to dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [income],
    "DistanceFromHome": [distance],
    "OverTime_Yes": [1 if overtime=="Yes" else 0],
    "WorkLifeBalance": [worklife]
})

# Prediction
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    st.write("Prediction:", "Employee may leave" if prediction==1 else "Employee likely to stay")
