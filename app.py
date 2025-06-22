import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model, feature_names = joblib.load("anxiety_rf_model.pkl")

st.title("Smart City - Anxiety Risk Predictor")
st.write("This app predicts the anxiety risk level (Low, Moderate, High) based on individual inputs.")

# Input sliders
age = st.slider("Age", 10, 80, 25)
sleep = st.slider("Sleep Hours", 0.0, 12.0, 6.0)
physical = st.slider("Physical Activity (hrs/week)", 0.0, 15.0, 2.0)
caffeine = st.slider("Caffeine Intake (mg/day)", 0, 1000, 200)
alcohol = st.slider("Alcohol Consumption (drinks/week)", 0, 20, 2)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
heart_rate = st.slider("Heart Rate (bpm)", 50, 130, 80)
breathing = st.slider("Breathing Rate (breaths/min)", 10, 30, 18)
sweating = st.slider("Sweating Level (1-5)", 1, 5, 3)
diet = st.slider("Diet Quality (1-10)", 1, 10, 5)

# Predict button
if st.button("Predict Anxiety Level"):
    input_data = [[age, sleep, physical, caffeine, alcohol, stress, heart_rate, breathing, sweating, diet]]
    input_df = pd.DataFrame(input_data, columns=feature_names)
    prediction = model.predict(input_df)[0]
    levels = {0: "Low", 1: "Moderate", 2: "High"}
    st.success(f"Predicted Anxiety Level: **{levels[prediction]}**")

