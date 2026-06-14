import streamlit as st
import joblib
import numpy as np

model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.pkl")

st.title("🌱 AI Crop Recommendation System")

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Predict Crop"):
    sample = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    crop = encoder.inverse_transform(
        prediction.astype(int).reshape(-1, 1)
    )

    st.success(f"Recommended Crop: {crop[0][0]}")