import streamlit as st
import joblib
import numpy as np

# Load saved model
model = joblib.load("gadget_price_model.pkl")

st.title("📱 Gadget Price Prediction App")

ram = st.number_input("Enter RAM (GB)", 1, 32)
storage = st.number_input("Enter Storage (GB)", 16, 1024)
battery = st.number_input("Enter Battery (mAh)", 1000, 6000)
processor = st.number_input("Enter Processor Speed (GHz)", 1.0, 4.0)

if st.button("Predict Price"):
    features = np.array([[ram, storage, battery, processor]])
    prediction = model.predict(features)[0]
    st.success(f" Predicted Price: ₹{prediction:,.2f}")
