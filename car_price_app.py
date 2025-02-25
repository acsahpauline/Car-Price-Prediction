import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Streamlit UI
st.title("Car Price Predictor")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212; /* Dark Grey */
        color: #FFD700; /* Yellow */
    }
    .stMarkdown, .stTextInput label, .stNumberInput label {
        color: #FFFFFF !important; /* White for column names */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.write("Enter the car details to predict its price:")

# User inputs
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Mileage (in km)", min_value=0, max_value=500000, value=30000)
engine = st.number_input("Engine Size (in cc)", min_value=500, max_value=6000, value=1500)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
company = st.selectbox("Car Company", ["BMW", "Honda", "Toyota", "Hyundai", "Mercedes", "Ford", "Mahindra", "Tata", "Audi"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Convert categorical values
input_data = pd.DataFrame([[year, mileage, engine, fuel, company, transmission]], columns=['Year', 'Mileage', 'Engine', 'Fuel', 'Company', 'Transmission'])

# One-hot encode to match model's columns
input_data = pd.get_dummies(input_data, columns=['Fuel', 'Company', 'Transmission'], drop_first=True)

# Ensure all columns match model's training columns
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing columns with default value 0

input_data = input_data[model_columns]  # Reorder columns

# Make prediction
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Car Price: ₹{predicted_price:.2f} Lakhs")
