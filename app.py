# car_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv('car_data/car data.csv')

# Feature engineering
df['Car_Age'] = 2025 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Train-Test Split
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)

# ===============================
# Streamlit UI
# ===============================
st.title("🚗 Car Price Prediction App")
st.write("Predict the selling price of your car based on its details!")

# User Inputs
present_price = st.number_input("Present Price of Car (in Lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
car_age = st.slider("Car Age (Years)", min_value=0, max_value=30, step=1)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Convert categorical to numeric
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0
seller_individual = 1 if seller_type == "Individual" else 0
trans_manual = 1 if transmission == "Manual" else 0

# Prepare input for prediction
user_input = np.array([[present_price, kms_driven, owner, car_age,
                        fuel_diesel, fuel_petrol, seller_individual, trans_manual]])
user_input_poly = poly.transform(user_input)

# Predict button
if st.button("Predict Selling Price"):
    predicted_price = model.predict(user_input_poly)[0]
    # Cap prediction so it does not exceed Present Price
    predicted_price = min(predicted_price, present_price)
    st.success(f"💰 Predicted Selling Price: {round(predicted_price, 2)} Lakhs")
