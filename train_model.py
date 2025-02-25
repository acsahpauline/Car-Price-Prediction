import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("car_price_dataset.csv")

# Selecting features and target
X = df[['Year', 'Mileage', 'Engine', 'Fuel', 'Company', 'Transmission']]
y = df['Price']

# Convert categorical columns into numerical values
X = pd.get_dummies(X, columns=['Fuel', 'Company', 'Transmission'], drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and column names
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model trained and saved successfully!")
