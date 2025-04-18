import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

import sys
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from matplotlib.backends.backend_pdf import PdfPages
from graphs import generate_graphs

# Load data
df = pd.read_csv('data/ev_sales_india.csv')

# Parse date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Handle missing values safely
df['EV_Sales_Quantity'] = df['EV_Sales_Quantity'].fillna(df['EV_Sales_Quantity'].median())
df = df.fillna(df.mode().iloc[0])

# Feature engineering
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df_original = df.copy()

# Save dropdown values BEFORE encoding
os.makedirs('model', exist_ok=True)
dropdown_data = {
    'State': sorted(df['State'].dropna().unique()),
    'Vehicle_Class': sorted(df['Vehicle_Class'].dropna().unique()),
    'Vehicle_Category': sorted(df['Vehicle_Category'].dropna().unique()),
    'Vehicle_Type': sorted(df['Vehicle_Type'].dropna().unique())
}
joblib.dump(dropdown_data, 'model/dropdown_data.pkl')
print("Dropdown data saved.")

# Drop if 'Month_Name' column exists
if 'Month_Name' in df.columns:
    df.drop('Month_Name', axis=1, inplace=True)

# Encode categorical features
df = pd.get_dummies(df, columns=['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'], drop_first=True)

# Drop original date column
df.drop(['Date'], axis=1, inplace=True)

# Define features and target
X = df.drop('EV_Sales_Quantity', axis=1)
y = df['EV_Sales_Quantity']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with progress reporting
model = RandomForestRegressor(n_estimators=100, random_state=42)
n_estimators = model.n_estimators

# Split into 10 parts to simulate progress
batch_size = n_estimators // 10
for i in range(1, 11):  # 10 steps
    model.set_params(n_estimators=batch_size * i)  # Increase n_estimators gradually
    model.fit(X_train, y_train)  # Train model
    progress = (i * 10)  # Calculate percentage progress
    print(f"PROGRESS:{progress}", flush=True)  # Send progress to Flask app

# Final evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}")

# Save model and feature names
joblib.dump(model, 'model/model.pkl')
joblib.dump(list(X.columns), 'model/features.pkl')

# Generate graphs dynamically from original dataframe
generate_graphs(df_original)

