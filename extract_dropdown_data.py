import pandas as pd
import os
import joblib

# Load original CSV
df = pd.read_csv('data/ev_sales_india.csv')

# Drop rows with missing values in dropdown-relevant columns
df = df.dropna(subset=['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'])

# Get sorted unique values
dropdown_data = {
    'states': sorted(df['State'].unique()),
    'vehicle_classes': sorted(df['Vehicle_Class'].unique()),
    'vehicle_categories': sorted(df['Vehicle_Category'].unique()),
    'vehicle_types': sorted(df['Vehicle_Type'].unique())
}

# Save to file
os.makedirs('model', exist_ok=True)
joblib.dump(dropdown_data, 'model/dropdown_data.pkl')
print("✅ Dropdown data saved at model/dropdown_data.pkl")