import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the model
model = pickle.load(open('Model.sav', 'rb'))

# Sample input data (replace with actual input data)
data = {
    'Dependents': 'Yes',
    'tenure': 12,
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'MonthlyCharges': 85.0,
    'TotalCharges': 1020.0
}

# Convert data to DataFrame
df = pd.DataFrame([data])

# Encode categorical features
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = encoder.fit_transform(df[feature])

# Make prediction
prediction = model.predict(df)
probability = model.predict_proba(df)[:, 1] * 100

if prediction == 1:
    op1 = "This Customer is likely to be Churned!"
    op2 = f"Confidence level is {np.round(probability[0], 2)}"
else:
    op1 = "This Customer is likely to Continue!"
    op2 = f"Confidence level is {np.round(probability[0], 2)}"

print(op1)
print(op2)
