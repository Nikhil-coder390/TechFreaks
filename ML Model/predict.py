# Importing necessary packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Load the trained model
loaded_model = pickle.load(open('Model.sav', 'rb'))

# Input data for prediction
Dependents = 'No'
tenure = 1
OnlineSecurity = 'No'
OnlineBackup = 'Yes'
DeviceProtection = 'No'
TechSupport = 'Yes'
Contract = 'Month-to-month'
PaperlessBilling = 'No'
MonthlyCharges = 2.85
TotalCharges = 56.85

# Create a dataframe with input data
data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]
df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

# Encode categorical features
encoder = LabelEncoder()
for feature in df.columns:
    if df[feature].dtypes == 'O':
        df[feature] = encoder.fit_transform(df[feature])

# Make predictions
single = loaded_model.predict(df)
probability = loaded_model.predict_proba(df)[:,1]

# Print prediction result
if single == 1:
    print("This Customer is likely to be Churned!")
    print(f"Confidence level is {np.round(probability*100, 2)}")
else:
    print("This Customer is likely to continue!")
    print(f"Confidence level is {np.round(probability*100, 2)}")
