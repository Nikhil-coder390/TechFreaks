import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest, f_classif

# Load the data
data = pd.read_csv('Telco-Customer-Churn.csv')

# Data Cleaning
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
data.drop(columns=['customerID'], inplace=True)

# Feature Encoding
encoder = LabelEncoder()
for feature in data.columns:
    if data[feature].dtypes == 'O':
        data[feature] = encoder.fit_transform(data[feature])

# Separate into features and target
X = data.drop(columns='Churn')
y = data['Churn']

# Feature Selection
num_features = X.shape[1]  # Get the number of features in the input data
selector = SelectKBest(score_func=f_classif, k=min(10, num_features))  # Adjust k based on available features
X = selector.fit_transform(X, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SMOTEENN for handling imbalanced data
smoteenn = SMOTEENN()
X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)

# Train the model
gbc = GradientBoostingClassifier()
gbc.fit(X_train_resampled, y_train_resampled)

# Save the model using pickle
filename = 'model.pkl'
pickle.dump(gbc, open(filename, 'wb'))

def predict_churn(data):
    model = pickle.load(open('model.pkl', 'rb'))
    selected_features = selector.get_support(indices=True)
    data_selected = data[:, selected_features]
    prediction = model.predict(data_selected)
    confidence = model.predict_proba(data_selected)
    return prediction, confidence

def main():
    st.title('Telecom Customer Churn Prediction')

    # Input fields for customer information
    dependents = st.selectbox('Dependents:', ['No', 'Yes'])
    tenure = st.number_input('Tenure:')
    online_security = st.selectbox('OnlineSecurity:', ['No', 'Yes'])
    online_backup = st.selectbox('OnlineBackup:', ['No', 'Yes'])
    device_protection = st.selectbox('DeviceProtection:', ['No', 'Yes'])
    tech_support = st.selectbox('TechSupport:', ['No', 'Yes'])
    contract_options = ['Month-to-month', 'One year', 'Two year']
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    contract = st.selectbox('Contract:', contract_options)
    contract_numeric = contract_mapping[contract]
    paperless_billing = st.selectbox('PaperlessBilling:', ['No', 'Yes'])
    monthly_charges = st.number_input('MonthlyCharges:')
    total_charges = st.number_input('TotalCharges:')

    # Check for missing inputs
    missing_inputs = pd.isnull([dependents, tenure, online_security, online_backup, device_protection,
                                tech_support, contract_numeric, paperless_billing, monthly_charges, total_charges]).sum()
    if missing_inputs > 0:
        st.error(f'Please fill in all fields. {missing_inputs} fields are missing.')
    else:
        # Add a prediction button
        if st.button('Predict'):
            # Prepare data for prediction
            input_data = np.array([[dependents, tenure, online_security, online_backup, device_protection,
                                    tech_support, contract_numeric, paperless_billing, monthly_charges, total_charges]])
            selected_features = selector.get_support(indices=True)

            # Check if selected_features is within bounds
            if np.max(selected_features) < input_data.shape[1]:
                input_data_selected = input_data[:, selected_features]
                prediction, confidence = predict_churn(input_data_selected)

                # Display prediction and confidence level
                if prediction[0] == 1:
                    st.error('Customer is likely to churn!')
                    st.write("This Customer is likely to be Churned!")
                    st.write(f"Confidence level is {np.round(confidence[0][1]*100, 2)}")
                else:
                    st.success('Customer is likely to continue!')
                    st.write("This Customer is likely to continue!")
                    st.write(f"Confidence level is {np.round(confidence[0][0]*100, 2)}")
            else:
                st.error("Feature selection indices are out of bounds for the input data.")

if __name__ == "__main__":  
    main()
