from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load the trained model
loaded_model = pickle.load(open('Model.sav', 'rb'))

@app.route('/predict', methods=['POST'])
def predict_churn():
    # Get input data from the request
    input_data = request.get_json()

    # Extract input features from the data
    Dependents = input_data['Dependents']
    tenure = input_data['tenure']
    OnlineSecurity = input_data['OnlineSecurity']
    OnlineBackup = input_data['OnlineBackup']
    DeviceProtection = input_data['DeviceProtection']
    TechSupport = input_data['TechSupport']
    Contract = input_data['Contract']
    PaperlessBilling = input_data['PaperlessBilling']
    MonthlyCharges = input_data['MonthlyCharges']
    TotalCharges = input_data['TotalCharges']

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

    # Prepare response
    if single == 1:
        result = {
            "prediction": "Churn",
            "confidence_level": float(probability)
        }
    else:
        result = {
            "prediction": "Continue",
            "confidence_level": float(probability)
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
