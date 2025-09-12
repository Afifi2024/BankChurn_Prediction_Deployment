import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Add this to suppress a common TensorFlow warning

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

app = Flask("Bank Churn Predictor")

# --- CORRECT ---
# Load all models and artifacts once when the app worker starts.
# This is the correct and efficient method.
try:
    model = load_model("backend_files/best_model_adam.h5")
    label_encoders = joblib.load("backend_files/label_encoders.joblib")
    scaler = joblib.load("backend_files/sc.joblib")
    model_columns = joblib.load("backend_files/model_columns.joblib")
    print("Models and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading models on startup: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- CORRECT ---
        # The call to load_models() has been removed.
        data = request.get_json()
        df = pd.DataFrame([data])

        df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')

        for col, encoder in label_encoders.items():
            df[col] = encoder.transform(df[col])

        df = df.reindex(columns=model_columns, fill_value=0)

        df[['Balance', 'EstimatedSalary']] = scaler.transform(df[['Balance', 'EstimatedSalary']])

        pred_prob = model.predict(df)[0][0]
        prediction = int(pred_prob > 0.5)
        return jsonify({"Prediction": prediction})

    except Exception as e:
        # Add error handling for robustness
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
