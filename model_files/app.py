import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load saved artifacts
tf.random.set_seed(2)
np.random.seed(2)

model = load_model("bank_churn_best_model.h5", compile=False)
label_encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")
model_columns = joblib.load("model_columns.joblib")

# Input feature names per training
input_features = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

# Prepare options for categorical variables from encoders
geography_options = list(label_encoders['Geography'].classes_)
gender_options = list(label_encoders['Gender'].classes_)

def preprocess_input(data_dict):
    # Create DataFrame from dictionary
    df = pd.DataFrame([data_dict])

    # Encode categorical features using stored LabelEncoders
    for col in ['Geography', 'Gender']:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

    # Scale numeric columns
    df[['Balance', 'EstimatedSalary']] = scaler.transform(df[['Balance', 'EstimatedSalary']])

    # Reindex to model columns with 0 fill for any missing columns
    df = df.reindex(columns=model_columns, fill_value=0)

    return df

def predict_churn(
    CreditScore, Geography, Gender, Age, Tenure,
    Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
):
    input_dict = {
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary
    }

    processed_df = preprocess_input(input_dict)
    pred_prob = model.predict(processed_df)[0][0]
    prediction = int(pred_prob > 0.5)
    status = "likely to churn" if prediction == 1 else "not likely to churn"
    conf_score = f"Churn probability: {pred_prob:.3f}"
    result_text = f"Customer is {status}."
    return result_text, conf_score

# Create Gradio interface
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Credit Score", value=600, precision=0),
        gr.Dropdown(choices=geography_options, label="Geography"),
        gr.Dropdown(choices=gender_options, label="Gender"),
        gr.Number(label="Age", value=30, precision=0),
        gr.Number(label="Tenure (years)", value=5, precision=0),
        gr.Number(label="Balance", value=50000.0, precision=2),
        gr.Number(label="Number of Products", value=1, precision=0),
        gr.Dropdown(choices=[0, 1], label="Has Credit Card"),
        gr.Dropdown(choices=[0, 1], label="Is Active Member"),
        gr.Number(label="Estimated Salary", value=100000.0, precision=2),
    ],
    outputs=[
        gr.Textbox(label="Churn Prediction"),
        gr.Textbox(label="Confidence Score")
    ],
    title="Bank Customer Churn Prediction",
    description="Enter customer data to predict their probability of leaving the bank in the next 6 months."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
