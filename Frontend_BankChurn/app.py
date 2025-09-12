import streamlit as st
import requests
import pandas as pd
import json


# Input fields for user

st.title("Bank Churn Prediction App")
Flask_url = "https://afifi00-flask-bankchurn-prediction.hf.space"

# Input fields
customer_id = st.text_input('Customer ID', '15634602')
surname = st.text_input('Surname', 'Hargrave')
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=600)
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=20, value=5)
balance = st.number_input('Balance', value=50000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])
estimated_salary = st.number_input('Estimated Salary', value=100000.0)

# Create a dictionary from the inputs
data = {
    'CustomerId': customer_id,
    'Surname': surname,
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

if st.button("Predict"):
    response = requests.post(f"{Flask_url}/predict", json=data)
    if response.status_code == 200:
        result = response.json()
        prediction = result.get("Prediction", None)
        try:
            prediction = int(float(prediction))
        except Exception:
            st.error("Invalid prediction format from backend.")
            prediction = None

        if prediction == 1:
            st.write(f"Customer with ID {customer_id} is likely to churn.")
        elif prediction == 0:
            st.write(f"Customer with ID {customer_id} is not likely to churn.")
        else:
            st.write("No valid prediction returned.")
    else:
        st.error(f"Error in API request: {response.status_code}")
        st.error(response.text)


