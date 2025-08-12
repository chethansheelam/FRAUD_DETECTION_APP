import streamlit as st
import pandas as pd
import numpy as np 
import joblib

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('fraud_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model,scaler

model, scaler = load_model_and_scaler()
st.title('Financial Fraud Detection App 🕵️')
st.write('This app detects potential financial fraud based on transaction data.')
st.sidebar.header('Transaction Details')

def get_user_input():
    transaction_type = st.sidebar.selectbox('Transaction Type', ['TRANSFER', 'CASH_OUT'])
    amount = st.sidebar.number_input('Transaction Amount', min_value=0.0, format="%.2f")
    oldbalanceOrg = st.sidebar.number_input("Originator's Old Balance", min_value=0.0, format="%.2f")
    newbalanceOrig =st.sidebar.number_input("Originator's New Balance", min_value=0.0, format="%.2f")
    oldbalanceDest =st.sidebar.number_input("Recipient's Old Balance", min_value=0.0, format="%.2f")
    newbalanceDest = st.sidebar.number_input("Recipient's New Balance", min_value=0.0, format="%.2f")
    step = st.sidebar.slider('Time Step (Hour)' ,1,744,12)


    input_data = {
        'transaction_type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'step': step
    }
    return input_data

user_data = get_user_input()

if st.sidebar.button('Predict Fraud Status'):

    type_TRANSFER = 1 if user_data['transaction_type'] == 'TRANSFER' else 0
    errorBalance = user_data['newbalanceOrig'] + user_data['amount'] - user_data['oldbalanceOrg']
    isFlaggedFraud = 1 if user_data['amount'] > 200000 else 0


    features = np.array([[
        user_data['step'],
        user_data['amount'],
        user_data['oldbalanceOrg'],
        user_data['newbalanceOrig'],
        user_data['oldbalanceDest'],
        user_data['newbalanceDest'],
        isFlaggedFraud,
        errorBalance,
        type_TRANSFER
    ]])

    scaled_features = scaler.transform(features)

    predicton_proba = model.predict_proba(scaled_features)[0][1]
    prediction = 1 if predicton_proba > 0.5 else 0

    st.subheader('Prediction Result')
    if prediction == 1:
        st.error(f'FRAUD DETECTED (Probability: {predicton_proba:.2%})')
        st.write("This transaction shows strong indicators of fraudulent activity. It is highly recommended to block this transaction and investigate immediately.")
    else:
        st.success(f'Transaction Appears legitimate (Fraud Probability: {predicton_proba:.2%})')
        st.write("This transaction does not match known fraudulent patterns.")

st.write('---')
st.write("Disclaimer: This is a demo app based on a predictive model. All predictions should be reviewed by a human expert.")

