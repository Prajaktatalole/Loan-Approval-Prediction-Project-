# importing the libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd 
st.title(" Welcome to Loan Approval System ")

# loading the model
loaded_model=joblib.load('model.pkl')
scaler=joblib.load('scaler.pkl')

# Prediction
def prediction_fun(user_input):
    prediction=loaded_model.predict(user_input)
    if prediction==0:
        return "You are not Eligible for loan"
    else:
        return "You are Eligible for loan "

# A function for taking inputs from user 


def main():
    st.title("User Data")

    # Sidebar inputs
    st.header("Enter Users Data")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income", step=1000)
    loan_amount = st.number_input("Loan Amount",  step=1000)
    loan_term = st.number_input("Loan Term (in months)",  step=1)
    cibil_score = st.number_input("CIBIL Score",  step=1)
    residential_assets_value = st.number_input("Residential Assets Value",  step=1000)
    commercial_assets_value = st.number_input("Commercial Assets Value", step=1000)
    luxury_assets_value = st.number_input("Luxury Assets Value", step=1000)
    bank_asset_value = st.number_input("Bank Asset Value", step=1000)
    
    # Create DataFrame
    data = {
        "Number of Dependents": [no_of_dependents],
        "Education": [education],
        "Self Employed": [self_employed],
        "Annual Income": [income_annum],
        "Loan Amount": [loan_amount],
        "Loan Term (in months)": [loan_term],
        "CIBIL Score": [cibil_score],
        "Residential Assets Value": [residential_assets_value],
        "Commercial Assets Value": [commercial_assets_value],
        "Luxury Assets Value": [luxury_assets_value],
        "Bank Asset Value": [bank_asset_value]
    }
    df = pd.DataFrame(data)
    
    # Display the dataframe
    st.write("### User Input Data:")
    st.dataframe(df)

    if education=='Graduate':
        education_1=0
    else:
        education_1=1
        
    if self_employed=='Yes':
        self_employed_1=1
    else:
        self_employed_1=0

    
    user_input=np.array([[no_of_dependents,education_1,self_employed_1,income_annum,
                          loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,
                          luxury_assets_value,bank_asset_value]])
    user_input_scaled=scaler.transform(user_input)

    if st.button("Will my loan get Approved ?"):
        result=prediction_fun(user_input_scaled)
        st.success(result)

if __name__ == "__main__":
    main()

    
    
