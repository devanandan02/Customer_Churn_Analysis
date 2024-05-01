import streamlit as st
import pandas as pd
import joblib
from sklearn import preprocessing

# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return joblib.load(model_path)

def main():
    st.title("Random Forest Customer Churn Prediction")

    # Load pre-trained model
    model = load_model("models/model.pkl")

    # Get input features from user
    st.sidebar.subheader("Enter Customer Information:")
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    seniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    tenure = st.sidebar.number_input("Tenure", min_value=1, value=1)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "Unknown"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "Unknown"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "Unknown"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "Unknown"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0, value=0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0, value=0)

    # Map string inputs to numeric values
    gender_map = {"Male": 0, "Female": 1}
    phone_service_map = {"Yes": 1, "No": 0}
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    paperless_billing_map = {"Yes": 1, "No": 0}
    payment_method_map = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}
    binary_map = {"Yes": 1, "No": 0, "Unknown": 2}

    # Map input values to numeric ones
    gender_numeric = gender_map[gender]
    phone_service_numeric = phone_service_map[phone_service]
    contract_numeric = contract_map[contract]
    paperless_billing_numeric = paperless_billing_map[paperless_billing]
    payment_method_numeric = payment_method_map[payment_method]
    online_security_numeric = binary_map[online_security]
    online_backup_numeric = binary_map[online_backup]
    tech_support_numeric = binary_map[tech_support]
    streaming_movies_numeric = binary_map[streaming_movies]

    scaler = preprocessing.StandardScaler()
    monthly_charges_scaled = scaler.fit_transform([[monthly_charges]])[0][0]
    total_charges_scaled = scaler.transform([[total_charges]])[0][0]
    # Create input DataFrame
    input_data = pd.DataFrame({
            "gender": [gender_numeric],
            "SeniorCitizen": [seniorCitizen],
            "tenure": [tenure],
            "PhoneService": [phone_service_numeric],
            "OnlineSecurity": [online_security_numeric],
            "OnlineBackup": [online_backup_numeric],
            "TechSupport": [tech_support_numeric],
            "StreamingMovies": [streaming_movies_numeric],
            "Contract": [contract_numeric],
            "PaperlessBilling": [paperless_billing_numeric],
            "PaymentMethod": [payment_method_numeric],
            "MonthlyCharges": [monthly_charges_scaled],
            "TotalCharges": [total_charges_scaled]
        })

    # Make prediction
    prediction_button = st.sidebar.button("Predict Churn")
    if prediction_button:
        # Predict churn
        prediction = model.predict(input_data)

        # Display prediction
        if prediction[0] == 1:
            st.write("Customer is likely to churn.")
        else:
            st.write("Customer is not likely to churn.")

if __name__ == "__main__":
    main()