# Random Forest Customer Churn Prediction

This is a simple Streamlit app for predicting customer churn using a Random Forest model.
The dataset is taken from Kaggle : https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/input?select=WA_Fn-UseC_-Telco-Customer-Churn.csv

## Overview

The app allows users to upload a pre-trained Random Forest model and input customer information to predict whether a customer is likely to churn or not. The input features include gender, senior citizenship, tenure, phone service, online security, online backup, tech support, streaming TV, streaming movies, contract type, paperless billing, payment method, monthly charges, and total charges.

## Installation

1. Clone this repository:
   https://github.com/devanandan02/Customer_Churn_Analysis.git

2. Navigate to the project directory:
   Customer_Churn_Analysis

3. Install the required Python packages:
   pip install -r requirements.txt

## Usage

1. Run the Streamlit app:
   streamlit run app.py

2. Enter customer information in the sidebar inputs.

3. Click the "Predict Churn" button to see the prediction.

## Folder Structure

├── app.py # Streamlit app script
├── src/
│ ├── init.py # Init file for src package
│ ├── data_loader.py # Module for loading data
│ ├── model_evaluator.py # Module for evaluating models
│ ├── model_saver.py # Module for saving models
│ ├── model_trainer.py # Module for training models
│ ├── utils.py # Utility functions
├── models/ # Folder to store saved models
├── data/ # Folder to store data files
├── README.md # Project README file
└── requirements.txt # List of Python dependencies
