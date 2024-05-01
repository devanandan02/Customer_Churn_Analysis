import pandas as pd
from sklearn import preprocessing

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = preprocessing.LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

def preprocess_data(df):
    df = df.drop(['customerID','Partner','MultipleLines','InternetService','Dependents','StreamingTV','DeviceProtection',], axis = 1)
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.fillna(df["TotalCharges"].mean())
    df = df.apply(lambda x: object_to_int(x))
    df = df.dropna()
    df[['MonthlyCharges','TotalCharges']] = preprocessing.StandardScaler().fit_transform(df[['MonthlyCharges','TotalCharges']])
    x = df.drop(columns = ['Churn'])
    y = df['Churn'].values
    return x , y