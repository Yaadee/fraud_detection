
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(df):
    return df.dropna()

def clean_data(df):
    df = df.drop_duplicates()
    return df

def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def preprocess_fraud_data():
    df = load_data('data/raw/Fraud_Data.csv')
    df = handle_missing_values(df)
    df = clean_data(df)
    df = encode_categorical(df)
    df.to_csv('data/processed/processed_Fraud_Data.csv', index=False)

def preprocess_credit_data():
    df = load_data('data/raw/creditcard.csv')
    df = handle_missing_values(df)
    df.to_csv('data/processed/processed_creditcard.csv', index=False)

if __name__ == "__main__":
    preprocess_fraud_data()
    preprocess_credit_data()
