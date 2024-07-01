# src/data_preprocessing.py
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(df):
    # Implement missing value handling logic
    return df.dropna()

def clean_data(df):
    # Remove duplicates and correct data types
    df = df.drop_duplicates()
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def preprocess_fraud_data():
    df = load_data('data/raw/Fraud_Data.csv')
    df = handle_missing_values(df)
    df = clean_data(df)
    df.to_csv('data/processed/processed_Fraud_Data.csv', index=False)

if __name__ == "__main__":
    preprocess_fraud_data()
