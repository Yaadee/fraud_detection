
import pandas as pd

def create_features(df):
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['signup_hour'] = df['signup_time'].dt.hour
    df['signup_day'] = df['signup_time'].dt.dayofweek
    df['purchase_hour'] = df['purchase_time'].dt.hour
    df['purchase_day'] = df['purchase_time'].dt.dayofweek
    df['time_diff'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df = df.drop(columns=['signup_time', 'purchase_time'])
    return df

def process_fraud_data():
    df = pd.read_csv('data/processed/processed_Fraud_Data.csv')
    df = create_features(df)
    df.to_csv('data/processed/processed_Fraud_Data.csv', index=False)

if __name__ == "__main__":
    process_fraud_data()
