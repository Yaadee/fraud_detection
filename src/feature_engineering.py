import pandas as pd

def create_features(df):
    # Convert the 'purchase_time' column to datetime format
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    df['transaction_frequency'] = df.groupby('user_id')['purchase_time'].transform('count')
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df

def process_fraud_data():
    df = pd.read_csv('data/processed/processed_Fraud_Data.csv')
    df = create_features(df)
    df.to_csv('data/processed/processed_Fraud_Data.csv', index=False)

if __name__ == "__main__":
    process_fraud_data()