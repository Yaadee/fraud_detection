import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineering(data):
    # Convert timestamps to datetime format
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
    # Calculate hour_of_day and day_of_week
    data['hour_of_day'] = data['purchase_time'].dt.hour
    data['day_of_week'] = data['purchase_time'].dt.dayofweek
    
    # Calculate transaction_count and transaction_velocity
    data['transaction_count'] = data.groupby('user_id')['user_id'].transform('count')
    data['transaction_velocity'] = data.groupby('user_id')['purchase_time'].transform(
        lambda x: (x.max() - x.min()).days + 1
    )
    
    # Label encode categorical features
    label_encoders = {}
    categorical_features = ['device_id', 'source', 'browser', 'sex', 'country']
    
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        data[feature] = label_encoders[feature].fit_transform(data[feature])
    
    return data

if __name__ == "__main__":
    merged_data_path = 'data/processed/merged_data.csv'
    merged_data = pd.read_csv(merged_data_path)
    
    feature_data = feature_engineering(merged_data)
    feature_data.to_csv('data/processed/processed_Fraud_Data.csv', index=False)

