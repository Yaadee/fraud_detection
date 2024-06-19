# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# # Load data
# def load_data():
#     fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
#     ip_data = pd.read_csv('data/raw/IpAddress_to_Country.csv')
#     return fraud_data, ip_data

# # Handle Missing Values
# def handle_missing_values(data):
#     imputer = SimpleImputer(strategy='median')
#     num_cols = data.select_dtypes(include=[np.number]).columns
#     data[num_cols] = imputer.fit_transform(data[num_cols])
#     return data

# # Data Cleaning
# def clean_data(data):
#     data = data.drop_duplicates()
#     data['purchase_value'] = pd.to_numeric(data['purchase_value'], errors='coerce')
#     data['signup_time'] = pd.to_datetime(data['signup_time'], errors='coerce')
#     data['purchase_time'] = pd.to_datetime(data['purchase_time'], errors='coerce')
#     return data

# # Exploratory Data Analysis (EDA)
# def exploratory_data_analysis(data):
#     plt.figure(figsize=(12, 6))
#     sns.histplot(data=data, x='purchase_value', bins=30, kde=True)
#     plt.title('Distribution of Purchase Value')
#     plt.xlabel('Purchase Value')
#     plt.ylabel('Density')
#     plt.show()
    
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(x='class', y='purchase_value', data=data)
#     plt.title('Boxplot of Purchase Value by Class')
#     plt.xlabel('Class')
#     plt.ylabel('Purchase Value')
#     plt.show()
    
#     plt.figure(figsize=(12, 6))
#     sns.countplot(x='class', data=data)
#     plt.title('Distribution of Class')
#     plt.xlabel('Class')
#     plt.ylabel('Count')
#     plt.show()
    
#     plt.figure(figsize=(12, 6))
#     sns.countplot(x='source', hue='class', data=data)
#     plt.title('Source vs Class')
#     plt.xlabel('Source')
#     plt.ylabel('Count')
#     plt.show()

#     plt.figure(figsize=(12, 6))
#     sns.countplot(x='browser', hue='class', data=data)
#     plt.title('Browser vs Class')
#     plt.xlabel('Browser')
#     plt.ylabel('Count')
#     plt.show()
    
#     plt.figure(figsize=(12, 6))
#     sns.countplot(x='sex', hue='class', data=data)
#     plt.title('Sex vs Class')
#     plt.xlabel('Sex')
#     plt.ylabel('Count')
#     plt.show()

# # Merge Datasets for Geolocation Analysis
# def merge_datasets(fraud_data, ip_data):
#     ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(np.int64)
#     ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(np.int64)
#     fraud_data['ip_address'] = fraud_data['ip_address'].astype(np.int64)
    
#     # Merge datasets without introducing NaN values
#     fraud_data['country'] = fraud_data.apply(
#         lambda row: ip_data.loc[
#             (ip_data['lower_bound_ip_address'] <= row['ip_address']) &
#             (ip_data['upper_bound_ip_address'] >= row['ip_address']), 
#             'country'
#         ].values[0] if not ip_data.loc[
#             (ip_data['lower_bound_ip_address'] <= row['ip_address']) &
#             (ip_data['upper_bound_ip_address'] >= row['ip_address'])
#         ].empty else np.nan, axis=1
#     )
    
#     return fraud_data

# # Feature Engineering
# def feature_engineering(data):
#     data['hour_of_day'] = data['purchase_time'].dt.hour
#     data['day_of_week'] = data['purchase_time'].dt.dayofweek
#     data['transaction_count'] = data.groupby('user_id')['user_id'].transform('count')
#     data['transaction_velocity'] = data.groupby('user_id')['purchase_time'].transform(lambda x: (x.max() - x.min()).days + 1)
#     return data

# # Normalization and Scaling
# def normalize_and_scale(data):
#     scaler = StandardScaler()
#     data[['purchase_value', 'hour_of_day', 'day_of_week']] = scaler.fit_transform(data[['purchase_value', 'hour_of_day', 'day_of_week']])
#     return data

# # Encode Categorical Features
# def encode_categorical_features(data):
#     data = pd.get_dummies(data, columns=['country'])
#     return data

# # Save processed data
# def save_processed_data(data, path):
#     data.to_csv(path, index=False)

# # Main Execution
# if __name__ == "__main__":
#     fraud_data, ip_data = load_data()
#     fraud_data = handle_missing_values(fraud_data)
#     fraud_data = clean_data(fraud_data)
#     exploratory_data_analysis(fraud_data)
#     merged_data = merge_datasets(fraud_data, ip_data)
#     merged_data = feature_engineering(merged_data)
#     merged_data = normalize_and_scale(merged_data)
#     final_data = encode_categorical_features(merged_data)
#     save_processed_data(final_data, 'data/processed/processed_Data.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
def load_data():
    fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
    ip_data = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    return fraud_data, ip_data

# Handle Missing Values
def handle_missing_values(data):
    imputer = SimpleImputer(strategy='median')
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = imputer.fit_transform(data[num_cols])
    return data

# Data Cleaning
def clean_data(data):
    data = data.drop_duplicates()
    data['purchase_value'] = pd.to_numeric(data['purchase_value'], errors='coerce')
    data['signup_time'] = pd.to_datetime(data['signup_time'], errors='coerce')
    data['purchase_time'] = pd.to_datetime(data['purchase_time'], errors='coerce')
    return data

# Exploratory Data Analysis (EDA)
def exploratory_data_analysis(data):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='purchase_value', bins=30, kde=True)
    plt.title('Distribution of Purchase Value')
    plt.xlabel('Purchase Value')
    plt.ylabel('Density')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='class', y='purchase_value', data=data)
    plt.title('Boxplot of Purchase Value by Class')
    plt.xlabel('Class')
    plt.ylabel('Purchase Value')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='class', data=data)
    plt.title('Distribution of Class')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='source', hue='class', data=data)
    plt.title('Source vs Class')
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='browser', hue='class', data=data)
    plt.title('Browser vs Class')
    plt.xlabel('Browser')
    plt.ylabel('Count')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='sex', hue='class', data=data)
    plt.title('Sex vs Class')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.show()
    
    # Time series analysis of purchases
    plt.figure(figsize=(12, 6))
    data.set_index('purchase_time').resample('D').size().plot()
    plt.title('Number of Purchases per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Purchases')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    data[data['class'] == 1].set_index('purchase_time').resample('D').size().plot()
    plt.title('Number of Fraudulent Purchases per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Fraudulent Purchases')
    plt.show()

# Merge Datasets for Geolocation Analysis
def merge_datasets(fraud_data, ip_data):
    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(np.int64)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(np.int64)
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(np.int64)
    
    # Merge datasets without introducing NaN values
    fraud_data['country'] = fraud_data.apply(
        lambda row: ip_data.loc[
            (ip_data['lower_bound_ip_address'] <= row['ip_address']) &
            (ip_data['upper_bound_ip_address'] >= row['ip_address']), 
            'country'
        ].values[0] if not ip_data.loc[
            (ip_data['lower_bound_ip_address'] <= row['ip_address']) &
            (ip_data['upper_bound_ip_address'] >= row['ip_address'])
        ].empty else np.nan, axis=1
    )
    
    return fraud_data

# Feature Engineering
def feature_engineering(data):
    data['hour_of_day'] = data['purchase_time'].dt.hour
    data['day_of_week'] = data['purchase_time'].dt.dayofweek
    data['transaction_count'] = data.groupby('user_id')['user_id'].transform('count')
    data['transaction_velocity'] = data.groupby('user_id')['purchase_time'].transform(lambda x: (x.max() - x.min()).days + 1)
    return data

# Normalization and Scaling
def normalize_and_scale(data):
    scaler = StandardScaler()
    data[['purchase_value', 'hour_of_day', 'day_of_week']] = scaler.fit_transform(data[['purchase_value', 'hour_of_day', 'day_of_week']])
    return data

# Encode Categorical Features
def encode_categorical_features(data):
    data = pd.get_dummies(data, columns=['country'])
    return data

# Save processed data
def save_processed_data(data, path):
    data.to_csv(path, index=False)

# Main Execution
if __name__ == "__main__":
    fraud_data, ip_data = load_data()
    fraud_data = handle_missing_values(fraud_data)
    fraud_data = clean_data(fraud_data)
    exploratory_data_analysis(fraud_data)
    merged_data = merge_datasets(fraud_data, ip_data)
    merged_data = feature_engineering(merged_data)
    merged_data = normalize_and_scale(merged_data)
    final_data = encode_categorical_features(merged_data)
    save_processed_data(final_data, 'data/processed/processed_Data.csv')
