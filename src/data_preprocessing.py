# import os
# import pandas as pd

# # Directory Paths
# data_dir = "./data/raw/"
# output_dir = "./data/processed/"

# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Files
# datasets = {
#     "fraud_data": "Fraud_Data.csv",
#     "ip_data": "IpAddress_to_Country.csv",
#     "creditcard_data": "creditcard.csv"
# }

# # Function to display missing values
# def display_missing_values(df, dataset_name):
#     missing_values = df.isnull().sum()
#     print(f"Missing values in {dataset_name}:")
#     print(missing_values[missing_values > 0])

# # Function to handle missing values
# def handle_missing_values(df):
#     for column in df.columns:
#         if df[column].dtype == "object":  # For non-numeric columns
#             df[column].fillna(df[column].mode()[0], inplace=True)  # Fill with mode
#         else:
#             df[column].fillna(df[column].median(), inplace=True)  # Fill with median
#     return df

# # Function to preprocess data
# def preprocess_data(file_path, dataset_name):
#     df = pd.read_csv(file_path)
#     display_missing_values(df, dataset_name)
#     df = handle_missing_values(df)
#     return df

# # Process each dataset
# for dataset_name, filename in datasets.items():
#     file_path = os.path.join(data_dir, filename)
#     if os.path.exists(file_path):
#         processed_df = preprocess_data(file_path, dataset_name)
#         processed_df.to_csv(os.path.join(output_dir, f"processed_{filename}"), index=False)
#     else:
#         print(f"File {filename} not found in {data_dir}")


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# import matplotlib.pyplot as plt
# import seaborn as sns

# class DataProcessor:
#     def __init__(self):
#         self.fraud_data = None
#         self.ip_data = None
#         self.merged_data = None
        
#     def load_data(self):
#         self.fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
#         self.ip_data = pd.read_csv('data/raw/IpAddress_to_Country.csv')
        
#     def handle_missing_values(self):
#         # Identify numeric columns
#         numeric_cols = self.fraud_data.select_dtypes(include=[np.number]).columns.tolist()
        
#         # Fill missing values in numeric columns with median
#         self.fraud_data[numeric_cols] = self.fraud_data[numeric_cols].fillna(self.fraud_data[numeric_cols].median())        
#     def merge_datasets(self):
#         # Convert IP addresses to integer format and merge datasets
#         self.fraud_data['ip_address'] = self.fraud_data['ip_address'].astype(int)
        
#         # Convert IP address columns in ip_data if they are not already converted
#         if not pd.api.types.is_integer_dtype(self.ip_data['lower_bound_ip_address']):
#             self.ip_data['lower_bound_ip_address'] = self.ip_data['lower_bound_ip_address'].apply(lambda x: int(''.join(str(x).split('.'))) if pd.notna(x) else np.nan)
        
#         if not pd.api.types.is_integer_dtype(self.ip_data['upper_bound_ip_address']):
#             self.ip_data['upper_bound_ip_address'] = self.ip_data['upper_bound_ip_address'].apply(lambda x: int(''.join(str(x).split('.'))) if pd.notna(x) else np.nan)
        
#         self.merged_data = pd.merge(self.fraud_data, self.ip_data, how='left', left_on='ip_address', right_on='lower_bound_ip_address')
        
#     def feature_engineering(self):
#         # Example: Add time-based features
#         self.merged_data['time_diff'] = (pd.to_datetime(self.merged_data['purchase_time']) - pd.to_datetime(self.merged_data['signup_time'])).dt.total_seconds()
#         self.merged_data['hour_of_day'] = pd.to_datetime(self.merged_data['purchase_time']).dt.hour
#         self.merged_data['day_of_week'] = pd.to_datetime(self.merged_data['purchase_time']).dt.dayofweek
        
#     def normalize_and_scale(self):
#         # Example: Normalize and scale features
#         scaler = StandardScaler()
#         numerical_features = ['purchase_value', 'age', 'time_diff']
#         self.merged_data[numerical_features] = scaler.fit_transform(self.merged_data[numerical_features])
        
#     def encode_categorical_features(self):
#         # Example: Encode categorical features
#         categorical_features = ['source', 'browser', 'sex']
#         encoder = OneHotEncoder()
#         encoded_features = pd.DataFrame(encoder.fit_transform(self.merged_data[categorical_features]).toarray(), columns=encoder.get_feature_names_out(categorical_features))
#         self.merged_data = pd.concat([self.merged_data, encoded_features], axis=1)
#         self.merged_data = self.merged_data.drop(categorical_features, axis=1)
        
#     def save_processed_data(self, filename):
#         self.merged_data.to_csv(filename, index=False)
        
#     def visualize_data(self):
#         # Univariate analysis
#         plt.figure(figsize=(12, 6))
#         sns.histplot(self.merged_data['purchase_value'].dropna(), bins=30, kde=True)
#         plt.title('Distribution of Purchase Value')
#         plt.xlabel('Purchase Value')
#         plt.ylabel('Frequency')
#         plt.show()

#         plt.figure(figsize=(12, 6))
#         sns.countplot(x='class', data=self.merged_data)
#         plt.title('Class Distribution')
#         plt.show()

#         # Bivariate analysis
#         plt.figure(figsize=(12, 6))
#         sns.boxplot(x='class', y='purchase_value', data=self.merged_data)
#         plt.title('Purchase Value by Class')
#         plt.show()

#         plt.figure(figsize=(12, 6))
#         sns.scatterplot(x='purchase_value', y='age', hue='class', data=self.merged_data)
#         plt.title('Purchase Value vs. Age')
#         plt.show()

# # Example usage:
# if __name__ == "__main__":
#     processor = DataProcessor()
#     processor.load_data()
#     processor.handle_missing_values()
#     processor.merge_datasets()
#     processor.feature_engineering()
#     processor.normalize_and_scale()
#     processor.encode_categorical_features()
#     processor.save_processed_data('data/processed/processed_data.csv')
#     processor.visualize_data()


import pandas as pd
import numpy as np
from datetime import datetime

# Load raw data
fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
ip_data = pd.read_csv('data/raw/IpAddress_to_Country.csv')
credit_data = pd.read_csv('data/raw/creditcard.csv')

# Handle Missing Values
def handle_missing_values(data):
    # Check for missing values
    null_counts = data.isnull().sum()
    print("Null value counts:")
    print(null_counts)
    
    # Handle missing values (example: drop all rows with any missing values)
    data.dropna(inplace=True)
    return data

# Data Cleaning
def data_cleaning(data):
    # Remove duplicates
    data.drop_duplicates(inplace=True)
    
    # Correct data types
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    data['ip_address'] = data['ip_address'].astype(int)  # Assuming IP addresses should be integers
    return data

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    # Univariate analysis
    print("Summary statistics:")
    print(data.describe())
    
    # Bivariate analysis (excluding non-numeric columns)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_columns].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Merge with IP data for geolocation analysis
    data = data.merge(ip_data, how='left', left_on='ip_address', right_on='lower_bound_ip_address')
    
    return data

# Feature Engineering
def feature_engineering(data):
    # Transaction frequency and velocity
    data['transaction_count'] = data.groupby('user_id')['user_id'].transform('count')
    data['transaction_velocity'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds() / data['transaction_count']
    
    # Time-based features
    data['hour_of_day'] = data['purchase_time'].dt.hour
    data['day_of_week'] = data['purchase_time'].dt.dayofweek
    
    # Normalization and Scaling (example: min-max scaling)
    data['purchase_value_scaled'] = (data['purchase_value'] - data['purchase_value'].min()) / (data['purchase_value'].max() - data['purchase_value'].min())
    data['age_scaled'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())
    
    # Encode categorical features (example: one-hot encoding)
    data = pd.get_dummies(data, columns=['source', 'browser', 'sex', 'country'])
    
    return data

# Execute the preprocessing pipeline
def preprocess_data(fraud_data, ip_data):
    # Handle missing values
    fraud_data = handle_missing_values(fraud_data)
    
    # Data cleaning
    fraud_data = data_cleaning(fraud_data)
    
    # EDA and feature engineering
    fraud_data = perform_eda(fraud_data)
    fraud_data = feature_engineering(fraud_data)
    
    return fraud_data

# Preprocess the fraud data
processed_fraud_data = preprocess_data(fraud_data, ip_data)

# Save processed data
processed_fraud_data.to_csv('data/processed/processed_Fraud_Data.csv', index=False)

# Optionally, you can save the processed IP data as well
ip_data.to_csv('data/processed/processed_IpAddress_to_Country.csv', index=False)

# Display processed data summary
print("Processed Fraud Data Summary:")
print(processed_fraud_data.info())
