
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






# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# def load_data():
#     """
#     Loads the Fraud_Data.csv and IpAddress_to_Country.csv files.
    
#     Returns:
#         fraud_data (pandas.DataFrame): Fraud data.
#         ip_data (pandas.DataFrame): IP address to country mapping data.
#     """
#     fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
#     ip_data = pd.read_csv('data/raw/IpAddress_to_Country.csv')
#     return fraud_data, ip_data

# def handle_missing_values(data):
#     """
#     Imputes missing values in numerical columns with the median.
    
#     Args:
#         data (pandas.DataFrame): The input data.
        
#     Returns:
#         data (pandas.DataFrame): The data with missing values imputed.
#     """
#     imputer = SimpleImputer(strategy='median')
#     num_cols = data.select_dtypes(include=[np.number]).columns
#     data[num_cols] = imputer.fit_transform(data[num_cols])
#     return data

# def clean_data(data):
#     """
#     Cleans the data by removing duplicates and correcting data types.
    
#     Args:
#         data (pandas.DataFrame): The input data.
        
#     Returns:
#         data (pandas.DataFrame): The cleaned data.
#     """
#     # Remove duplicates
#     data = data.drop_duplicates()
    
#     # Correct data types
#     data['purchase_value'] = pd.to_numeric(data['purchase_value'], errors='coerce')
#     data['signup_time'] = pd.to_datetime(data['signup_time'], errors='coerce')
#     data['purchase_time'] = pd.to_datetime(data['purchase_time'], errors='coerce')
    
    
#     return data

# def exploratory_data_analysis(data):
#     """
#     Performs exploratory data analysis on the input data, including univariate and bivariate analysis.
    
#     Args:
#         data (pandas.DataFrame): The input data.
#     """
#     # Ensure 'purchase_value' is a numeric column
#     data['purchase_value'] = pd.to_numeric(data['purchase_value'], errors='coerce')
    
#     # Drop any rows with NaN values in 'purchase_value' after conversion
#     data = data.dropna(subset=['purchase_value'])
    
#     # Convert to numpy array for seaborn compatibility
#     purchase_value_array = data['purchase_value'].to_numpy()
    
#     # Univariate analysis
#     plt.figure(figsize=(12, 6))
#     sns.histplot(purchase_value_array, bins=30, kde=True)
#     plt.title('Distribution of Purchase Value')
#     plt.xlabel('Purchase Value')
#     plt.ylabel('Density')
#     plt.show()
    
#     # Bivariate analysis
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(x='class', y='purchase_value', data=data)
#     plt.title('Boxplot of Purchase Value by Class')
#     plt.xlabel('Class')
#     plt.ylabel('Purchase Value')
#     plt.show()

# def merge_datasets(fraud_data, ip_data):
#     """
#     Merges the fraud data and IP address to country mapping data based on the IP address.
    
#     Args:
#         fraud_data (pandas.DataFrame): The fraud data.
#         ip_data (pandas.DataFrame): The IP address to country mapping data.
        
#     Returns:
#         fraud_data (pandas.DataFrame): The merged data.
#     """
#     # Convert IP addresses to integer format
#     ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(np.int64)
#     ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(np.int64)
#     fraud_data['ipmatch_address'] = fraud_data['ip_address'].astype(np.int64)
    
#     # Initialize country column with NaN values
#     fraud_data['country'] = np.nan

#     # Merge datasets without introducing NaN values
#     for index, row in fraud_data.iterrows():
#         match = ip_data[(ip_data['lower_bound_ip_address'] <= row['ip_address']) & (ip_data['upper_bound_ip_address'] >= row['ip_address'])]
#         if not match.empty:
#             fraud_data.at[index, 'country'] = str(match['country'].values[0])
    
#     return fraud_data

# def feature_engineering(data):
#     """
#     Performs feature engineering on the input data, adding time-based features and transaction-related features.
    
#     Args:
#         data (pandas.DataFrame): The input data.
        
#     Returns:
#         data (pandas.DataFrame): The data with added features.
#     """
#     # Add time-based features
#     data['hour_of_day'] = data['purchase_time'].dt.hour
#     data['day_of_week'] = data['purchase_time'].dt.dayofweek
    
#     # Transaction frequency and velocity
#     data['transaction_count'] = data.groupby('user_id')['user_id'].transform('count')
#     data['transaction_velocity'] = data.groupby('user_id')['purchase_time'].transform(lambda x: (x.max() - x.min()).days + 1)
    
#     return data

# def normalize_and_scale(data):
#     """
#     Normalizes and scales the selected features in the input data.
    
#     Args:
#         data (pandas.DataFrame): The input data.
        
#     Returns:
#         data (pandas.DataFrame): The normalized and scaled data.
#     """
#     # Normalize and scale features
#     scaler = StandardScaler()
#     data[['purchase_value', 'hour_of_day', 'day_of_week']] = scaler.fit_transform(data[['purchase_value', 'hour_of_day', 'day_of_week']])
#     return data

# def encode_categorical_features(data):
#     """
#     Encodes the categorical features in the input data using one-hot encoding.
    
#     Args:
#         data (pandas.DataFrame): The input data.
        
#     Returns:
#         data (pandas.DataFrame): The data with encoded categorical features.
#     """
#     # Encode categorical features
#     data = pd.get_dummies(data, columns=['country'])
#     return data

# def save_processed_data(data, path):
#     """
#     Saves the processed data to a CSV file.
    
#     Args:
#         data (pandas.DataFrame): The processed data.
#         path (str): The path to save the data.
#     """
#     data.to_csv(path, index=False)

# # Load data
# fraud_data, ip_data = load_data()


# # Handle Missing Values
# fraud_data = handle_missing_values(fraud_data)

# # Data Cleaning
# fraud_data = clean_data(fraud_data)

# # Exploratory Data Analysis (EDA)
# exploratory_data_analysis(fraud_data)

# # Merge Datasets for Geolocation Analysis
# merged_data = merge_datasets(fraud_data, ip_data)

# # Feature Engineering
# merged_data = feature_engineering(merged_data)

# # Normalization and Scaling
# merged_data = normalize_and_scale(merged_data)

# # Encode Categorical Features
# final_data = encode_categorical_features(merged_data)

# # Save the cleaned and processed data
# save_processed_data(final_data, 'data/processed/merged_data.csv')