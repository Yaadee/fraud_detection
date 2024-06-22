import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# Directory Paths
data_dir = "./data/raw/"
output_dir = "./data/processed/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Files
datasets = {
    "fraud_data": "Fraud_Data.csv",
    "ip_data": "IpAddress_to_Country.csv",
    "creditcard_data": "creditcard.csv"
}

# Function to display missing values
def display_missing_values(df, dataset_name):
    missing_values = df.isnull().sum()
    print(f"Missing values in {dataset_name}:")
    print(missing_values[missing_values > 0])

# Function to handle missing values
def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype == "object":  # For non-numeric columns
            df[column].fillna(df[column].mode()[0], inplace=True)  # Fill with mode
        else:
            df[column].fillna(df[column].median(), inplace=True)  # Fill with median
    return df

# Function to preprocess data
def preprocess_data(file_path, dataset_name):
    df = pd.read_csv(file_path)
    display_missing_values(df, dataset_name)
    df = handle_missing_values(df)
    return df

# Process each dataset
for dataset_name, filename in datasets.items():
    file_path = os.path.join(data_dir, filename)
    if os.path.exists(file_path):
        processed_df = preprocess_data(file_path, dataset_name)
        processed_df.to_csv(os.path.join(output_dir, f"processed_{filename}"), index=False)
    else:
        print(f"File {filename} not found in {data_dir}")

# Function to prepare credit card data
def prepare_creditcard_data(creditcard_file):
    creditcard_data = pd.read_csv(creditcard_file)
    X = creditcard_data.drop(columns=['Class'])
    y = creditcard_data['Class']
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Function to prepare fraud data
def prepare_fraud_data(fraud_file, ip_to_country_file):
    fraud_data = pd.read_csv(fraud_file)
    ip_to_country = pd.read_csv(ip_to_country_file)
    
    # Function to merge IP address to country mapping
    def merge_ip_to_country(ip):
        try:
            return ip_to_country[(ip_to_country['lower_bound_ip_address'] <= ip) & 
                                 (ip_to_country['upper_bound_ip_address'] >= ip)]['country'].values[0]
        except IndexError:
            return 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(merge_ip_to_country)
    
    X = fraud_data.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'])
    y = fraud_data['class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    return X_train_smote, X_test, y_train_smote, y_test