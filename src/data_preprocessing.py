import os
import pandas as pd

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
