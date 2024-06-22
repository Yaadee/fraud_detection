import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_creditcard_data(creditcard_file):
    creditcard_data = pd.read_csv(creditcard_file)
    X = creditcard_data.drop(columns=['Class'])
    y = creditcard_data['Class']
    return train_test_split(X, y, test_size=0.3, random_state=42)

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
