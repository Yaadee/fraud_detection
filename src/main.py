from src.data_preprocessing import prepare_creditcard_data, prepare_fraud_data
from src.model_training import train_and_evaluate_creditcard_models, train_and_evaluate_fraud_detection_models
from src.feature_engineering import merge_ip_to_country, extract_time_features

# Paths to data files
creditcard_file = 'data/raw/creditcard.csv'
fraud_file = 'data/raw/Fraud_Data.csv'
ip_to_country_file = 'data/raw/IpAddress_to_Country.csv'

# Perform feature engineering
merge_ip_to_country(fraud_file, ip_to_country_file)
extract_time_features(creditcard_file)

# Prepare credit card data
X_train_credit, X_test_credit, y_train_credit, y_test_credit = prepare_creditcard_data('data/processed/processed_creditcard.csv')

# Prepare fraud detection data
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = prepare_fraud_data('data/processed/processed_Fraud_Data.csv', ip_to_country_file)

# Train and evaluate credit card models
train_and_evaluate_creditcard_models(X_train_credit, X_test_credit, y_train_credit, y_test_credit)

# Train and evaluate fraud detection models
class_weights_dict = {0: 1, 1: 1}  # Update with actual class weights
train_and_evaluate_fraud_detection_models(X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud, class_weights_dict)
