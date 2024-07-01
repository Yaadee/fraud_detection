import pandas as pd
import numpy as np
import shap
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load datasets
creditcard_data = pd.read_csv('data/processed/processed_creditcard.csv')
fraud_data = pd.read_csv('data/processed/processed_Fraud_Data.csv')

# Prepare data
def prepare_creditcard_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def prepare_fraud_data(data, target_column):
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
    data['signup_hour'] = data['signup_time'].dt.hour
    data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
    
    data['purchase_hour'] = data['purchase_time'].dt.hour
    data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    
    X = data.drop(columns=['signup_time', 'purchase_time', target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.3, random_state=42)

X_train_cc, X_test_cc, y_train_cc, y_test_cc = prepare_creditcard_data(creditcard_data, 'Class')
X_train_fd, X_test_fd, y_train_fd, y_test_fd = prepare_fraud_data(fraud_data, 'class')

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_fd_smote, y_train_fd_smote = smote.fit_resample(X_train_fd, y_train_fd)

# Load a model from MLflow
logged_model = 'runs:/<run_id>/model'
model = mlflow.sklearn.load_model(logged_model)

# Explain the model using SHAP
explainer = shap.Explainer(model, X_train_fd_smote)
shap_values = explainer(X_test_fd)

# Summary Plot
shap.summary_plot(shap_values, X_test_fd, show=False)
plt.savefig('shap_summary_plot.png')
plt.close()

# Force Plot for a single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test_fd.iloc[0, :], matplotlib=True)
plt.savefig('shap_force_plot.png')
plt.close()

# Dependence Plot
shap.dependence_plot('feature_name', shap_values, X_test_fd, show=False)
plt.savefig('shap_dependence_plot.png')
plt.close()