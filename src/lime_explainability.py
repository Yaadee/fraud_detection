import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load datasets
creditcard_data = pd.read_csv('data/processed/processed_creditcard.csv')
fraud_data = pd.read_csv('data/processed/processed_for_model.csv')

# Prepare data
def prepare_data(data, target_column):
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
    data['signup_hour'] = data['signup_time'].dt.hour
    data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
    
    data['purchase_hour'] = data['purchase_time'].dt.hour
    data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    
    X = data.drop(columns=['signup_time', 'purchase_time', target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.3, random_state=42)

X_train_cc, X_test_cc, y_train_cc, y_test_cc = prepare_data(creditcard_data, 'Class')
X_train_fd, X_test_fd, y_train_fd, y_test_fd = prepare_data(fraud_data, 'class')

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_fd_smote, y_train_fd_smote = smote.fit_resample(X_train_fd, y_train_fd)

# Load a model from MLflow
logged_model = 'runs:/<run_id>/model'
model = mlflow.sklearn.load_model(logged_model)

# Explain the model using LIME
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_fd_smote.values, feature_names=X_train_fd.columns, class_names=['Not Fraud', 'Fraud'], discretize_continuous=True)

i = 0
exp = explainer.explain_instance(X_test_fd.iloc[i], model.predict_proba, num_features=10)
exp.save_to_file('lime_explanation.html')

# Feature Importance Plot for a specific prediction
fig = exp.as_pyplot_figure()
plt.savefig('lime_feature_importance_plot.png')
plt.close()
