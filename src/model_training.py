# import os 
# import joblib  
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report
# import mlflow
# import mlflow.sklearn

# # Define models to train
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier(),
#     "MLP": MLPClassifier(max_iter=1000)
# }

# def train_model(model_name, model, X_train, y_train, X_test, y_test):
#     with mlflow.start_run(run_name=model_name):
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
        
#         mlflow.log_params(model.get_params())
#         mlflow.log_metric("accuracy", model.score(X_test, y_test))
#         mlflow.sklearn.log_model(model, model_name)
        
#         # Save the trained model as a .pkl file
#         model_dir = "models"
#         if not os.path.exists(model_dir):
#             os.makedirs(model_dir)
#         model_path = os.path.join(model_dir, f"{model_name}.pkl")
#         joblib.dump(model, model_path)
        
#         print(f"Classification Report for {model_name}:\n")
#         print(classification_report(y_test, y_pred))

# def main():
#     # Load fraud data
#     fraud_df = pd.read_csv('data/processed/processed_Fraud_Data.csv')
#     X_fraud = fraud_df.drop('class', axis=1)
#     y_fraud = fraud_df['class']
    
#     # Load credit card data
#     credit_df = pd.read_csv('data/processed/processed_creditcard.csv')
#     X_credit = credit_df.drop('Class', axis=1)
#     y_credit = credit_df['Class']
    
#     # Train-Test Split for fraud data
#     X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
    
#     # Train-Test Split for credit card data
#     X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
    
#     # Train and evaluate each model on fraud data
#     print("Training models on fraud data")
#     for model_name, model in models.items():
#         train_model(f"{model_name}_fraud", model, X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud)
    
#     # Train and evaluate each model on credit card data
#     print("Training models on credit card data")
#     for model_name, model in models.items():
#         train_model(f"{model_name}_credit", model, X_train_credit, y_train_credit, X_test_credit, y_test_credit)

# if __name__ == "__main__":
#     main()









import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(f"Classification Report for {model_name}:\n")
    print(report)
    
    mlflow.log_param("model_name", model_name)
    mlflow.sklearn.log_model(model, model_name)
    
    with open(f"{model_name}_classification_report.txt", "w") as f:
        f.write(report)
    
    mlflow.log_artifact(f"{model_name}_classification_report.txt")

def main():
    print("Training models on fraud data")
    fraud_data = pd.read_csv("data/processed/processed_Fraud_Data.csv")
    X_fraud = fraud_data.drop("class", axis=1)
    y_fraud = fraud_data["class"]

    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
    
    models = [
        ("Logistic_Regression_fraud", LogisticRegression(max_iter=10000)),
        ("Decision_Tree_fraud", DecisionTreeClassifier()),
        ("Random_Forest_fraud", RandomForestClassifier()),
        ("Gradient_Boosting_fraud", GradientBoostingClassifier()),
        ("MLP_fraud", MLPClassifier(max_iter=10000))
    ]

    for model_name, model in models:
        with mlflow.start_run(run_name=model_name):
            train_and_evaluate_model(model, X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud, model_name)

    print("Training models on credit card data")
    credit_data = pd.read_csv("data/processed/processed_creditcard.csv")
    X_credit = credit_data.drop("Class", axis=1)
    y_credit = credit_data["Class"]

    X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)

    models = [
        ("Logistic_Regression_credit", LogisticRegression(max_iter=10000)),
        ("Decision_Tree_credit", DecisionTreeClassifier()),
        ("Random_Forest_credit", RandomForestClassifier()),
        ("Gradient_Boosting_credit", GradientBoostingClassifier()),
        ("MLP_credit", MLPClassifier(max_iter=10000))
    ]

    for model_name, model in models:
        with mlflow.start_run(run_name=model_name):
            train_and_evaluate_model(model, X_train_credit, y_train_credit, X_test_credit, y_test_credit, model_name)

if __name__ == "__main__":
    main()
