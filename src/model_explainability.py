
import pandas as pd
import shap
import mlflow

def explain_model(model_name, data_path, target_col):
    model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
    df = pd.read_csv(data_path)
    X = df.drop(target_col, axis=1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)

if __name__ == "__main__":
    explain_model("Random Forest_fraud", "data/processed/processed_Fraud_Data.csv", "class")
    explain_model("Random Forest_credit", "data/processed/processed_creditcard.csv", "Class")
