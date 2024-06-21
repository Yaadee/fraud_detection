# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import mlflow
# import mlflow.sklearn
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN

# def prepare_data(data, target_column):
#     X = data.drop(columns=[target_column])
#     y = data[target_column]
#     return train_test_split(X, y, test_size=0.3, random_state=42)

# # Load the data
# fraud_data = pd.read_csv('data/processed/processed_Fraud_Data_with_Features.csv')
# creditcard_data = pd.read_csv('data/processed/processed_creditcard.csv')

# # Prepare the data
# X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = prepare_data(fraud_data, 'class')
# X_cc_train, X_cc_test, y_cc_train, y_cc_test = prepare_data(creditcard_data, 'Class')

# def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{model_name} Accuracy: {accuracy}")
#     print(classification_report(y_test, y_pred))
#     return accuracy

# # Set the experiment
# mlflow.set_experiment("Fraud Detection Models")

# # Define models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier(),
#     "MLP": MLPClassifier()
# }

# # Train and evaluate models
# for model_name, model in models.items():
#     with mlflow.start_run(run_name=model_name):
#         print(f"Training {model_name} on fraud_data")
#         fraud_accuracy = train_and_evaluate_model(model, X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test, model_name)
#         mlflow.log_metric("fraud_accuracy", fraud_accuracy)
        
#         print(f"Training {model_name} on creditcard_data")
#         cc_accuracy = train_and_evaluate_model(model, X_cc_train, y_cc_train, X_cc_test, y_cc_test, model_name)
#         mlflow.log_metric("creditcard_accuracy", cc_accuracy)
        
#         mlflow.sklearn.log_model(model, model_name)

# # Neural Network models
# def create_cnn_model(input_shape):
#     model = Sequential([
#         Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
#         MaxPooling1D(pool_size=2),
#         Flatten(),
#         Dense(100, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def create_lstm_model(input_shape):
#     model = Sequential([
#         LSTM(50, input_shape=input_shape),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def create_rnn_model(input_shape):
#     model = Sequential([
#         SimpleRNN(50, input_shape=input_shape),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Reshape data for neural networks
# X_fraud_train_nn = X_fraud_train.values.reshape((X_fraud_train.shape[0], X_fraud_train.shape[1], 1))
# X_fraud_test_nn = X_fraud_test.values.reshape((X_fraud_test.shape[0], X_fraud_test.shape[1], 1))

# X_cc_train_nn = X_cc_train.values.reshape((X_cc_train.shape[0], X_cc_train.shape[1], 1))
# X_cc_test_nn = X_cc_test.values.reshape((X_cc_test.shape[0], X_cc_test.shape[1], 1))

# # Define neural network models
# nn_models = {
#     "CNN": create_cnn_model(X_fraud_train_nn.shape[1:]),
#     "LSTM": create_lstm_model(X_fraud_train_nn.shape[1:]),
#     "RNN": create_rnn_model(X_fraud_train_nn.shape[1:])
# }

# # Train and evaluate neural network models
# for model_name, model in nn_models.items():
#     with mlflow.start_run(run_name=model_name):
#         print(f"Training {model_name} on fraud_data")
#         model.fit(X_fraud_train_nn, y_fraud_train, epochs=5, batch_size=64, validation_data=(X_fraud_test_nn, y_fraud_test), verbose=1)
#         fraud_accuracy = model.evaluate(X_fraud_test_nn, y_fraud_test, verbose=0)[1]
#         mlflow.log_metric("fraud_accuracy", fraud_accuracy)
        
#         print(f"Training {model_name} on creditcard_data")
#         model.fit(X_cc_train_nn, y_cc_train, epochs=5, batch_size=64, validation_data=(X_cc_test_nn, y_cc_test), verbose=1)
#         cc_accuracy = model.evaluate(X_cc_test_nn, y_cc_test, verbose=0)[1]
#         mlflow.log_metric("creditcard_accuracy", cc_accuracy)
        
#         mlflow.keras.log_model(model, model_name)





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn

# Function to preprocess data
def prepare_data(data, target_column):
    # Parse datetime columns
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
    # Extract features from datetime columns
    data['signup_hour'] = data['signup_time'].dt.hour
    data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
    
    data['purchase_hour'] = data['purchase_time'].dt.hour
    data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    
    # Drop original datetime columns and target column
    X = data.drop(columns=['signup_time', 'purchase_time', target_column])

    y = data[target_column]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Load your dataset
data = pd.read_csv('data/processed/processed_Features.csv')

# Prepare the data
X_train, X_test, y_train, y_test = prepare_data(data, 'class')

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier()
}

# Function to train and evaluate model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    return accuracy

# Set the experiment
mlflow.set_experiment("Fraud Detection Models")

# Train and evaluate models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name}")
        accuracy = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name)
