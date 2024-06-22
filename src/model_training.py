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





# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import mlflow
# import mlflow.sklearn

# # Function to preprocess data
# def prepare_data(data, target_column):
#     # Parse datetime columns
#     data['signup_time'] = pd.to_datetime(data['signup_time'])
#     data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
#     # Extract features from datetime columns
#     data['signup_hour'] = data['signup_time'].dt.hour
#     data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
    
#     data['purchase_hour'] = data['purchase_time'].dt.hour
#     data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    
#     # Drop original datetime columns and target column
#     X = data.drop(columns=['signup_time', 'purchase_time', target_column])

#     y = data[target_column]
#     return train_test_split(X, y, test_size=0.3, random_state=42)

# # Load your dataset
# data = pd.read_csv('data/processed/processed_for_model.csv')

# # Prepare the data
# X_train, X_test, y_train, y_test = prepare_data(data, 'class')

# # Define models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier(),
#     "MLP": MLPClassifier()
# }

# # Function to train and evaluate model
# def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{model_name} Accuracy: {accuracy}")
#     print(classification_report(y_test, y_pred))
#     return accuracy

# # Set the experiment
# mlflow.set_experiment("Fraud Detection Models")

# # Train and evaluate models
# for model_name, model in models.items():
#     with mlflow.start_run(run_name=model_name):
#         print(f"Training {model_name}")
#         accuracy = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.sklearn.log_model(model, model_name)





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE
# import mlflow
# import mlflow.sklearn
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN

# def prepare_data(data, target_column):
#     data['signup_time'] = pd.to_datetime(data['signup_time'])
#     data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
#     data['signup_hour'] = data['signup_time'].dt.hour
#     data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
    
#     data['purchase_hour'] = data['purchase_time'].dt.hour
#     data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    
#     X = data.drop(columns=['signup_time', 'purchase_time', target_column])
#     y = data[target_column]
#     return train_test_split(X, y, test_size=0.3, random_state=42)

# data = pd.read_csv('data/processed/processed_for_model.csv')
# X_train, X_test, y_train, y_test = prepare_data(data, 'class')

# # Apply SMOTE to balance the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Compute class weights
# classes = np.array([0, 1])
# class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# # Define models with class weights where applicable
# models = {
#     "Logistic Regression": LogisticRegression(class_weight='balanced'),
#     "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
#     "Random Forest": RandomForestClassifier(class_weight='balanced'),
#     "Gradient Boosting": GradientBoostingClassifier(),
#     "MLP": MLPClassifier()
# }

# def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{model_name} Accuracy: {accuracy}")
#     print(classification_report(y_test, y_pred))
#     return accuracy

# mlflow.set_experiment("Fraud Detection Models")

# for model_name, model in models.items():
#     with mlflow.start_run(run_name=model_name):
#         print(f"Training {model_name}")
#         accuracy = train_and_evaluate_model(model, X_train_smote, y_train_smote, X_test, y_test, model_name)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.sklearn.log_model(model, model_name)

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

# X_train_nn = X_train_smote.values.reshape((X_train_smote.shape[0], X_train_smote.shape[1], 1))
# X_test_nn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# nn_models = {
#     "CNN": create_cnn_model(X_train_nn.shape[1:]),
#     "LSTM": create_lstm_model(X_train_nn.shape[1:]),
#     "RNN": create_rnn_model(X_train_nn.shape[1:])
# }

# for model_name, model in nn_models.items():
#     with mlflow.start_run(run_name=model_name):
#         print(f"Training {model_name} on fraud_data")
#         model.fit(X_train_nn, y_train_smote, epochs=5, batch_size=64, validation_data=(X_test_nn, y_test), class_weight=class_weights_dict, verbose=1)
#         accuracy = model.evaluate(X_test_nn, y_test, verbose=0)[1]
#         mlflow.log_metric("accuracy", accuracy)
        
      






















# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE
# import mlflow
# import mlflow.sklearn
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN

# # Function to prepare data
# def prepare_data(data, target_column):
#     data['signup_time'] = pd.to_datetime(data['signup_time'])
#     data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
#     data['signup_hour'] = data['signup_time'].dt.hour
#     data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
    
#     data['purchase_hour'] = data['purchase_time'].dt.hour
#     data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    
#     X = data.drop(columns=['signup_time', 'purchase_time', target_column])
#     y = data[target_column]
#     return train_test_split(X, y, test_size=0.3, random_state=42)

# # Load and prepare data
# data = pd.read_csv('../data/processed/processed_for_model.csv')
# X_train, X_test, y_train, y_test = prepare_data(data, 'class')

# # Apply SMOTE to balance the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Compute class weights
# classes = np.unique(y_train)
# class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights_dict = dict(zip(classes, class_weights))

# # Initialize MLflow experiment
# mlflow.set_experiment("Fraud Detection Models")

# # Define models with class weights where applicable
# models = {
#     "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
#     "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
#     "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42),
#     "MLP": MLPClassifier(random_state=42)
# }

# # Function to train and evaluate models
# def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
#     with mlflow.start_run(run_name=model_name):
#         print(f"Training {model_name}")
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f"{model_name} Accuracy: {accuracy}")
#         print(classification_report(y_test, y_pred))
#         mlflow.log_param("model_name", model_name)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.sklearn.log_model(model, model_name)

# # Train and evaluate models
# for model_name, model in models.items():
#     train_and_evaluate_model(model, X_train_smote, y_train_smote, X_test, y_test, model_name)

# # Function to create and train neural network models
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

# # Train and evaluate neural network models
# def train_neural_network(model, model_name, X_train, y_train, X_test, y_test, class_weights_dict):
#     X_train_nn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshape for Conv1D, LSTM, and RNN input
#     X_test_nn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))      # Reshape for Conv1D, LSTM, and RNN input

#     with mlflow.start_run(run_name=model_name):
#         print(f"Training {model_name}")
#         model.fit(X_train_nn, y_train, epochs=5, batch_size=64, validation_data=(X_test_nn, y_test), class_weight=class_weights_dict, verbose=1)
#         _, accuracy = model.evaluate(X_test_nn, y_test, verbose=0)
#         mlflow.log_metric("accuracy", accuracy)

# # Define neural network models
# nn_models = {
#     "CNN": create_cnn_model((X_train.shape[1], 1)),
#     "LSTM": create_lstm_model((X_train.shape[1], 1)),
#     "RNN": create_rnn_model((X_train.shape[1], 1))
# }

# # Train and evaluate neural network models
# for model_name, model in nn_models.items():
#     train_neural_network(model, model_name, X_train_smote, y_train_smote, X_test, y_test, class_weights_dict)







import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, SimpleRNN

def prepare_data(data, target_column):
    if 'signup_time' in data.columns and 'purchase_time' in data.columns:
        data['signup_time'] = pd.to_datetime(data['signup_time'])
        data['purchase_time'] = pd.to_datetime(data['purchase_time'])
        
        data['signup_hour'] = data['signup_time'].dt.hour
        data['signup_day_of_week'] = data['signup_time'].dt.dayofweek
        
        data['purchase_hour'] = data['purchase_time'].dt.hour
        data['purchase_day_of_week'] = data['purchase_time'].dt.dayofweek
    
    X = data.drop(columns=['signup_time', 'purchase_time', target_column], errors='ignore')
    y = data[target_column]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    return accuracy

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_log_nn_model(model, X_train_nn, y_train_smote, X_test_nn, y_test, model_name, class_weights_dict):
    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name} on fraud_data")
        model.fit(X_train_nn, y_train_smote, epochs=5, batch_size=64, validation_data=(X_test_nn, y_test), class_weight=class_weights_dict, verbose=1)
        accuracy = model.evaluate(X_test_nn, y_test, verbose=0)[1]
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.keras.log_model(model, model_name)

# Load datasets
creditcard_data = pd.read_csv('data/processed/processed_creditcard.csv')
fraud_data = pd.read_csv('data/processed/processed_for_model.csv')

# Prepare data
X_train_cc, X_test_cc, y_train_cc, y_test_cc = prepare_data(creditcard_data, 'Class')
X_train_fd, X_test_fd, y_train_fd, y_test_fd = prepare_data(fraud_data, 'class')

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_fd_smote, y_train_fd_smote = smote.fit_resample(X_train_fd, y_train_fd)

# Compute class weights
classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_fd)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Define models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier()
}

mlflow.set_experiment("Fraud Detection Models")

pip_requirements = ['scikit-learn==1.5.0', 'cloudpickle==3.0.0']

# Train and log models for credit card data
for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name} (creditcard)"):
        print(f"Training {model_name} on creditcard data")
        accuracy = train_and_evaluate_model(model, X_train_cc, y_train_cc, X_test_cc, y_test_cc, model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name, pip_requirements=pip_requirements)

# Train and log models for fraud data
for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name} (fraud_data)"):
        print(f"Training {model_name} on fraud data")
        accuracy = train_and_evaluate_model(model, X_train_fd_smote, y_train_fd_smote, X_test_fd, y_test_fd, model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name, pip_requirements=pip_requirements)

# Prepare data for neural networks
X_train_fd_nn = X_train_fd_smote.values.reshape((X_train_fd_smote.shape[0], X_train_fd_smote.shape[1], 1))
X_test_fd_nn = X_test_fd.values.reshape((X_test_fd.shape[0], X_test_fd.shape[1], 1))

# Define neural network models
nn_models = {
    "CNN": create_cnn_model(X_train_fd_nn.shape[1:]),
    "LSTM": create_lstm_model(X_train_fd_nn.shape[1:]),
    "RNN": create_rnn_model(X_train_fd_nn.shape[1:])
}

# Train and log neural network models for fraud data
for model_name, model in nn_models.items():
    train_and_log_nn_model(model, X_train_fd_nn, y_train_fd_smote, X_test_fd_nn, y_test_fd, model_name, class_weights_dict)
