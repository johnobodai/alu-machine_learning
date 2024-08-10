import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import preprocess_dataset
import os

def load_model(model_path='models/neural_network_model.joblib'):
    return joblib.load(model_path)



def train_model(X_train, y_train):
    model = RandomForestClassifier()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def retrain_model():
    data = pd.read_csv('data/train/dataset.csv')
    X, y, scaler = preprocess_dataset(data)  # Apply preprocessing

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)

    # Save the model and scaler
    joblib.dump(model, 'models/neural_network_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f'Model retrained with accuracy: {accuracy}')

    return model, scaler

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
def predict(model, data):
    prediction = model.predict(data)
    return prediction
