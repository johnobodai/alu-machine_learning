import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_dataset(df):
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

def preprocess_input(data, scaler=None):
    if scaler is None:
        try:
            scaler = joblib.load('models/scaler.joblib')
        except FileNotFoundError:
            print("Scaler not found. Using default StandardScaler.")
            scaler = StandardScaler()
            scaler.fit(data)
    
    # Ensure data has the correct number of features
    expected_features = scaler.n_features_in_
    if data.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, but got {data.shape[1]} features.")
    
    return scaler.transform(data)

def clean_data(data):
    # Implement data cleaning steps
    return data

def scale_data(data):
    # Implement data scaling if needed
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

