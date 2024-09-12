#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data(file_prefix):
    """
    Load the preprocessed data.
    
    Args:
        file_prefix (str): Prefix for the preprocessed data files.
    
    Returns:
        np.array: X data for model input.
        np.array: y data for model output.
    """
    X = np.load(f'{file_prefix}_X.npy')
    y = np.load(f'{file_prefix}_y.npy')
    return X, y

def load_model(file_prefix):
    """
    Load the trained model.
    
    Args:
        file_prefix (str): Prefix for the model file.
    
    Returns:
        Sequential: The loaded RNN model.
    """
    return tf.keras.models.load_model(f'{file_prefix}_rnn_model.h5')

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on validation data.
    
    Args:
        model (Sequential): The RNN model.
        X_val (np.array): Validation input data.
        y_val (np.array): Validation output data.
    
    Returns:
        float: MSE or RMSE of the predictions.
    """
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    return rmse, y_pred

def plot_predictions(y_val, y_pred, title):
    """
    Plot the actual vs predicted values.
    
    Args:
        y_val (np.array): Actual values.
        y_pred (np.array): Predicted values.
        title (str): Title for the plot.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_val, label='Actual', color='b')
    plt.plot(y_pred, label='Predicted', color='r')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Define the file prefixes for the datasets
    datasets = [
        ('bitstamp', 'bitstamp_rnn_model'),
        ('coinbase', 'coinbase_rnn_model')
    ]

    # Evaluate and plot the model for each dataset
    for data_prefix, model_prefix in datasets:
        # Load the validation data
        X_val, y_val = load_data(data_prefix)
        
        # Load the trained model
        model = load_model(model_prefix)
        
        # Evaluate the model
        rmse, y_pred = evaluate_model(model, X_val, y_val)
        print(f"{model_prefix} - RMSE: {rmse}")
        
        # Plot predictions vs actual values
        plot_predictions(y_val, y_pred, f'{model_prefix} Predictions vs Actual')

