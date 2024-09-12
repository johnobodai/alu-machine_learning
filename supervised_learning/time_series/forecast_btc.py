#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

def build_model(input_shape):
    """
    Build the RNN model using LSTM layers.
    
    Args:
        input_shape (tuple): Shape of the input data.
    
    Returns:
        Sequential: Compiled RNN model.
    """
    model = Sequential()

    # Add LSTM layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Add Dense layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Train the RNN model.
    
    Args:
        model (Sequential): The RNN model.
        X_train (np.array): Model input data.
        y_train (np.array): Model output data.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.
    
    Returns:
        history: Training history for the model.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

def save_model(model, file_prefix):
    """
    Save the trained model to disk.
    
    Args:
        model (Sequential): The trained RNN model.
        file_prefix (str): Prefix for the saved model files.
    
    Returns:
        None
    """
    model.save(f'{file_prefix}_rnn_model.h5')
    print(f"Model saved as {file_prefix}_rnn_model.h5")

if __name__ == "__main__":
    # Define the file prefixes for the datasets
    datasets = [
        ('bitstamp', 'bitstamp_rnn_model'),
        ('coinbase', 'coinbase_rnn_model')
    ]

    # Train and save the model for each dataset
    for data_prefix, model_prefix in datasets:
        # Load the data
        X, y = load_data(data_prefix)
        
        # Build the model
        model = build_model((X.shape[1], 1))
        
        # Train the model
        train_model(model, X, y, epochs=10, batch_size=32)
        
        # Save the model
        save_model(model, model_prefix)

