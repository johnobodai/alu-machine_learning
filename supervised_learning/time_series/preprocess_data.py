#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data(file_path):
    """
    Load and clean the dataset.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Cleaned data.
    """
    df = pd.read_csv(file_path)
    
    # Convert the 'Timestamp' column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    
    # Set the 'Timestamp' as the index
    df.set_index('Timestamp', inplace=True)
    
    # Forward fill missing values
    df.ffill(inplace=True)
    
    return df

def feature_selection(df):
    """
    Select the relevant features for training.
    
    Args:
        df (pd.DataFrame): Dataframe with all features.
    
    Returns:
        pd.DataFrame: Dataframe with selected features.
    """
    # Selecting 'Close' as the target feature
    df = df[['Close']]
    return df

def rescale_data(df):
    """
    Normalize the features.
    
    Args:
        df (pd.DataFrame): Dataframe with selected features.
    
    Returns:
        pd.DataFrame: Rescaled data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Close'] = scaler.fit_transform(df[['Close']])
    return df, scaler

def create_time_series(df, window_size):
    """
    Create sliding windows of data for time series forecasting.
    
    Args:
        df (pd.DataFrame): Dataframe with rescaled data.
        window_size (int): The size of the sliding window.
    
    Returns:
        np.array: X data for model input.
        np.array: y data for model output.
    """
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df.iloc[i-window_size:i, 0].values)
        y.append(df.iloc[i, 0])
    return np.array(X), np.array(y)

def save_processed_data(X, y, file_prefix):
    """
    Save the processed data.
    
    Args:
        X (np.array): Model input data.
        y (np.array): Model output data.
        file_prefix (str): Prefix for the output files.
    
    Returns:
        None
    """
    np.save(f'{file_prefix}_X.npy', X)
    np.save(f'{file_prefix}_y.npy', y)

def preprocess(file_path, window_size, file_prefix):
    """
    Full preprocessing pipeline.
    
    Args:
        file_path (str): Path to the CSV file.
        window_size (int): The size of the sliding window.
        file_prefix (str): Prefix for the output files.
    
    Returns:
        None
    """
    df = load_data(file_path)
    df = feature_selection(df)
    df, scaler = rescale_data(df)
    X, y = create_time_series(df, window_size)
    save_processed_data(X, y, file_prefix)
    print(f"Preprocessing complete. Processed data saved with prefix '{file_prefix}'.")

if __name__ == "__main__":
    # Paths to the CSV files and corresponding file prefixes
    datasets = [
        ('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', 'bitstamp'),
        ('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', 'coinbase')
    ]

    # Window size for time series
    window_size = 60  # 60 minutes

    # Preprocess each dataset
    for file_path, file_prefix in datasets:
        preprocess(file_path, window_size, file_prefix)

