"""
LSTM (Long Short-Term Memory) Deep Learning Forecasting Module.

This module provides:
1. Chronological 80/20 train/test splitting (never shuffled).
2. Data normalization with MinMaxScaler fitted strictly on training data (no data leakage).
3. Sliding window sequence construction (60 past days to predict next day).
4. Alignment for the test set using the final 60 points of training data,
   ensuring the test forecast covers 100% of test dates.
5. Keras Sequential LSTM architecture with Dropout regularization and Dense layers.
6. Model training with EarlyStopping and validation monitoring.
7. Scale inversion and metrics computation (MAE, RMSE).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.utils import calculate_metrics, set_random_seeds


def prepare_lstm_datasets(
    series: pd.Series,
    seq_length: int = 60,
    train_ratio: float = 0.8,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    pd.Series,
    pd.Series,
]:
    """
    Split time series chronologically and construct sliding window sequences for LSTM.

    Zero Data Leakage Guarantee:
    - Scaler is fitted exclusively on the training partition.
    - Test sequences use the last `seq_length` rows of training data so that
      the very first test prediction aligns exactly with `test_series.iloc[0]`.

    Parameters:
    -----------
    series : pd.Series
        Historical closing price series with DatetimeIndex.
    seq_length : int
        Window size of past observations (default: 60).
    train_ratio : float
        Fraction of data for training (default: 0.8).

    Returns:
    --------
    tuple containing:
        - X_train : np.ndarray of shape (N_train, seq_length, 1)
        - y_train : np.ndarray of shape (N_train,)
        - X_test  : np.ndarray of shape (N_test, seq_length, 1)
        - y_test  : np.ndarray of shape (N_test,)
        - scaler  : Fitted MinMaxScaler instance
        - train_dates : DatetimeIndex for training targets
        - test_dates  : DatetimeIndex for test targets (identical to test_series.index)
        - train_series : pd.Series of training actual prices
        - test_series  : pd.Series of test actual prices
    """
    total_len = len(series)
    train_size = int(total_len * train_ratio)

    train_series = series.iloc[:train_size]
    test_series = series.iloc[train_size:]

    # Fit scaler strictly on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_vals = train_series.values.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_vals)

    # Generate training sequences
    X_train, y_train = [], []
    for i in range(seq_length, len(train_scaled)):
        X_train.append(train_scaled[i - seq_length : i, 0])
        y_train.append(train_scaled[i, 0])

    X_train = np.array(X_train)[..., np.newaxis]
    y_train = np.array(y_train)
    train_target_dates = train_series.index[seq_length:]

    # For test data: concatenate the trailing `seq_length` points of training data
    # so we can forecast the first test data point without missing any test dates
    test_input_combined = pd.concat([train_series.iloc[-seq_length:], test_series])
    test_scaled = scaler.transform(test_input_combined.values.reshape(-1, 1))

    X_test, y_test = [], []
    for i in range(seq_length, len(test_scaled)):
        X_test.append(test_scaled[i - seq_length : i, 0])
        y_test.append(test_scaled[i, 0])

    X_test = np.array(X_test)[..., np.newaxis]
    y_test = np.array(y_test)
    test_target_dates = test_series.index

    print(f"[INFO] Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
    print(f"[INFO] Test sequence count ({len(X_test)}) perfectly matches test set size ({len(test_series)}).")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        train_target_dates,
        test_target_dates,
        train_series,
        test_series,
    )


def build_lstm_model(
    seq_length: int = 60,
    lstm_units: tuple[int, int] = (50, 50),
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """
    Construct a Keras Sequential LSTM neural network.

    Architecture:
    - Input Layer (seq_length, 1)
    - LSTM Layer 1 (units=50, return_sequences=True)
    - Dropout Layer (rate=0.2)
    - LSTM Layer 2 (units=50, return_sequences=False)
    - Dropout Layer (rate=0.2)
    - Dense Hidden Layer (units=25, activation='relu')
    - Dense Output Layer (units=1)

    Parameters:
    -----------
    seq_length : int
        Number of time steps in each input sequence.
    lstm_units : tuple of ints
        Number of units in each LSTM layer.
    dropout_rate : float
        Fraction of neurons dropped out to prevent overfitting.
    learning_rate : float
        Initial learning rate for the Adam optimizer.

    Returns:
    --------
    Sequential
        Compiled Keras model.
    """
    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(units=lstm_units[0], return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units[1], return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=25, activation="relu"),
        Dense(units=1),
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def train_and_forecast_lstm(
    series: pd.Series,
    seq_length: int = 60,
    train_ratio: float = 0.8,
    epochs: int = 25,
    batch_size: int = 32,
    patience: int = 5,
    seed: int = 42,
) -> tuple[pd.Series, Sequential, dict[str, float], pd.Series, pd.Series]:
    """
    Full pipeline to prepare data, train LSTM model with EarlyStopping,
    predict test horizon, inverse-transform, and evaluate metrics.

    Parameters:
    -----------
    series : pd.Series
        Historical closing prices with DatetimeIndex.
    seq_length : int
        Window size of observations (default: 60).
    train_ratio : float
        Proportion for training split (default: 0.8).
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Patience epochs for EarlyStopping before terminating.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    tuple:
        - forecast_series : pd.Series of LSTM predictions aligned to test_series.index
        - model : Trained Keras Sequential model
        - metrics : Dictionary of evaluation metrics (MAE, RMSE, MAPE)
        - train_series : pd.Series of actual training prices
        - test_series : pd.Series of actual test prices
    """
    set_random_seeds(seed)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        train_target_dates,
        test_target_dates,
        train_series,
        test_series,
    ) = prepare_lstm_datasets(series, seq_length=seq_length, train_ratio=train_ratio)

    model = build_lstm_model(seq_length=seq_length)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"[INFO] Training LSTM model (epochs={epochs}, batch_size={batch_size})...")
    # Time-series data should not be shuffled in training
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Predict test sequences
    pred_scaled = model.predict(X_test, verbose=0)
    pred_unscaled = scaler.inverse_transform(pred_scaled).flatten()

    forecast_series = pd.Series(pred_unscaled, index=test_series.index, name="LSTM Forecast")
    metrics = calculate_metrics(test_series.values, forecast_series.values)

    print(f"[INFO] LSTM Test Results -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

    return forecast_series, model, metrics, train_series, test_series
