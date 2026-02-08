"""
Data preprocessing utilities for wind turbine SCADA data.
Handles cleaning, feature engineering, and time-series preparation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def clean_scada_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    remove_outliers: bool = True,
    outlier_z_threshold: float = 4.0,
    fill_method: str = 'forward'
) -> pd.DataFrame:
    """
    Clean SCADA data by removing outliers and handling missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column (power output)
    feature_cols : List[str]
        List of feature column names
    remove_outliers : bool
        Whether to remove outliers using Z-score
    outlier_z_threshold : float
        Z-score threshold for outlier removal
    fill_method : str
        Method for filling missing values ('forward', 'backward', 'interpolate', 'drop')
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Select relevant columns
    all_cols = [target_col] + feature_cols
    missing_cols = [col for col in all_cols if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")
    
    df_clean = df_clean[all_cols].copy()
    
    # Remove negative power (invalid)
    if target_col in df_clean.columns:
        df_clean = df_clean[df_clean[target_col] >= 0].copy()
    
    # Remove outliers using Z-score
    if remove_outliers:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < outlier_z_threshold].copy()
    
    # Handle missing values
    if fill_method == 'forward':
        df_clean = df_clean.ffill().bfill()
    elif fill_method == 'backward':
        df_clean = df_clean.bfill().ffill()
    elif fill_method == 'interpolate':
        df_clean = df_clean.interpolate(method='time' if isinstance(df_clean.index, pd.DatetimeIndex) else 'linear')
        df_clean = df_clean.ffill().bfill()
    elif fill_method == 'drop':
        df_clean = df_clean.dropna()
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")
    
    return df_clean


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from timestamp index.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with DatetimeIndex
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added time features
    """
    df_feat = df.copy()
    
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have DatetimeIndex")
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_year'] = df_feat.index.dayofyear
    df_feat['month'] = df_feat.index.month
    
    # Cyclical encoding for periodic features
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_year'] / 365)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_year'] / 365)
    
    return df_feat


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 3, 6, 12]
) -> pd.DataFrame:
    """
    Create lagged features for time-series prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to create lags for
    lags : List[int]
        List of lag values
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with lagged features
    """
    df_lag = df.copy()
    
    for col in columns:
        if col not in df_lag.columns:
            continue
        for lag in lags:
            df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag)
    
    return df_lag


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [3, 6, 12, 24]
) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to create rolling features for
    windows : List[int]
        List of window sizes
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with rolling features
    """
    df_roll = df.copy()
    
    for col in columns:
        if col not in df_roll.columns:
            continue
        for window in windows:
            df_roll[f'{col}_mean_{window}'] = df_roll[col].rolling(window=window, min_periods=1).mean()
            df_roll[f'{col}_std_{window}'] = df_roll[col].rolling(window=window, min_periods=1).std()
            df_roll[f'{col}_max_{window}'] = df_roll[col].rolling(window=window, min_periods=1).max()
            df_roll[f'{col}_min_{window}'] = df_roll[col].rolling(window=window, min_periods=1).min()
    
    return df_roll


def prepare_sequences(
    data: np.ndarray,
    target: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1,
    stride: int = 1,
    max_samples: Optional[int] = None,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for time-series prediction.
    
    Parameters:
    -----------
    data : np.ndarray
        Input features (n_samples, n_features)
    target : np.ndarray
        Target values (n_samples,)
    sequence_length : int
        Length of input sequences
    forecast_horizon : int
        Number of steps ahead to predict
    stride : int
        Stride for sequence generation
    max_samples : int, optional
        Maximum number of sequences to generate. If None, generates all sequences.
        Useful for very large datasets to avoid memory issues.
    dtype : type
        Data type for arrays (default: np.float32 to save memory)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X (n_sequences, sequence_length, n_features), y (n_sequences, forecast_horizon)
    """
    n_samples = len(data)
    max_sequences = n_samples - sequence_length - forecast_horizon + 1
    
    # Convert input data to specified dtype to save memory
    if data.dtype != dtype:
        data = data.astype(dtype)
    if target.dtype != dtype:
        target = target.astype(dtype)
    
    # Determine actual number of sequences to generate
    if max_samples is not None and max_sequences > max_samples:
        # Use larger stride to sample evenly
        stride = max(1, max_sequences // max_samples)
        n_sequences = max_samples
        print(f"Large dataset detected. Using stride={stride} to limit to {max_samples} sequences")
    else:
        n_sequences = max_sequences
    
    # Pre-allocate arrays for better memory efficiency
    n_features = data.shape[1] if len(data.shape) > 1 else 1
    X = np.zeros((n_sequences, sequence_length, n_features), dtype=dtype)
    y = np.zeros((n_sequences, forecast_horizon), dtype=dtype)
    
    # Fill arrays incrementally
    count = 0
    for i in range(0, n_samples - sequence_length - forecast_horizon + 1, stride):
        if count >= n_sequences:
            break
        X[count] = data[i:i + sequence_length]
        y[count] = target[i + sequence_length:i + sequence_length + forecast_horizon]
        count += 1
    
    # Trim to actual size if we didn't fill completely
    if count < n_sequences:
        X = X[:count]
        y = y[:count]
    
    # Squeeze y if forecast_horizon is 1 to make it 1D
    if forecast_horizon == 1 and len(y.shape) > 1:
        y = y.squeeze(-1)
    
    return X, y


def time_aware_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (no data leakage).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe (should be sorted by time)
    train_ratio : float
        Proportion for training
    val_ratio : float
        Proportion for validation
    test_ratio : float
        Proportion for testing
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test splits
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    return train_data, val_data, test_data


class FeatureScaler:
    """Wrapper for feature scaling that preserves scaling parameters."""
    
    def __init__(self, method: str = 'standard'):
        """
        Parameters:
        -----------
        method : str
            Scaling method ('standard' or 'minmax')
        """
        self.method = method
        self.scaler = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame):
        """Fit scaler to data."""
        self.feature_names = X.columns.tolist()
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        self.scaler.fit(X.values)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X.values)
        return pd.DataFrame(X_scaled, index=X.index, columns=self.feature_names)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

