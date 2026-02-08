"""
Data loading utilities for SCADA wind turbine data.
Handles CSV loading, timestamp parsing, and basic validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_scada_data(
    csv_path: str,
    timestamp_col: Optional[str] = None,
    parse_dates: bool = True,
    chunksize: Optional[int] = None,
    max_rows: Optional[int] = None,
    sample_ratio: Optional[float] = None
) -> pd.DataFrame:
    """
    Load SCADA data from CSV file. Handles large files with chunking.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    timestamp_col : str, optional
        Name of timestamp column. If None, attempts auto-detection.
    parse_dates : bool
        Whether to parse timestamp column as datetime
    chunksize : int, optional
        Number of rows to read per chunk. If None, attempts to load all at once.
        If file is too large, use chunksize (e.g., 100000) to load in chunks.
    max_rows : int, optional
        Maximum number of rows to load. If specified, only loads first N rows.
    sample_ratio : float, optional
        If specified (0-1), randomly samples this fraction of rows. Useful for very large files.
        
    Returns:
    --------
    pd.DataFrame
        Loaded SCADA data with parsed timestamps
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # First, try to read just the header to detect timestamp column
    try:
        header_df = pd.read_csv(csv_path, nrows=1000, low_memory=False)
    except Exception as e:
        raise ValueError(f"Error reading CSV header: {e}")
    
    # Auto-detect timestamp column if not specified
    if timestamp_col is None:
        # Common timestamp column names
        possible_names = ['timestamp', 'Timestamp', 'Time', 'time', 'DateTime', 
                         'datetime', 'Date', 'date', 'TimeStamp']
        timestamp_col = None
        for name in possible_names:
            if name in header_df.columns:
                timestamp_col = name
                break
        
        # If still not found, try first column if it looks like datetime
        if timestamp_col is None and len(header_df.columns) > 0:
            first_col = header_df.columns[0]
            if header_df[first_col].dtype == 'object':
                try:
                    pd.to_datetime(header_df[first_col].iloc[0])
                    timestamp_col = first_col
                except:
                    pass
    
    # Load CSV with chunking if specified or if file is very large
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    
    if chunksize is None and file_size_mb > 500:
        # Auto-set chunksize for large files
        chunksize = 100000
        print(f"Large file detected ({file_size_mb:.1f} MB). Using chunking with chunksize={chunksize}")
    
    if chunksize is not None:
        # Load in chunks
        print(f"Loading data in chunks of {chunksize} rows...")
        chunks = []
        chunk_count = 0
        
        try:
            for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
                chunks.append(chunk)
                chunk_count += 1
                
                if max_rows is not None and len(pd.concat(chunks, ignore_index=True)) >= max_rows:
                    break
                
                if chunk_count % 10 == 0:
                    print(f"  Loaded {chunk_count} chunks ({len(pd.concat(chunks, ignore_index=True)):,} rows)...")
            
            df = pd.concat(chunks, ignore_index=True)
            print(f"Loaded {len(df):,} rows in {chunk_count} chunks")
            
        except MemoryError:
            raise MemoryError(
                f"Out of memory loading file. Try:\n"
                f"  1. Use chunksize parameter (e.g., chunksize=50000)\n"
                f"  2. Use max_rows parameter to limit rows\n"
                f"  3. Use sample_ratio parameter to sample data"
            )
    else:
        # Try to load all at once
        try:
            if max_rows is not None:
                df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
            else:
                df = pd.read_csv(csv_path, low_memory=False)
        except MemoryError:
            raise MemoryError(
                f"Out of memory loading file. Try:\n"
                f"  1. Use chunksize parameter (e.g., chunksize=50000)\n"
                f"  2. Use max_rows parameter to limit rows\n"
                f"  3. Use sample_ratio parameter to sample data"
            )
    
    # Apply sampling if specified
    if sample_ratio is not None and 0 < sample_ratio < 1:
        n_sample = int(len(df) * sample_ratio)
        df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df):,} rows ({sample_ratio*100:.1f}% of data)")
    
    # Parse timestamps
    if timestamp_col and timestamp_col in df.columns and parse_dates:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        df.index = df[timestamp_col]
    
    return df


def get_feature_info(df: pd.DataFrame) -> dict:
    """
    Get information about features in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary with feature information
    """
    info = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': df.select_dtypes(exclude=[np.number]).columns.tolist(),
    }
    
    # Sampling frequency if timestamp index exists
    if isinstance(df.index, pd.DatetimeIndex):
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            median_freq = time_diffs.median()
            info['sampling_frequency'] = str(median_freq)
            info['sampling_frequency_seconds'] = median_freq.total_seconds()
    
    return info


def identify_target_features(df: pd.DataFrame) -> dict:
    """
    Identify key features for wind power prediction.
    Attempts to find columns matching common SCADA feature names.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary mapping feature types to column names
    """
    columns_lower = [col.lower() for col in df.columns]
    
    features = {
        'wind_speed': None,
        'wind_direction': None,
        'power': None,
        'temperature': None,
        'rotor_speed': None,
        'pitch_angle': None,
    }
    
    # Wind speed patterns
    for pattern in ['wind_speed', 'windspeed', 'ws', 'wind speed', 'v', 'velocity']:
        for i, col_lower in enumerate(columns_lower):
            if pattern in col_lower and features['wind_speed'] is None:
                features['wind_speed'] = df.columns[i]
                break
    
    # Wind direction patterns
    for pattern in ['wind_direction', 'winddirection', 'wd', 'wind dir', 'direction']:
        for i, col_lower in enumerate(columns_lower):
            if pattern in col_lower and features['wind_direction'] is None:
                features['wind_direction'] = df.columns[i]
                break
    
    # Power patterns
    for pattern in ['power', 'active_power', 'p', 'output', 'generated']:
        for i, col_lower in enumerate(columns_lower):
            if pattern in col_lower and features['power'] is None:
                features['power'] = df.columns[i]
                break
    
    # Temperature patterns
    for pattern in ['temperature', 'temp', 't', 'ambient']:
        for i, col_lower in enumerate(columns_lower):
            if pattern in col_lower and features['temperature'] is None:
                features['temperature'] = df.columns[i]
                break
    
    # Rotor speed patterns
    for pattern in ['rotor', 'rpm', 'rotational', 'rotor_speed']:
        for i, col_lower in enumerate(columns_lower):
            if pattern in col_lower and features['rotor_speed'] is None:
                features['rotor_speed'] = df.columns[i]
                break
    
    # Pitch angle patterns
    for pattern in ['pitch', 'blade', 'angle']:
        for i, col_lower in enumerate(columns_lower):
            if pattern in col_lower and features['pitch_angle'] is None:
                features['pitch_angle'] = df.columns[i]
                break
    
    return features

