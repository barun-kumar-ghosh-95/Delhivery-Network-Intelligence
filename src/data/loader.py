"""
Data Loader Module
==================
Loads the raw Delhivery CSV, parses datetimes, splits train/test.
"""

import pandas as pd
import numpy as np
import os
import yaml


def load_config(config_path=None):
    """Load project configuration."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'configs', 'config.yaml'
        )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_data(data_path=None):
    """
    Load raw delivery data CSV.
    
    Parameters
    ----------
    data_path : str, optional
        Path to CSV file. If None, uses project root delivery_data.csv.
    
    Returns
    -------
    pd.DataFrame
        Raw dataframe with all columns.
    """
    if data_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(project_root, 'delivery_data.csv')
    
    print(f"[Loader] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[Loader] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def parse_datetimes(df):
    """
    Parse datetime columns from string to datetime64.
    
    Columns parsed:
    - trip_creation_time
    - od_start_time
    - od_end_time
    - cutoff_timestamp
    """
    datetime_cols = ['trip_creation_time', 'od_start_time', 'od_end_time', 'cutoff_timestamp']
    
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            parsed_pct = df[col].notna().mean() * 100
            print(f"[Loader] Parsed {col}: {parsed_pct:.1f}% valid")
    
    return df


def split_train_test(df, config=None):
    """
    Split data into train and test sets using the 'data' column.
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_df, test_df)
    """
    if config is None:
        config = load_config()
    
    col = config['data']['train_test_column']
    train_val = config['data']['train_value']
    test_val = config['data']['test_value']
    
    train_df = df[df[col] == train_val].copy()
    test_df = df[df[col] == test_val].copy()
    
    print(f"[Loader] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df


def load_and_prepare(data_path=None):
    """
    Full loading pipeline: load → parse datetimes → return.
    
    Returns
    -------
    pd.DataFrame
        Prepared dataframe with parsed datetimes.
    """
    df = load_raw_data(data_path)
    df = parse_datetimes(df)
    return df
