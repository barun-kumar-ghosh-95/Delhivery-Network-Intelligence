"""
Temporal Feature Engineering
============================
Extract time-based features from datetime columns.
"""

import pandas as pd
import numpy as np


def extract_temporal_features(df):
    """Extract temporal features from trip_creation_time and od_start_time."""
    print("[Features] Extracting temporal features...")
    df = df.copy()
    
    # From trip_creation_time
    if 'trip_creation_time' in df.columns:
        dt = pd.to_datetime(df['trip_creation_time'], errors='coerce')
        df['creation_hour'] = dt.dt.hour
        df['creation_day_of_week'] = dt.dt.dayofweek
        df['creation_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        df['creation_is_peak'] = dt.dt.hour.isin([8,9,10,17,18,19]).astype(int)
        df['creation_is_night'] = dt.dt.hour.isin([22,23,0,1,2,3,4,5]).astype(int)
        df['creation_day_of_month'] = dt.dt.day
        df['creation_week_of_year'] = dt.dt.isocalendar().week.astype(int)
        
        # Cyclical encoding for hour
        df['creation_hour_sin'] = np.sin(2 * np.pi * df['creation_hour'] / 24)
        df['creation_hour_cos'] = np.cos(2 * np.pi * df['creation_hour'] / 24)
        df['creation_dow_sin'] = np.sin(2 * np.pi * df['creation_day_of_week'] / 7)
        df['creation_dow_cos'] = np.cos(2 * np.pi * df['creation_day_of_week'] / 7)
    
    # From od_start_time
    if 'od_start_time' in df.columns:
        dt = pd.to_datetime(df['od_start_time'], errors='coerce')
        df['start_hour'] = dt.dt.hour
        df['start_is_peak'] = dt.dt.hour.isin([8,9,10,17,18,19]).astype(int)
        
        # Time since trip creation
        if 'trip_creation_time' in df.columns:
            creation_dt = pd.to_datetime(df['trip_creation_time'], errors='coerce')
            df['time_since_creation_min'] = (dt - creation_dt).dt.total_seconds() / 60
            df['time_since_creation_min'] = df['time_since_creation_min'].clip(lower=0).fillna(0)
    
    print(f"[Features] Added temporal features. Shape: {df.shape}")
    return df
