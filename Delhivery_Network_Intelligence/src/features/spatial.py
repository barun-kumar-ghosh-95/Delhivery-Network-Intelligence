"""
Spatial Feature Engineering
===========================
Facility encoding and spatial features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_facilities(df, target_col='actual_time'):
    """Target-encode high-cardinality facility columns."""
    print("[Features] Encoding facility features...")
    df = df.copy()
    
    # Target encoding for source/destination centers
    for col in ['source_center', 'destination_center']:
        if col in df.columns and target_col in df.columns:
            # Compute mean target per category on training data
            train_mask = df['data'] == 'training' if 'data' in df.columns else pd.Series(True, index=df.index)
            means = df.loc[train_mask].groupby(col)[target_col].mean()
            global_mean = df.loc[train_mask, target_col].mean()
            
            # Apply encoding
            encoded_col = f'{col}_encoded'
            df[encoded_col] = df[col].map(means).fillna(global_mean)
    
    # Label encode facility types
    for col in ['source_facility_type', 'dest_facility_type']:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown').astype(str))
    
    # Label encode states
    for col in ['source_state', 'dest_state']:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown').astype(str))
    
    print(f"[Features] Encoded spatial features. Shape: {df.shape}")
    return df


def add_spatial_features(df):
    """Add derived spatial features."""
    df = df.copy()
    
    # Is same city?
    if 'source_city' in df.columns and 'dest_city' in df.columns:
        df['is_same_city'] = (df['source_city'] == df['dest_city']).astype(int)
    
    # Facility type interactions
    if 'source_facility_type' in df.columns and 'dest_facility_type' in df.columns:
        df['facility_type_pair'] = df['source_facility_type'].fillna('X') + '_' + df['dest_facility_type'].fillna('X')
        le = LabelEncoder()
        df['facility_type_pair_encoded'] = le.fit_transform(df['facility_type_pair'])
    
    return df
