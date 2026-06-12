"""
Route Feature Engineering
=========================
OSRM and route-level features.
"""

import pandas as pd
import numpy as np


def add_route_features(df):
    """Add route-level features."""
    print("[Features] Adding route features...")
    df = df.copy()
    
    # Route type encoding
    if 'route_type' in df.columns:
        df['is_ftl'] = (df['route_type'] == 'FTL').astype(int)
    
    # OSRM-derived features
    if 'osrm_time' in df.columns and 'osrm_distance' in df.columns:
        # Speed
        df['osrm_speed_kmh'] = np.where(
            df['osrm_time'] > 0,
            df['osrm_distance'] / (df['osrm_time'] / 60),
            0
        )
        
        # Log transforms for skewed distributions
        df['log_osrm_time'] = np.log1p(df['osrm_time'].clip(lower=0))
        df['log_osrm_distance'] = np.log1p(df['osrm_distance'].clip(lower=0))
    
    # Segment features
    if 'num_segments' in df.columns:
        df['log_num_segments'] = np.log1p(df['num_segments'])
    
    # Segment OSRM aggregations
    for col in ['segment_osrm_time_sum', 'segment_osrm_distance_sum']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0).fillna(0))
    
    # Avg segment distance
    if 'segment_osrm_distance_sum' in df.columns and 'num_segments' in df.columns:
        df['avg_segment_distance'] = np.where(
            df['num_segments'] > 0,
            df['segment_osrm_distance_sum'] / df['num_segments'],
            0
        )
    
    # Avg segment time
    if 'segment_osrm_time_sum' in df.columns and 'num_segments' in df.columns:
        df['avg_segment_time'] = np.where(
            df['num_segments'] > 0,
            df['segment_osrm_time_sum'] / df['num_segments'],
            0
        )
    
    print(f"[Features] Added route features. Shape: {df.shape}")
    return df
