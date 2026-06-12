"""
Graph Features Module
=====================
Merge graph embeddings and structural features into ML feature set.
"""

import pandas as pd
import numpy as np


def merge_graph_features(trip_df, embeddings_df, graph_stats_df):
    """
    Merge graph-derived features into the trip-level dataframe.
    Each trip has source and destination facilities → merge both sets.
    """
    print("[GraphFeatures] Merging graph features into trip data...")
    df = trip_df.copy()
    
    # Merge source embeddings
    if embeddings_df is not None and len(embeddings_df) > 0:
        emb_cols = [c for c in embeddings_df.columns if c.startswith('emb_')]
        src_emb = embeddings_df[['facility_id'] + emb_cols].copy()
        src_emb = src_emb.rename(columns={c: f'src_{c}' for c in emb_cols})
        df = df.merge(src_emb, left_on='source_center', right_on='facility_id', how='left').drop(columns=['facility_id'], errors='ignore')
        
        # Merge destination embeddings
        dst_emb = embeddings_df[['facility_id'] + emb_cols].copy()
        dst_emb = dst_emb.rename(columns={c: f'dst_{c}' for c in emb_cols})
        df = df.merge(dst_emb, left_on='destination_center', right_on='facility_id', how='left').drop(columns=['facility_id'], errors='ignore')
    
    # Merge source graph stats
    if graph_stats_df is not None and len(graph_stats_df) > 0:
        stat_cols = [c for c in graph_stats_df.columns if c.startswith('graph_')]
        src_stats = graph_stats_df[['facility_id'] + stat_cols].copy()
        src_stats = src_stats.rename(columns={c: f'src_{c}' for c in stat_cols})
        df = df.merge(src_stats, left_on='source_center', right_on='facility_id', how='left').drop(columns=['facility_id'], errors='ignore')
        
        dst_stats = graph_stats_df[['facility_id'] + stat_cols].copy()
        dst_stats = dst_stats.rename(columns={c: f'dst_{c}' for c in stat_cols})
        df = df.merge(dst_stats, left_on='destination_center', right_on='facility_id', how='left').drop(columns=['facility_id'], errors='ignore')
    
    # Fill NaN graph features with 0
    graph_cols = [c for c in df.columns if c.startswith('src_emb_') or c.startswith('dst_emb_') or 
                  c.startswith('src_graph_') or c.startswith('dst_graph_')]
    df[graph_cols] = df[graph_cols].fillna(0)
    
    print(f"[GraphFeatures] Merged {len(graph_cols)} graph features. Shape: {df.shape}")
    return df
