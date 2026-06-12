"""
Trip-Level Aggregator Module
=============================
Aggregates segment-level data to trip-level for modeling.
Each row in raw data is a segment cutoff observation (~9.78 segments per trip).
We need trip-level aggregation for the target (actual_time).
"""

import pandas as pd
import numpy as np


def aggregate_trips(df):
    """
    Aggregate segment-level data to trip-level.
    
    Strategy:
    - Cumulative columns (osrm_time, osrm_distance, actual_time): take MAX per trip
    - Segment columns: aggregate (sum, mean, count)
    - Categorical columns: take first (they're constant within a trip)
    - Temporal columns: take first (trip-level)
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned segment-level dataframe
    
    Returns
    -------
    pd.DataFrame
        Trip-level dataframe (one row per trip)
    """
    print("[Aggregator] Aggregating segments to trip level...")
    
    # Define aggregation rules
    agg_dict = {
        # Meta
        'data': 'first',
        'route_schedule_uuid': 'first',
        'route_type': 'first',
        
        # Facilities (first and last in trip)
        'source_center': 'first',
        'source_name': 'first',
        'destination_center': 'last',
        'destination_name': 'last',
        
        # Parsed facility info
        'source_city': 'first',
        'source_state': 'first',
        'source_facility_type': 'first',
        'dest_city': 'last',
        'dest_state': 'last',
        'dest_facility_type': 'last',
        'is_intrastate': 'first',
        
        # Temporal (trip-level)
        'trip_creation_time': 'first',
        'od_start_time': 'first',
        
        # Cumulative columns → take MAX (cumulative max = trip total)
        'osrm_time': 'max',
        'osrm_distance': 'max',
        'actual_time': 'max',
        'actual_distance_to_destination': 'max',
        'start_scan_to_end_scan': 'max',
        
        # Segment-level aggregations
        'segment_osrm_time': ['sum', 'mean', 'std', 'max'],
        'segment_osrm_distance': ['sum', 'mean', 'std', 'max'],
        'segment_actual_time': ['sum', 'mean', 'std', 'max'],
        
        # Factor columns (for analysis only)
        'factor': 'max',
        
        # Count of segments
        'is_cutoff': 'sum',
    }
    
    # Group by trip_uuid and aggregate
    trip_df = df.groupby('trip_uuid').agg(agg_dict)
    
    # Flatten multi-level column names
    trip_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] != 'first' and col[1] != 'last' and col[1] != 'max' and col[1] != ''
        else f"{col[0]}_{col[1]}" if col[1] in ['sum', 'mean', 'std', 'max'] and col[0].startswith('segment')
        else col[0]
        for col in trip_df.columns
    ]
    
    # Fix column names manually for clarity
    rename_map = {}
    for col in trip_df.columns:
        if col.endswith('_first') or col.endswith('_last'):
            base = col.rsplit('_', 1)[0]
            if base in ['source_center', 'source_name', 'destination_center', 'destination_name',
                        'source_city', 'source_state', 'source_facility_type',
                        'dest_city', 'dest_state', 'dest_facility_type',
                        'data', 'route_schedule_uuid', 'route_type', 'is_intrastate',
                        'trip_creation_time', 'od_start_time']:
                rename_map[col] = base
    
    trip_df = trip_df.rename(columns=rename_map)
    
    # Add segment count
    seg_counts = df.groupby('trip_uuid').size()
    trip_df['num_segments'] = seg_counts
    
    # Reset index
    trip_df = trip_df.reset_index()
    
    # Add derived features
    trip_df['osrm_speed'] = np.where(
        trip_df['osrm_time'] > 0,
        trip_df['osrm_distance'] / (trip_df['osrm_time'] / 60),  # km/h
        0
    )
    
    # Rename is_cutoff sum to cutoff_count
    if 'is_cutoff' in trip_df.columns:
        trip_df = trip_df.rename(columns={'is_cutoff': 'cutoff_count'})
    
    print(f"[Aggregator] Aggregated {len(df):,} segments → {len(trip_df):,} trips")
    print(f"[Aggregator] Columns: {trip_df.shape[1]}")
    
    return trip_df


def get_corridor_stats(df):
    """
    Compute corridor-level statistics from segment data.
    A corridor is defined as (source_center, destination_center).
    
    Returns
    -------
    pd.DataFrame
        Corridor-level statistics for graph edge features.
    """
    print("[Aggregator] Computing corridor statistics...")
    
    corridor_df = df.groupby(['source_center', 'destination_center']).agg({
        'actual_time': ['mean', 'median', 'std', 'count'],
        'osrm_time': ['mean', 'max'],
        'osrm_distance': ['mean', 'max'],
        'factor': ['mean', 'median', 'std'],
        'route_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown',
        'trip_uuid': 'nunique',
    }).reset_index()
    
    # Flatten columns
    corridor_df.columns = [
        'source_center', 'destination_center',
        'actual_time_mean', 'actual_time_median', 'actual_time_std', 'segment_count',
        'osrm_time_mean', 'osrm_time_max',
        'osrm_distance_mean', 'osrm_distance_max',
        'delay_ratio_mean', 'delay_ratio_median', 'delay_ratio_std',
        'primary_route_type', 'trip_count',
    ]
    
    # Derived corridor metrics
    corridor_df['corridor_reliability'] = np.where(
        corridor_df['delay_ratio_mean'] > 0,
        1 - (corridor_df['delay_ratio_std'].fillna(0) / corridor_df['delay_ratio_mean']),
        0
    )
    corridor_df['corridor_reliability'] = corridor_df['corridor_reliability'].clip(0, 1)
    
    # SLA breach percentage (factor > 1.5)
    sla_breach = df.groupby(['source_center', 'destination_center']).apply(
        lambda x: (x['factor'] > 1.5).mean()
    ).reset_index(name='sla_breach_pct')
    
    corridor_df = corridor_df.merge(sla_breach, on=['source_center', 'destination_center'], how='left')
    
    # Delay classification
    corridor_df['delay_class'] = pd.cut(
        corridor_df['delay_ratio_mean'],
        bins=[0, 1.3, 2.0, 3.0, float('inf')],
        labels=['Healthy', 'Moderate', 'Severe', 'Critical']
    )
    
    print(f"[Aggregator] Computed stats for {len(corridor_df):,} corridors")
    return corridor_df


def get_facility_stats(df):
    """
    Compute facility-level statistics for graph node features.
    
    Returns
    -------
    pd.DataFrame
        Facility-level statistics.
    """
    print("[Aggregator] Computing facility statistics...")
    
    # Source facility stats
    src_stats = df.groupby('source_center').agg({
        'actual_time': ['mean', 'median', 'count'],
        'osrm_time': 'mean',
        'factor': ['mean', 'std'],
        'destination_center': 'nunique',
        'trip_uuid': 'nunique',
    }).reset_index()
    src_stats.columns = [
        'facility_id',
        'avg_delay_as_source', 'median_delay_as_source', 'throughput_as_source',
        'avg_osrm_as_source',
        'avg_factor_as_source', 'factor_std_as_source',
        'out_degree', 'unique_trips_as_source',
    ]
    
    # Destination facility stats
    dst_stats = df.groupby('destination_center').agg({
        'actual_time': ['mean', 'median', 'count'],
        'osrm_time': 'mean',
        'factor': ['mean', 'std'],
        'source_center': 'nunique',
        'trip_uuid': 'nunique',
    }).reset_index()
    dst_stats.columns = [
        'facility_id',
        'avg_delay_as_dest', 'median_delay_as_dest', 'throughput_as_dest',
        'avg_osrm_as_dest',
        'avg_factor_as_dest', 'factor_std_as_dest',
        'in_degree', 'unique_trips_as_dest',
    ]
    
    # Merge
    facility_df = src_stats.merge(dst_stats, on='facility_id', how='outer')
    facility_df = facility_df.fillna(0)
    
    # Derived metrics
    facility_df['total_throughput'] = facility_df['throughput_as_source'] + facility_df['throughput_as_dest']
    facility_df['total_degree'] = facility_df['out_degree'] + facility_df['in_degree']
    facility_df['avg_delay'] = (
        facility_df['avg_delay_as_source'] * facility_df['throughput_as_source'] +
        facility_df['avg_delay_as_dest'] * facility_df['throughput_as_dest']
    ) / facility_df['total_throughput'].clip(lower=1)
    
    # SLA breach rate
    sla_src = df.groupby('source_center').apply(lambda x: (x['factor'] > 1.5).mean()).reset_index()
    sla_src.columns = ['facility_id', 'sla_breach_pct_src']
    sla_dst = df.groupby('destination_center').apply(lambda x: (x['factor'] > 1.5).mean()).reset_index()
    sla_dst.columns = ['facility_id', 'sla_breach_pct_dst']
    
    facility_df = facility_df.merge(sla_src, on='facility_id', how='left')
    facility_df = facility_df.merge(sla_dst, on='facility_id', how='left')
    facility_df['sla_breach_pct'] = (
        facility_df['sla_breach_pct_src'].fillna(0) + facility_df['sla_breach_pct_dst'].fillna(0)
    ) / 2
    
    # Get facility name and type
    name_map = {}
    for _, row in df.drop_duplicates('source_center')[['source_center', 'source_name']].dropna().iterrows():
        name_map[row['source_center']] = row['source_name']
    for _, row in df.drop_duplicates('destination_center')[['destination_center', 'destination_name']].dropna().iterrows():
        name_map[row['destination_center']] = row['destination_name']
    
    facility_df['facility_name'] = facility_df['facility_id'].map(name_map)
    
    # Extract facility type from name
    def get_type(name):
        if pd.isna(name):
            return 'Unknown'
        for t in ['HB', 'DC', 'DPP', 'CP', 'BO', '_H ', '_H(', '_D ', '_D(']:
            if t in name or name.endswith(t.strip()):
                return t.strip().replace('(', '').replace('_', '')
        return 'Unknown'
    
    facility_df['facility_type'] = facility_df['facility_name'].apply(get_type)
    
    print(f"[Aggregator] Computed stats for {len(facility_df):,} facilities")
    return facility_df
