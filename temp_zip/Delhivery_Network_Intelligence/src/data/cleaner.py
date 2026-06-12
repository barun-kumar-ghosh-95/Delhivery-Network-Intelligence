"""
Data Cleaner Module
===================
Handles data quality issues:
- Negative segment_actual_time values
- Zero OSRM values
- Facility name parsing (city, state, facility_type)
- Missing value imputation
"""

import pandas as pd
import numpy as np
import re


def extract_facility_info(name):
    """
    Parse facility name to extract city, state, and facility type.
    
    Format: 'City_Area_TYPE (State)' or 'City_AreaTYPE_suffix (State)'
    Examples:
        'Gurgaon_Bilaspur_HB (Haryana)' → city=Gurgaon, state=Haryana, type=HB
        'Bangalore_Nelmngla_H (Karnataka)' → city=Bangalore, state=Karnataka, type=H
        'Khambhat_MotvdDPP_D (Gujarat)' → city=Khambhat, state=Gujarat, type=DPP
    """
    if pd.isna(name) or name == '':
        return None, None, None
    
    # Extract state from parentheses
    state = None
    state_match = re.search(r'\(([^)]+)\)', name)
    if state_match:
        state = state_match.group(1).strip()
    
    # Extract city (first part before underscore)
    name_part = re.sub(r'\s*\([^)]+\)\s*', '', name).strip()
    parts = name_part.split('_')
    city = parts[0] if parts else None
    
    # Extract facility type (last meaningful suffix)
    facility_type = 'Unknown'
    type_patterns = ['HB', 'H', 'DC', 'DPP', 'D', 'CP', 'BO', 'BM', 'GW']
    
    # Check last part for type
    if len(parts) >= 2:
        last_part = parts[-1].upper()
        for tp in type_patterns:
            if last_part == tp or last_part.endswith(tp):
                facility_type = tp
                break
    
    # Fallback: search in full name
    if facility_type == 'Unknown':
        for tp in type_patterns:
            if f'_{tp}' in name.upper() or f'_{tp} ' in name.upper():
                facility_type = tp
                break
    
    return city, state, facility_type


def clean_data(df):
    """
    Clean raw data:
    1. Handle negative segment_actual_time (21 rows) → clip to 0
    2. Handle zero OSRM values → replace with NaN for imputation
    3. Parse facility names → city, state, facility_type
    4. Fill missing source_name / destination_name
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with new columns
    """
    df = df.copy()
    print("[Cleaner] Starting data cleaning...")
    
    # 1. Handle negative segment_actual_time
    neg_count = (df['segment_actual_time'] < 0).sum()
    if neg_count > 0:
        df.loc[df['segment_actual_time'] < 0, 'segment_actual_time'] = 0
        print(f"[Cleaner] Fixed {neg_count} negative segment_actual_time values → 0")
    
    # 2. Handle zero OSRM values
    for col in ['segment_osrm_time', 'segment_osrm_distance']:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            # Replace zeros with NaN; they'll be handled during aggregation
            df.loc[df[col] == 0, col] = np.nan
            print(f"[Cleaner] Marked {zero_count} zero values in {col} as NaN")
    
    # 3. Parse facility names
    print("[Cleaner] Parsing facility names...")
    
    # Source facilities
    source_info = df['source_name'].apply(extract_facility_info)
    df['source_city'] = source_info.apply(lambda x: x[0])
    df['source_state'] = source_info.apply(lambda x: x[1])
    df['source_facility_type'] = source_info.apply(lambda x: x[2])
    
    # Destination facilities
    dest_info = df['destination_name'].apply(extract_facility_info)
    df['dest_city'] = dest_info.apply(lambda x: x[0])
    df['dest_state'] = dest_info.apply(lambda x: x[1])
    df['dest_facility_type'] = dest_info.apply(lambda x: x[2])
    
    # 4. Fill missing names from center codes where possible
    # Build a mapping from center code to name
    src_map = df.dropna(subset=['source_name']).drop_duplicates('source_center')\
                .set_index('source_center')['source_name'].to_dict()
    dst_map = df.dropna(subset=['destination_name']).drop_duplicates('destination_center')\
                .set_index('destination_center')['destination_name'].to_dict()
    
    src_missing = df['source_name'].isna().sum()
    dst_missing = df['destination_name'].isna().sum()
    
    df['source_name'] = df['source_name'].fillna(df['source_center'].map(src_map))
    df['destination_name'] = df['destination_name'].fillna(df['destination_center'].map(dst_map))
    
    src_fixed = src_missing - df['source_name'].isna().sum()
    dst_fixed = dst_missing - df['destination_name'].isna().sum()
    print(f"[Cleaner] Filled {src_fixed}/{src_missing} missing source names, {dst_fixed}/{dst_missing} dest names")
    
    # 5. Add derived flags
    df['is_intrastate'] = (df['source_state'] == df['dest_state']).astype(int)
    
    print(f"[Cleaner] Cleaning complete. Shape: {df.shape}")
    return df
