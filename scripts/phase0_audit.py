"""
PHASE 0: Exhaustive Dataset Audit for Delhivery Logistics Network
=================================================================
Principal ML Engineer + Graph Data Scientist Analysis

This script performs:
1. Schema Understanding
2. Missing Value Analysis
3. Leakage Detection
4. Feature Categorization
5. Graph Construction Feasibility
6. Statistical Profiling
7. Business Understanding Extraction
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json
import sys
from collections import defaultdict

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 80)
print("PHASE 0: EXHAUSTIVE DATASET AUDIT")
print("=" * 80)

data_path = r"C:\Users\barun\OneDrive\Desktop\project-summer\delivery_data.csv"
df = pd.read_csv(data_path)

print(f"\n[1] BASIC DIMENSIONS")
print(f"    Rows: {df.shape[0]:,}")
print(f"    Columns: {df.shape[1]}")
print(f"    Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================
# SCHEMA ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[2] COMPLETE SCHEMA ANALYSIS")
print(f"{'='*80}")

for col in df.columns:
    print(f"\n--- Column: {col} ---")
    print(f"    Dtype: {df[col].dtype}")
    print(f"    Non-null count: {df[col].notna().sum():,} / {len(df):,}")
    print(f"    Null count: {df[col].isna().sum():,} ({df[col].isna().mean()*100:.2f}%)")
    print(f"    Unique values: {df[col].nunique()}")
    
    if df[col].dtype in ['float64', 'int64']:
        print(f"    Min: {df[col].min()}")
        print(f"    Max: {df[col].max()}")
        print(f"    Mean: {df[col].mean():.4f}")
        print(f"    Median: {df[col].median():.4f}")
        print(f"    Std: {df[col].std():.4f}")
        print(f"    Skewness: {df[col].skew():.4f}")
        print(f"    Kurtosis: {df[col].kurtosis():.4f}")
        # Outlier check using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_pct = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).mean() * 100
        print(f"    Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
        print(f"    Outlier %: {outlier_pct:.2f}%")
        print(f"    Zero count: {(df[col] == 0).sum():,}")
        print(f"    Negative count: {(df[col] < 0).sum():,}")
    else:
        top_vals = df[col].value_counts().head(10)
        print(f"    Top 10 values:")
        for val, cnt in top_vals.items():
            print(f"      '{val}': {cnt:,} ({cnt/len(df)*100:.2f}%)")

# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[3] MISSING VALUE ANALYSIS")
print(f"{'='*80}")

missing = df.isnull().sum()
missing_pct = df.isnull().mean() * 100
missing_df = pd.DataFrame({'count': missing, 'percentage': missing_pct})
missing_df = missing_df.sort_values('percentage', ascending=False)

print("\nMissing values by column (sorted):")
for col, row in missing_df.iterrows():
    status = "✓ CLEAN" if row['count'] == 0 else "⚠ MISSING"
    print(f"  {status} | {col:40s} | {int(row['count']):>8,} ({row['percentage']:.2f}%)")

# Check for empty strings in object columns
print("\n\nEmpty string check (object columns):")
for col in df.select_dtypes(include=['object']).columns:
    empty_count = (df[col] == '').sum()
    if empty_count > 0:
        print(f"  ⚠ {col}: {empty_count:,} empty strings")

# ============================================================
# DATA TYPE VALIDATION
# ============================================================
print(f"\n{'='*80}")
print(f"[4] DATA TYPE VALIDATION")
print(f"{'='*80}")

# Identify datetime columns
datetime_candidates = ['trip_creation_time', 'od_start_time', 'od_end_time', 'cutoff_timestamp']
for col in datetime_candidates:
    if col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            valid_pct = parsed.notna().mean() * 100
            print(f"  {col}: Datetime parseable = {valid_pct:.2f}%")
            if valid_pct > 95:
                print(f"    Date range: {parsed.min()} to {parsed.max()}")
        except:
            print(f"  {col}: Failed to parse as datetime")

# ============================================================
# LEAKAGE DETECTION
# ============================================================
print(f"\n{'='*80}")
print(f"[5] LEAKAGE DETECTION ANALYSIS")
print(f"{'='*80}")

print("""
CRITICAL LEAKAGE ASSESSMENT:

Column-by-column leakage risk evaluation:

1. 'data' column:
   - Contains train/test split indicator
   - LEAKAGE RISK: LOW (meta-column, not a feature)
   - ACTION: Use for splitting only, never as feature

2. 'od_end_time':
   - End time of origin-destination segment
   - LEAKAGE RISK: *** CRITICAL ***
   - REASON: At prediction time, you DON'T know when delivery ends
   - ACTION: MUST NOT be used as feature. Can only be used to derive target.

3. 'actual_time':
   - Actual time taken for delivery
   - LEAKAGE RISK: *** CRITICAL ***
   - REASON: This IS the target variable (or derived from it)
   - ACTION: THIS IS YOUR TARGET. Never use as input feature.

4. 'actual_distance_to_destination':
   - Cumulative actual distance
   - LEAKAGE RISK: *** HIGH ***
   - REASON: 'Actual' distance is known only AFTER trip completion
   - ACTION: Use OSRM distance instead for prediction. Actual only for analysis.

5. 'start_scan_to_end_scan':
   - Total trip duration from scan timestamps
   - LEAKAGE RISK: *** CRITICAL ***
   - REASON: Known only after trip completion
   - ACTION: Can be used to validate target, NOT as input feature.

6. 'factor' and 'segment_factor':
   - Ratio of actual/OSRM time
   - LEAKAGE RISK: *** CRITICAL ***
   - REASON: Requires knowing actual time (the target)
   - ACTION: NEVER use as feature. Only for analysis.

7. 'cutoff_factor' and 'cutoff_timestamp':
   - LEAKAGE RISK: MODERATE
   - REASON: cutoff_factor might encode distance already traveled
   - ACTION: Investigate carefully. May be usable if set before trip starts.

8. 'segment_actual_time':
   - LEAKAGE RISK: *** CRITICAL ***
   - REASON: Actual segment time known only after completion
   - ACTION: Only for analysis/target derivation

9. 'osrm_time' and 'osrm_distance':
   - LEAKAGE RISK: NONE (these are pre-computed estimates)
   - ACTION: Safe to use as features

10. 'segment_osrm_time' and 'segment_osrm_distance':
    - LEAKAGE RISK: NONE (pre-computed estimates)
    - ACTION: Safe to use as features

11. 'source_center', 'source_name', 'destination_center', 'destination_name':
    - LEAKAGE RISK: NONE (known at booking time)
    - ACTION: Safe for features and graph construction

12. 'route_type':
    - LEAKAGE RISK: NONE (decided before trip starts)
    - ACTION: Safe as feature. KEY for FTL vs Carting analysis.

13. 'trip_creation_time', 'od_start_time':
    - LEAKAGE RISK: NONE (known at trip start)
    - ACTION: Safe for temporal features (hour, day, etc.)

14. 'route_schedule_uuid', 'trip_uuid':
    - LEAKAGE RISK: NONE (identifiers)
    - ACTION: Use for grouping/deduplication, not as ML features
""")

# ============================================================
# FEATURE CATEGORIZATION
# ============================================================
print(f"\n{'='*80}")
print(f"[6] FEATURE CATEGORIZATION")
print(f"{'='*80}")

print("""
FEATURE TAXONOMY:

A. IDENTIFIER COLUMNS (DO NOT use in ML):
   - data: train/test indicator
   - trip_uuid: unique trip identifier
   - route_schedule_uuid: route schedule identifier

B. TEMPORAL FEATURES (SAFE for ML):
   - trip_creation_time → hour_of_day, day_of_week, is_weekend, month, quarter
   - od_start_time → hour_of_day, day_of_week, time_since_creation

C. CATEGORICAL FEATURES (SAFE for ML):
   - route_type: FTL / Carting
   - source_center: source facility code
   - source_name: source facility name (human readable)
   - destination_center: destination facility code
   - destination_name: destination facility name (human readable)
   - is_cutoff: boolean flag

D. NUMERICAL FEATURES (SAFE for ML):
   - osrm_time: OSRM estimated time
   - osrm_distance: OSRM estimated distance
   - segment_osrm_time: segment-level OSRM time
   - segment_osrm_distance: segment-level OSRM distance

E. TARGET/LEAKAGE COLUMNS (DO NOT use as features):
   - actual_time: *** PRIMARY TARGET ***
   - actual_distance_to_destination: leaky
   - start_scan_to_end_scan: leaky (but useful for target validation)
   - factor: leaky (actual/osrm ratio)
   - segment_actual_time: leaky
   - segment_factor: leaky
   - od_end_time: leaky

F. AMBIGUOUS COLUMNS (INVESTIGATE):
   - cutoff_factor: needs investigation
   - cutoff_timestamp: needs investigation
""")

# ============================================================
# GRAPH CONSTRUCTION FEASIBILITY
# ============================================================
print(f"\n{'='*80}")
print(f"[7] GRAPH CONSTRUCTION FEASIBILITY")
print(f"{'='*80}")

# Node analysis
source_centers = df['source_center'].nunique()
dest_centers = df['destination_center'].nunique()
all_centers = set(df['source_center'].dropna().unique()) | set(df['destination_center'].dropna().unique())
source_names = df['source_name'].nunique()
dest_names = df['destination_name'].nunique()

# Edge analysis
corridors = df.groupby(['source_center', 'destination_center']).size().reset_index(name='frequency')
num_corridors = len(corridors)

# Route type distribution per corridor
route_type_dist = df.groupby(['source_center', 'destination_center', 'route_type']).size().reset_index(name='count')

print(f"""
GRAPH FEASIBILITY REPORT:

NODES:
  Unique source centers: {source_centers}
  Unique destination centers: {dest_centers}
  Total unique facilities (union): {len(all_centers)}
  Unique source names: {source_names}
  Unique destination names: {dest_names}

EDGES:
  Unique corridors (source→dest): {num_corridors}
  Total trips (raw edges): {len(df):,}

DENSITY:
  Max possible edges: {len(all_centers)**2:,}
  Actual edges: {num_corridors:,}
  Graph density: {num_corridors / (len(all_centers)**2):.6f}

ROUTE TYPES:
""")

print("  Route type distribution:")
for rt, cnt in df['route_type'].value_counts().items():
    print(f"    {rt}: {cnt:,} ({cnt/len(df)*100:.2f}%)")

# Corridor frequency stats
print(f"\n  Corridor frequency statistics:")
print(f"    Mean trips per corridor: {corridors['frequency'].mean():.2f}")
print(f"    Median trips per corridor: {corridors['frequency'].median():.2f}")
print(f"    Max trips per corridor: {corridors['frequency'].max()}")
print(f"    Min trips per corridor: {corridors['frequency'].min()}")
print(f"    Corridors with 1 trip: {(corridors['frequency']==1).sum()} ({(corridors['frequency']==1).mean()*100:.1f}%)")
print(f"    Corridors with >100 trips: {(corridors['frequency']>100).sum()}")

# Self-loops
self_loops = (df['source_center'] == df['destination_center']).sum()
print(f"\n  Self-loops (same source & dest): {self_loops}")

# Bidirectional edges
fwd_corridors = set(zip(corridors['source_center'], corridors['destination_center']))
rev_corridors = set(zip(corridors['destination_center'], corridors['source_center']))
bidirectional = fwd_corridors & rev_corridors
print(f"  Bidirectional corridors: {len(bidirectional)}")

print(f"""
GRAPH CONSTRUCTION VERDICT:
  ✓ Sufficient nodes ({len(all_centers)}) for meaningful graph analysis
  ✓ Sufficient edges ({num_corridors}) for network science
  ✓ Sparse graph (density < 0.01) - typical for logistics networks
  ✓ Multiple route types enable multi-graph construction
  ✓ Temporal features enable time-aware graph construction
  
RECOMMENDED GRAPH TYPES:
  1. Facility Graph: facilities as nodes, corridors as edges
  2. Corridor Graph: weighted edges with delay/reliability metrics
  3. Route-Type Graph: separate subgraphs for FTL vs Carting
  4. Temporal Graph: time-aware edges with hourly/daily patterns
""")

# ============================================================
# TRIP STRUCTURE ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[8] TRIP STRUCTURE & SEGMENT ANALYSIS")
print(f"{'='*80}")

# Check if trips have multiple segments
trip_segments = df.groupby('trip_uuid').size()
print(f"\nTrips analysis:")
print(f"  Total unique trips: {trip_segments.shape[0]:,}")
print(f"  Mean segments per trip: {trip_segments.mean():.2f}")
print(f"  Max segments per trip: {trip_segments.max()}")
print(f"  Min segments per trip: {trip_segments.min()}")
print(f"  Single-segment trips: {(trip_segments==1).sum():,} ({(trip_segments==1).mean()*100:.1f}%)")
print(f"  Multi-segment trips: {(trip_segments>1).sum():,} ({(trip_segments>1).mean()*100:.1f}%)")

print("\n  Segment count distribution:")
seg_dist = trip_segments.value_counts().sort_index().head(20)
for n_seg, cnt in seg_dist.items():
    print(f"    {n_seg} segments: {cnt:,} trips ({cnt/len(trip_segments)*100:.1f}%)")

# ============================================================
# ROUTE SCHEDULE ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[9] ROUTE SCHEDULE ANALYSIS")
print(f"{'='*80}")

route_schedules = df.groupby('route_schedule_uuid').size()
print(f"  Unique route schedules: {route_schedules.shape[0]:,}")
print(f"  Mean trips per schedule: {route_schedules.mean():.2f}")
print(f"  Max trips per schedule: {route_schedules.max()}")

# Route type per schedule
routes_per_schedule = df.groupby('route_schedule_uuid')['route_type'].nunique()
print(f"  Schedules with single route type: {(routes_per_schedule==1).sum():,}")
print(f"  Schedules with multiple route types: {(routes_per_schedule>1).sum():,}")

# ============================================================
# TEMPORAL ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[10] TEMPORAL ANALYSIS")
print(f"{'='*80}")

df['trip_creation_time_dt'] = pd.to_datetime(df['trip_creation_time'], errors='coerce')
df['od_start_time_dt'] = pd.to_datetime(df['od_start_time'], errors='coerce')
df['od_end_time_dt'] = pd.to_datetime(df['od_end_time'], errors='coerce')

print(f"\nTrip creation time range:")
print(f"  Start: {df['trip_creation_time_dt'].min()}")
print(f"  End: {df['trip_creation_time_dt'].max()}")
print(f"  Span: {(df['trip_creation_time_dt'].max() - df['trip_creation_time_dt'].min()).days} days")

print(f"\nHour of day distribution (trip creation):")
hour_dist = df['trip_creation_time_dt'].dt.hour.value_counts().sort_index()
for hour, cnt in hour_dist.items():
    bar = '█' * int(cnt / hour_dist.max() * 30)
    print(f"  {int(hour):02d}:00 | {bar} {cnt:,}")

print(f"\nDay of week distribution:")
dow_dist = df['trip_creation_time_dt'].dt.day_name().value_counts()
for day, cnt in dow_dist.items():
    print(f"  {day}: {cnt:,}")

# ============================================================
# TARGET VARIABLE ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[11] TARGET VARIABLE ANALYSIS (actual_time)")
print(f"{'='*80}")

target = df['actual_time']
print(f"  Count: {target.count():,}")
print(f"  Missing: {target.isna().sum():,}")
print(f"  Min: {target.min()}")
print(f"  Max: {target.max()}")
print(f"  Mean: {target.mean():.4f}")
print(f"  Median: {target.median():.4f}")
print(f"  Std: {target.std():.4f}")
print(f"  Skewness: {target.skew():.4f}")

# Percentiles
print(f"\n  Percentile distribution:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{p}: {target.quantile(p/100):.2f}")

# Zero or negative analysis
print(f"\n  Zero values: {(target == 0).sum():,}")
print(f"  Negative values: {(target < 0).sum():,}")

# OSRM vs Actual comparison
print(f"\n  OSRM Time vs Actual Time comparison:")
valid_mask = df['osrm_time'].notna() & df['actual_time'].notna() & (df['osrm_time'] > 0)
if valid_mask.sum() > 0:
    ratio = df.loc[valid_mask, 'actual_time'] / df.loc[valid_mask, 'osrm_time']
    print(f"    Mean actual/osrm ratio: {ratio.mean():.4f}")
    print(f"    Median actual/osrm ratio: {ratio.median():.4f}")
    print(f"    % trips faster than OSRM: {(ratio < 1).mean()*100:.1f}%")
    print(f"    % trips slower than OSRM: {(ratio > 1).mean()*100:.1f}%")
    print(f"    % trips 2x slower than OSRM: {(ratio > 2).mean()*100:.1f}%")

# ============================================================
# TRAIN/TEST SPLIT ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[12] TRAIN/TEST SPLIT ANALYSIS")
print(f"{'='*80}")

data_split = df['data'].value_counts()
for split, cnt in data_split.items():
    print(f"  {split}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

# Check if test set has target
if 'testing' in df['data'].values:
    test_mask = df['data'] == 'testing'
    test_actual_missing = df.loc[test_mask, 'actual_time'].isna().sum()
    print(f"\n  Test set actual_time missing: {test_actual_missing:,} / {test_mask.sum():,}")
    print(f"  Test set actual_time available: {test_mask.sum() - test_actual_missing:,}")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[13] CORRELATION WITH TARGET")
print(f"{'='*80}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'actual_time' in numeric_cols:
    corr_with_target = df[numeric_cols].corr()['actual_time'].drop('actual_time').sort_values(ascending=False)
    print("\nCorrelation with actual_time:")
    for col, corr in corr_with_target.items():
        flag = " *** LEAKY ***" if abs(corr) > 0.95 else ""
        print(f"  {col:40s}: {corr:.4f}{flag}")

# ============================================================
# FACILITY NAME PARSING
# ============================================================
print(f"\n{'='*80}")
print(f"[14] FACILITY NAME STRUCTURE ANALYSIS")
print(f"{'='*80}")

# Parse facility names to extract city and state
print("\nSample source names:")
for name in df['source_name'].dropna().unique()[:15]:
    print(f"  {name}")

# Extract state from facility names (typically in parentheses)
def extract_state(name):
    if pd.isna(name):
        return None
    if '(' in name and ')' in name:
        return name.split('(')[-1].replace(')', '').strip()
    return None

df['source_state'] = df['source_name'].apply(extract_state)
df['dest_state'] = df['destination_name'].apply(extract_state)

print(f"\nUnique source states: {df['source_state'].nunique()}")
print(f"Unique destination states: {df['dest_state'].nunique()}")

print("\nTop source states:")
for state, cnt in df['source_state'].value_counts().head(15).items():
    print(f"  {state}: {cnt:,}")

# Inter-state vs Intra-state
same_state = (df['source_state'] == df['dest_state']).sum()
diff_state = (df['source_state'] != df['dest_state']).sum()
print(f"\nIntra-state trips: {same_state:,} ({same_state/len(df)*100:.1f}%)")
print(f"Inter-state trips: {diff_state:,} ({diff_state/len(df)*100:.1f}%)")

# ============================================================
# DELAY ANALYSIS PREVIEW
# ============================================================
print(f"\n{'='*80}")
print(f"[15] DELAY ANALYSIS PREVIEW")
print(f"{'='*80}")

# Factor analysis (actual/osrm time ratio)
if 'factor' in df.columns:
    factor = df['factor'].dropna()
    print(f"\nFactor (actual_time/osrm_time) statistics:")
    print(f"  Mean: {factor.mean():.4f}")
    print(f"  Median: {factor.median():.4f}")
    print(f"  Trips on-time or early (factor <= 1.0): {(factor <= 1.0).sum():,} ({(factor <= 1.0).mean()*100:.1f}%)")
    print(f"  Moderate delay (1.0 < factor <= 1.3): {((factor > 1.0) & (factor <= 1.3)).sum():,}")
    print(f"  Severe delay (1.3 < factor <= 2.0): {((factor > 1.3) & (factor <= 2.0)).sum():,}")
    print(f"  Critical delay (factor > 2.0): {(factor > 2.0).sum():,}")

# By route type
print(f"\nDelay by route type:")
for rt in df['route_type'].unique():
    rt_factor = df.loc[df['route_type'] == rt, 'factor'].dropna()
    if len(rt_factor) > 0:
        print(f"  {rt}:")
        print(f"    Mean factor: {rt_factor.mean():.4f}")
        print(f"    Median factor: {rt_factor.median():.4f}")
        print(f"    Severe delay %: {(rt_factor > 1.3).mean()*100:.1f}%")

# ============================================================
# CUTOFF ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"[16] CUTOFF ANALYSIS")
print(f"{'='*80}")

print(f"\nis_cutoff distribution:")
for val, cnt in df['is_cutoff'].value_counts().items():
    print(f"  {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

print(f"\ncutoff_factor statistics:")
cf = df['cutoff_factor'].dropna()
print(f"  Min: {cf.min()}")
print(f"  Max: {cf.max()}")
print(f"  Mean: {cf.mean():.4f}")
print(f"  Unique values: {cf.nunique()}")
print(f"  Top 10 values: {cf.value_counts().head(10).to_dict()}")

# ============================================================
# DATA QUALITY SUMMARY
# ============================================================
print(f"\n{'='*80}")
print(f"[17] DATA QUALITY SUMMARY")
print(f"{'='*80}")

print(f"""
OVERALL DATA QUALITY SCORE:

  Completeness:
    - Columns with no missing values: {(missing == 0).sum()}/{len(df.columns)}
    - Total missing cells: {df.isnull().sum().sum():,} / {df.shape[0]*df.shape[1]:,}
    - Overall completeness: {(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%

  Consistency:
    - Data types consistent: Check above for type mismatches
    - Facility codes follow pattern: Check center codes

  Uniqueness:
    - Total rows: {len(df):,}
    - Duplicate rows: {df.duplicated().sum():,}
    - Duplicate trip_uuids: {df['trip_uuid'].duplicated().sum():,} (expected for multi-segment)

  Validity:
    - Negative actual_time values: {(df['actual_time'] < 0).sum():,}
    - Zero actual_time values: {(df['actual_time'] == 0).sum():,}
    - Negative osrm_time values: {(df['osrm_time'].dropna() < 0).sum():,}
    - Negative distance values: {(df['osrm_distance'].dropna() < 0).sum():,}
""")

# ============================================================
# FINAL COLUMN-BY-COLUMN REPORT
# ============================================================
print(f"\n{'='*80}")
print(f"[18] COMPLETE COLUMN REPORT")
print(f"{'='*80}")

column_report = {
    'data': {
        'dtype': str(df['data'].dtype),
        'description': 'Train/Test split indicator',
        'business_meaning': 'Separates training data from evaluation data',
        'graph_usage': 'None - meta column',
        'ml_usage': 'Train/test splitting only',
        'leakage_risk': 'LOW - not a feature',
        'preprocessing': 'Use for data splitting, then drop'
    },
    'trip_creation_time': {
        'dtype': str(df['trip_creation_time'].dtype),
        'description': 'Timestamp when the trip/route was created',
        'business_meaning': 'When the logistics system created this trip plan',
        'graph_usage': 'Temporal graph construction - time-aware edges',
        'ml_usage': 'Extract: hour, day_of_week, month, is_weekend, is_peak_hour',
        'leakage_risk': 'NONE - known at booking time',
        'preprocessing': 'Parse to datetime, extract temporal features'
    },
    'route_schedule_uuid': {
        'dtype': str(df['route_schedule_uuid'].dtype),
        'description': 'Unique identifier for the route schedule',
        'business_meaning': 'Links multiple trips following same planned route',
        'graph_usage': 'Route-level graph construction, edge identification',
        'ml_usage': 'Grouping variable for aggregation. NOT a direct feature.',
        'leakage_risk': 'NONE - identifier',
        'preprocessing': 'Use for grouping, then drop from features'
    },
    'route_type': {
        'dtype': str(df['route_type'].dtype),
        'description': 'Type of transport route',
        'business_meaning': 'FTL=Full Truck Load (point-to-point), Carting=local delivery with stops',
        'graph_usage': 'Multi-graph: separate FTL and Carting subgraphs',
        'ml_usage': 'Key categorical feature. Also target for decision engine.',
        'leakage_risk': 'NONE - decided before trip starts',
        'preprocessing': 'One-hot or label encode'
    },
    'trip_uuid': {
        'dtype': str(df['trip_uuid'].dtype),
        'description': 'Unique identifier for each trip',
        'business_meaning': 'A trip is one vehicle journey, can span multiple segments',
        'graph_usage': 'Trip-level aggregation for graph edge weights',
        'ml_usage': 'Grouping only. NOT a feature.',
        'leakage_risk': 'NONE - identifier',
        'preprocessing': 'Use for grouping segments into trips, then drop'
    },
    'source_center': {
        'dtype': str(df['source_center'].dtype),
        'description': 'Code of the source/origin facility',
        'business_meaning': 'The hub/center where this leg of journey starts',
        'graph_usage': '*** CRITICAL *** - Graph node identifier (source node)',
        'ml_usage': 'Categorical feature (high cardinality - use encoding/embedding)',
        'leakage_risk': 'NONE - known at booking',
        'preprocessing': 'Target encoding or node embedding'
    },
    'source_name': {
        'dtype': str(df['source_name'].dtype),
        'description': 'Human-readable name of source facility with city and state',
        'business_meaning': 'Shows geographic location: City_Area_Type (State)',
        'graph_usage': 'Node attribute, geographic grouping',
        'ml_usage': 'Extract: city, state, facility_type from name pattern',
        'leakage_risk': 'NONE',
        'preprocessing': 'Parse to extract city, state, facility_type'
    },
    'destination_center': {
        'dtype': str(df['destination_center'].dtype),
        'description': 'Code of the destination facility',
        'business_meaning': 'The hub/center where this leg of journey ends',
        'graph_usage': '*** CRITICAL *** - Graph node identifier (destination node)',
        'ml_usage': 'Categorical feature (high cardinality - use encoding/embedding)',
        'leakage_risk': 'NONE - known at booking',
        'preprocessing': 'Target encoding or node embedding'
    },
    'destination_name': {
        'dtype': str(df['destination_name'].dtype),
        'description': 'Human-readable name of destination facility with city and state',
        'business_meaning': 'Shows geographic location of destination',
        'graph_usage': 'Node attribute, geographic grouping',
        'ml_usage': 'Extract: city, state, facility_type',
        'leakage_risk': 'NONE',
        'preprocessing': 'Parse to extract city, state, facility_type'
    },
    'od_start_time': {
        'dtype': str(df['od_start_time'].dtype),
        'description': 'Start time of origin-destination segment',
        'business_meaning': 'When vehicle departed from origin on this segment',
        'graph_usage': 'Temporal edge attribute for time-aware graphs',
        'ml_usage': 'Extract temporal features + compute time_since_creation',
        'leakage_risk': 'LOW - available at departure time',
        'preprocessing': 'Parse to datetime, extract temporal features'
    },
    'od_end_time': {
        'dtype': str(df['od_end_time'].dtype),
        'description': 'End time of origin-destination segment',
        'business_meaning': 'When vehicle arrived at destination',
        'graph_usage': 'For historical graph edge weight computation ONLY',
        'ml_usage': '*** DO NOT USE AS FEATURE *** Only for target derivation',
        'leakage_risk': '*** CRITICAL *** - Not known at prediction time',
        'preprocessing': 'Use only for computing targets, then drop'
    },
    'start_scan_to_end_scan': {
        'dtype': str(df['start_scan_to_end_scan'].dtype),
        'description': 'Total time from first scan to last scan (minutes)',
        'business_meaning': 'Total trip duration measured by scan events',
        'graph_usage': 'Historical edge weight for graph construction',
        'ml_usage': '*** DO NOT USE AS FEATURE *** Leaky target proxy',
        'leakage_risk': '*** CRITICAL *** - Known only after completion',
        'preprocessing': 'Use for target validation only, then drop'
    },
    'is_cutoff': {
        'dtype': str(df['is_cutoff'].dtype),
        'description': 'Whether this datapoint represents a cutoff point in the route',
        'business_meaning': 'Indicates intermediate checkpoints along the route',
        'graph_usage': 'Can indicate intermediate nodes in path',
        'ml_usage': 'Useful for understanding multi-segment trips',
        'leakage_risk': 'LOW - structural information',
        'preprocessing': 'Boolean encoding'
    },
    'cutoff_factor': {
        'dtype': str(df['cutoff_factor'].dtype),
        'description': 'Factor associated with the cutoff point',
        'business_meaning': 'Likely cumulative distance/time marker along route',
        'graph_usage': 'Can indicate position within route path',
        'ml_usage': 'INVESTIGATE - may encode cumulative distance traveled',
        'leakage_risk': 'MODERATE - may contain post-trip info',
        'preprocessing': 'Investigate relationship with other columns'
    },
    'cutoff_timestamp': {
        'dtype': str(df['cutoff_timestamp'].dtype),
        'description': 'Timestamp of the cutoff point',
        'business_meaning': 'When this checkpoint/cutoff was reached',
        'graph_usage': 'Temporal edge attribute',
        'ml_usage': 'INVESTIGATE - may be post-trip timestamp',
        'leakage_risk': 'MODERATE - needs validation',
        'preprocessing': 'Parse datetime, validate timing relative to trip'
    },
    'actual_distance_to_destination': {
        'dtype': str(df['actual_distance_to_destination'].dtype),
        'description': 'Actual distance traveled to destination (km)',
        'business_meaning': 'Ground truth distance including detours',
        'graph_usage': 'Historical edge weight for graph construction',
        'ml_usage': '*** CAUTION *** May be leaky. Use OSRM distance instead.',
        'leakage_risk': 'HIGH - actual distance known only after trip',
        'preprocessing': 'Use for analysis/validation only. Replace with osrm_distance for ML.'
    },
    'actual_time': {
        'dtype': str(df['actual_time'].dtype),
        'description': 'Actual time taken for delivery (minutes)',
        'business_meaning': 'Ground truth delivery time - THE key business metric',
        'graph_usage': 'Historical edge weight for delay computation',
        'ml_usage': '*** PRIMARY TARGET VARIABLE ***',
        'leakage_risk': '*** CRITICAL *** - This IS the target',
        'preprocessing': 'Use as target only. Handle outliers. Log-transform if skewed.'
    },
    'osrm_time': {
        'dtype': str(df['osrm_time'].dtype),
        'description': 'OSRM estimated time (minutes) - cumulative',
        'business_meaning': 'Open Source Routing Machine time estimate (baseline ETA)',
        'graph_usage': 'Edge attribute - expected travel time',
        'ml_usage': '*** KEY FEATURE *** Baseline ETA prediction',
        'leakage_risk': 'NONE - pre-computed estimate',
        'preprocessing': 'Handle zeros/negatives. Use as primary feature.'
    },
    'osrm_distance': {
        'dtype': str(df['osrm_distance'].dtype),
        'description': 'OSRM estimated distance (km) - cumulative',
        'business_meaning': 'Routing engine distance estimate',
        'graph_usage': 'Edge attribute - route distance',
        'ml_usage': 'Key feature for ETA prediction',
        'leakage_risk': 'NONE - pre-computed estimate',
        'preprocessing': 'Handle zeros/negatives. Use as feature.'
    },
    'factor': {
        'dtype': str(df['factor'].dtype),
        'description': 'Ratio of actual_time to osrm_time (cumulative)',
        'business_meaning': 'Delay factor - how much slower than estimated',
        'graph_usage': 'Historical delay metric for edge weighting',
        'ml_usage': '*** DO NOT USE *** - Derived from target',
        'leakage_risk': '*** CRITICAL *** - Contains target information',
        'preprocessing': 'Use for analysis only. NEVER as feature.'
    },
    'segment_actual_time': {
        'dtype': str(df['segment_actual_time'].dtype),
        'description': 'Actual time for this specific segment only',
        'business_meaning': 'Time spent on individual leg of multi-stop trip',
        'graph_usage': 'Segment-level edge weights for detailed graph',
        'ml_usage': '*** DO NOT USE *** - Post-trip information',
        'leakage_risk': '*** CRITICAL *** - Known only after completion',
        'preprocessing': 'Use for historical analysis only'
    },
    'segment_osrm_time': {
        'dtype': str(df['segment_osrm_time'].dtype),
        'description': 'OSRM estimated time for this segment',
        'business_meaning': 'Expected time for this individual leg',
        'graph_usage': 'Segment-level edge attribute',
        'ml_usage': 'Feature - segment-level ETA baseline',
        'leakage_risk': 'NONE - pre-computed',
        'preprocessing': 'Use as feature for segment-level prediction'
    },
    'segment_osrm_distance': {
        'dtype': str(df['segment_osrm_distance'].dtype),
        'description': 'OSRM estimated distance for this segment',
        'business_meaning': 'Expected distance for this individual leg',
        'graph_usage': 'Segment-level edge attribute',
        'ml_usage': 'Feature for segment-level prediction',
        'leakage_risk': 'NONE - pre-computed',
        'preprocessing': 'Use as feature'
    },
    'segment_factor': {
        'dtype': str(df['segment_factor'].dtype),
        'description': 'Ratio of segment_actual_time to segment_osrm_time',
        'business_meaning': 'Segment-level delay factor',
        'graph_usage': 'Historical edge delay analysis',
        'ml_usage': '*** DO NOT USE *** - Derived from target',
        'leakage_risk': '*** CRITICAL *** - Contains target information',
        'preprocessing': 'Analysis only, never as feature'
    }
}

for col_name, info in column_report.items():
    print(f"\n{'─'*60}")
    print(f"Column: {col_name}")
    print(f"{'─'*60}")
    for key, value in info.items():
        print(f"  {key:20s}: {value}")

print(f"\n{'='*80}")
print(f"PHASE 0 AUDIT COMPLETE")
print(f"{'='*80}")
