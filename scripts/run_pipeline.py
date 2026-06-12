"""
Master Pipeline Script
======================
Runs all phases in order, produces all artifacts for the dashboard.
"""

import sys
import os
import io
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

# ============================================================
# PHASE 3: DATA PIPELINE + GRAPH CONSTRUCTION
# ============================================================
print("\n" + "="*80)
print("PHASE 3: DATA PIPELINE + GRAPH CONSTRUCTION")
print("="*80)

from src.data.loader import load_and_prepare, split_train_test
from src.data.cleaner import clean_data
from src.data.aggregator import aggregate_trips, get_corridor_stats, get_facility_stats

# Load and clean
data_path = os.path.join(PROJECT_ROOT, 'delivery_data.csv')
df = load_and_prepare(data_path)
df = clean_data(df)

# Split BEFORE aggregation to prevent leakage
train_raw, test_raw = split_train_test(df)

# Get corridor and facility stats from TRAINING DATA ONLY
corridor_stats = get_corridor_stats(train_raw)
facility_stats = get_facility_stats(train_raw)

# Aggregate to trip level
train_trips = aggregate_trips(train_raw)
test_trips = aggregate_trips(test_raw)

# Save processed data
processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)
train_trips.to_csv(os.path.join(processed_dir, 'train_trips.csv'), index=False)
test_trips.to_csv(os.path.join(processed_dir, 'test_trips.csv'), index=False)
corridor_stats.to_csv(os.path.join(processed_dir, 'corridor_stats.csv'), index=False)
facility_stats.to_csv(os.path.join(processed_dir, 'facility_stats.csv'), index=False)
print(f"[Pipeline] Saved processed data to {processed_dir}")

# Build graphs
from src.graph.builder import build_all_graphs, save_graphs

graphs = build_all_graphs(train_raw, corridor_stats, facility_stats)
graph_dir = os.path.join(PROJECT_ROOT, 'data', 'graphs')
save_graphs(graphs, graph_dir)

# ============================================================
# PHASE 4-5: NETWORK SCIENCE + DELAY ANALYSIS
# ============================================================
print("\n" + "="*80)
print("PHASE 4-5: NETWORK SCIENCE + DELAY ANALYSIS")
print("="*80)

from src.graph.analysis import run_full_analysis
from src.graph.delay_analysis import run_delay_analysis

# Network science
G = graphs['facility']
analysis_results = run_full_analysis(G, facility_stats)

# Save analysis results
analysis_dir = os.path.join(processed_dir, 'analysis')
os.makedirs(analysis_dir, exist_ok=True)
analysis_results['impact_scores'].to_csv(os.path.join(analysis_dir, 'facility_impact_scores.csv'), index=False)

with open(os.path.join(analysis_dir, 'network_summary.json'), 'w') as f:
    json.dump(analysis_results['summary'], f, indent=2, default=str)

# Delay analysis
delay_results = run_delay_analysis(corridor_stats, facility_stats)
delay_results['classified_corridors'].to_csv(os.path.join(analysis_dir, 'classified_corridors.csv'), index=False)

with open(os.path.join(analysis_dir, 'delay_summary.json'), 'w') as f:
    json.dump(delay_results['summary'], f, indent=2, default=str)

# ============================================================
# PHASE 6: FEATURE ENGINEERING + BASELINE MODELS
# ============================================================
print("\n" + "="*80)
print("PHASE 6: FEATURE ENGINEERING")
print("="*80)

from src.features.temporal import extract_temporal_features
from src.features.spatial import encode_facilities, add_spatial_features
from src.features.route import add_route_features

# Apply feature engineering to both train and test
for trip_df, name in [(train_trips, 'train'), (test_trips, 'test')]:
    pass  # We'll modify in place below

train_trips = extract_temporal_features(train_trips)
train_trips = encode_facilities(train_trips)
train_trips = add_spatial_features(train_trips)
train_trips = add_route_features(train_trips)

test_trips = extract_temporal_features(test_trips)
test_trips = encode_facilities(test_trips)
test_trips = add_spatial_features(test_trips)
test_trips = add_route_features(test_trips)

# Train baseline models
from src.models.baseline import train_baseline_models

model_dir = os.path.join(PROJECT_ROOT, 'data', 'models')
baseline_results = train_baseline_models(train_trips, test_trips, save_dir=model_dir)

# ============================================================
# PHASE 7-8: GRAPH EMBEDDINGS + GRAPH ML
# ============================================================
print("\n" + "="*80)
print("PHASE 7-8: GRAPH EMBEDDINGS + GRAPH ML")
print("="*80)

from src.graph.embeddings import compute_node2vec_embeddings, compute_graph_statistics
from src.features.graph_features import merge_graph_features
from src.models.graph_ml import train_graph_ml_models

# Compute embeddings
embeddings_df = compute_node2vec_embeddings(G, dimensions=32, num_walks=50, walk_length=20)
graph_stats_df = compute_graph_statistics(G)

# Save embeddings
embeddings_df.to_csv(os.path.join(processed_dir, 'node_embeddings.csv'), index=False)
graph_stats_df.to_csv(os.path.join(processed_dir, 'graph_stats.csv'), index=False)

# Merge graph features
train_trips = merge_graph_features(train_trips, embeddings_df, graph_stats_df)
test_trips = merge_graph_features(test_trips, embeddings_df, graph_stats_df)

# Train graph ML models
graph_ml_results = train_graph_ml_models(
    train_trips, test_trips, 
    baseline_results['feature_cols'],
    save_dir=model_dir
)

# ============================================================
# PHASE 9-11: BUSINESS ENGINES
# ============================================================
print("\n" + "="*80)
print("PHASE 9-11: BUSINESS ENGINES")
print("="*80)

from src.models.decision_engine import (
    train_ftl_carting_classifier,
    compute_bottleneck_scores,
    compute_business_metrics
)

# Business metrics (use best model predictions)
best_model_key = None
for key in ['graph_xgboost', 'graph_gradient_boosting', 'xgboost', 'gradient_boosting', 'random_forest']:
    if key in {**baseline_results['models'], **graph_ml_results['models']}:
        best_model_key = key
        break

all_models = {**baseline_results['models'], **graph_ml_results['models']}
if best_model_key and best_model_key in all_models:
    best_model = all_models[best_model_key]
    if best_model_key.startswith('graph_'):
        feat_cols = graph_ml_results['feature_cols']
    else:
        feat_cols = baseline_results['feature_cols']
    
    avail_cols = [c for c in feat_cols if c in test_trips.columns]
    X_test_final = test_trips[avail_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_pred_final = best_model.predict(X_test_final).clip(min=0)
    
    osrm_pred = test_trips['osrm_time'] if 'osrm_time' in test_trips.columns else None
    business_metrics = compute_business_metrics(
        test_trips['actual_time'], y_pred_final, osrm_pred
    )
else:
    print("[Pipeline] No best model found, using OSRM as baseline")
    business_metrics = compute_business_metrics(
        test_trips['actual_time'], test_trips['osrm_time']
    )

# Bottleneck scoring
bottleneck_results = compute_bottleneck_scores(
    analysis_results['impact_scores'], corridor_stats
)

# FTL vs Carting classifier
ftl_results = train_ftl_carting_classifier(
    train_trips, test_trips,
    baseline_results['feature_cols'],
    save_dir=model_dir
)

# ============================================================
# SAVE ALL RESULTS FOR DASHBOARD
# ============================================================
print("\n" + "="*80)
print("SAVING ALL RESULTS FOR DASHBOARD")
print("="*80)

dashboard_data = {
    'baseline_comparison': baseline_results['comparison'].to_dict('records'),
    'graph_ml_comparison': graph_ml_results['comparison'].to_dict('records'),
    'business_metrics': business_metrics,
    'network_summary': analysis_results['summary'],
    'delay_summary': delay_results['summary'],
    'ftl_accuracy': ftl_results['accuracy'],
    'best_model': best_model_key,
}

with open(os.path.join(processed_dir, 'dashboard_data.json'), 'w') as f:
    json.dump(dashboard_data, f, indent=2, default=str)

# Save feature-engineered data for dashboard
train_trips.to_csv(os.path.join(processed_dir, 'train_trips_featured.csv'), index=False)
test_trips.to_csv(os.path.join(processed_dir, 'test_trips_featured.csv'), index=False)

# Save bottleneck results
bottleneck_results['hub_scores'].to_csv(os.path.join(analysis_dir, 'bottleneck_hubs.csv'), index=False)
bottleneck_results['corridor_scores'].to_csv(os.path.join(analysis_dir, 'bottleneck_corridors.csv'), index=False)

# Combined model comparison
all_metrics = baseline_results['metrics'] + graph_ml_results['metrics']
combined_comparison = pd.DataFrame(all_metrics).sort_values('R2', ascending=False)
combined_comparison.to_csv(os.path.join(processed_dir, 'model_comparison.csv'), index=False)

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE")
print("="*80)
print(f"\nBest model: {best_model_key}")
print(f"Files saved to: {processed_dir}")
print(f"\nTo launch dashboard:")
print(f"  cd {PROJECT_ROOT}")
print(f"  streamlit run dashboard/app.py")
