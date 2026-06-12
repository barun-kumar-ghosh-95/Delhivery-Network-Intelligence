import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
analysis_dir = os.path.join(processed_dir, 'analysis')

os.makedirs(analysis_dir, exist_ok=True)

print("Generating mock datasets for dashboard...")

# 1. model_comparison.csv
model_comp = pd.DataFrame([
    {"model": "Linear Regression", "MAE": 88.45, "RMSE": 112.30, "MAPE": 28.4, "R2": 0.65, "Acc@10%": 25.1, "Acc@15%": 42.1, "Acc@20%": 55.4},
    {"model": "Random Forest", "MAE": 76.12, "RMSE": 98.40, "MAPE": 22.1, "R2": 0.78, "Acc@10%": 38.4, "Acc@15%": 61.4, "Acc@20%": 73.2},
    {"model": "XGBoost", "MAE": 72.31, "RMSE": 92.15, "MAPE": 19.8, "R2": 0.83, "Acc@10%": 45.2, "Acc@15%": 68.2, "Acc@20%": 80.1},
    {"model": "Graph + XGBoost", "MAE": 64.00, "RMSE": 84.60, "MAPE": 15.2, "R2": 0.88, "Acc@10%": 58.9, "Acc@15%": 76.8, "Acc@20%": 87.5},
    {"model": "GraphSAGE-Approx (LightGBM)", "MAE": 62.09, "RMSE": 81.12, "MAPE": 14.1, "R2": 0.91, "Acc@10%": 65.4, "Acc@15%": 81.4, "Acc@20%": 92.1},
])
model_comp.to_csv(os.path.join(processed_dir, 'model_comparison.csv'), index=False)

# 2. corridor_stats.csv
np.random.seed(42)
n_corridors = 200
corridor_stats = pd.DataFrame({
    "source_center": [f"Hub_{i}" for i in np.random.randint(1, 50, n_corridors)],
    "destination_center": [f"Hub_{i}" for i in np.random.randint(51, 100, n_corridors)],
    "delay_ratio_mean": np.random.uniform(0.8, 3.5, n_corridors),
    "trip_count": np.random.randint(10, 500, n_corridors),
    "sla_breach_pct": np.random.uniform(5, 40, n_corridors),
    "corridor_reliability": np.random.uniform(60, 95, n_corridors)
})
corridor_stats['delay_class'] = pd.cut(corridor_stats['delay_ratio_mean'], bins=[0, 1.2, 1.5, 2.0, 10], labels=['Healthy', 'Moderate', 'Severe', 'Critical'])
corridor_stats.to_csv(os.path.join(processed_dir, 'corridor_stats.csv'), index=False)
corridor_stats.to_csv(os.path.join(analysis_dir, 'classified_corridors.csv'), index=False)

# 3. facility_stats.csv
facility_stats = pd.DataFrame({
    "facility_id": [f"Hub_{i}" for i in range(1, 101)],
    "impact_score_normalized": np.random.uniform(10, 100, 100),
    "delay_multiplier": np.random.uniform(1.0, 3.0, 100),
    "bottleneck_class": np.random.choice(['Low', 'Medium', 'High', 'Critical'], 100)
})
facility_stats.to_csv(os.path.join(processed_dir, 'facility_stats.csv'), index=False)
facility_stats.to_csv(os.path.join(analysis_dir, 'facility_impact_scores.csv'), index=False)
facility_stats.to_csv(os.path.join(analysis_dir, 'bottleneck_hubs.csv'), index=False)

# 4. bottleneck_corridors.csv
corridor_stats['risk_score'] = corridor_stats['delay_ratio_mean'] * 30
corridor_stats['severity'] = corridor_stats['delay_class']
corridor_stats.to_csv(os.path.join(analysis_dir, 'bottleneck_corridors.csv'), index=False)

# 5. test_trips_featured.csv
test_trips = pd.DataFrame({
    "actual_time": np.random.uniform(100, 1500, 1000),
    "osrm_time": np.random.uniform(90, 1400, 1000),
    "osrm_distance": np.random.uniform(50, 1500, 1000),
    "num_segments": np.random.randint(1, 10, 1000),
    "delay": np.random.uniform(0, 300, 1000),
    "source_center": [f"Hub_{i}" for i in np.random.randint(1, 100, 1000)],
    "destination_center": [f"Hub_{i}" for i in np.random.randint(1, 100, 1000)],
    "route_type": np.random.choice(['FTL', 'Carting'], 1000),
    "ftl_prob": np.random.uniform(0.1, 0.9, 1000),
    "is_ftl": np.random.choice([0, 1], 1000)
})
test_trips.to_csv(os.path.join(processed_dir, 'test_trips_featured.csv'), index=False)
test_trips.to_csv(os.path.join(processed_dir, 'train_trips_featured.csv'), index=False)

print("Mock datasets generated successfully.")
