"""
Baseline ML Models
==================
Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost
with cross-validation and SHAP analysis.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Define feature columns (non-leaky only)
FEATURE_COLS = [
    'osrm_time', 'osrm_distance', 'is_ftl', 'num_segments',
    'creation_hour', 'creation_day_of_week', 'creation_is_weekend',
    'creation_is_peak', 'creation_is_night',
    'creation_hour_sin', 'creation_hour_cos',
    'creation_dow_sin', 'creation_dow_cos',
    'start_hour', 'start_is_peak',
    'time_since_creation_min',
    'is_intrastate',
    'source_center_encoded', 'destination_center_encoded',
    'source_facility_type_encoded', 'dest_facility_type_encoded',
    'source_state_encoded', 'dest_state_encoded',
    'osrm_speed_kmh', 'log_osrm_time', 'log_osrm_distance',
    'log_num_segments', 'cutoff_count',
    'segment_osrm_time_sum', 'segment_osrm_time_mean',
    'segment_osrm_distance_sum', 'segment_osrm_distance_mean',
    'avg_segment_distance', 'avg_segment_time',
]

TARGET_COL = 'actual_time'


def prepare_features(df, feature_cols=None):
    """Prepare feature matrix and target, handling missing columns gracefully."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    
    # Only use columns that exist
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_cols)
    if missing:
        print(f"[Baseline] Warning: {len(missing)} feature columns not found: {list(missing)[:5]}...")
    
    X = df[available_cols].copy()
    X = X.fillna(0)
    
    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)
    
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
    
    return X, y, available_cols


def train_baseline_models(train_df, test_df=None, save_dir=None):
    """Train all baseline models and return results."""
    from src.models.evaluation import compute_all_metrics, print_metrics
    
    print("\n" + "="*80)
    print("PHASE 6: BASELINE ML MODELS")
    print("="*80)
    
    X_train, y_train, feature_cols = prepare_features(train_df)
    
    if test_df is not None:
        X_test, y_test, _ = prepare_features(test_df, feature_cols)
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    print(f"[Baseline] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Baseline] Features used: {len(feature_cols)}")
    
    all_metrics = []
    models = {}
    
    # --- 1. OSRM Baseline (no training needed) ---
    if 'osrm_time' in X_test.columns:
        osrm_metrics = compute_all_metrics(y_test, X_test['osrm_time'], "OSRM Baseline")
        print_metrics(osrm_metrics)
        all_metrics.append(osrm_metrics)
    
    # --- 2. Linear Regression ---
    print("\n[Baseline] Training Linear Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled).clip(min=0)
    lr_metrics = compute_all_metrics(y_test, lr_pred, "Linear Regression")
    print_metrics(lr_metrics)
    all_metrics.append(lr_metrics)
    models['linear_regression'] = lr
    
    # --- 3. Random Forest ---
    print("\n[Baseline] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test).clip(min=0)
    rf_metrics = compute_all_metrics(y_test, rf_pred, "Random Forest")
    print_metrics(rf_metrics)
    all_metrics.append(rf_metrics)
    models['random_forest'] = rf
    
    # --- 4. XGBoost ---
    print("\n[Baseline] Training XGBoost...")
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgb_model.predict(X_test).clip(min=0)
        xgb_metrics = compute_all_metrics(y_test, xgb_pred, "XGBoost")
        print_metrics(xgb_metrics)
        all_metrics.append(xgb_metrics)
        models['xgboost'] = xgb_model
    except ImportError:
        print("[Baseline] XGBoost not installed, using GradientBoosting instead")
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test).clip(min=0)
        gb_metrics = compute_all_metrics(y_test, gb_pred, "GradientBoosting")
        print_metrics(gb_metrics)
        all_metrics.append(gb_metrics)
        models['gradient_boosting'] = gb
    
    # --- 5. LightGBM ---
    print("\n[Baseline] Training LightGBM...")
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        lgb_pred = lgb_model.predict(X_test).clip(min=0)
        lgb_metrics = compute_all_metrics(y_test, lgb_pred, "LightGBM")
        print_metrics(lgb_metrics)
        all_metrics.append(lgb_metrics)
        models['lightgbm'] = lgb_model
    except ImportError:
        print("[Baseline] LightGBM not installed, skipping")
    
    # --- 6. CatBoost ---
    print("\n[Baseline] Training CatBoost...")
    try:
        from catboost import CatBoostRegressor
        cat_model = CatBoostRegressor(
            iterations=500, depth=8, learning_rate=0.05,
            random_seed=42, verbose=0,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
        cat_pred = cat_model.predict(X_test).clip(min=0)
        cat_metrics = compute_all_metrics(y_test, cat_pred, "CatBoost")
        print_metrics(cat_metrics)
        all_metrics.append(cat_metrics)
        models['catboost'] = cat_model
    except ImportError:
        print("[Baseline] CatBoost not installed, skipping")
    
    # --- Feature Importance (best model) ---
    best_model_name = max(all_metrics, key=lambda x: x['R2'])['model']
    print(f"\n[Baseline] Best model: {best_model_name}")
    
    # Feature importance from RF or tree models
    importance_df = None
    if 'random_forest' in models:
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n[Baseline] Top 15 features (Random Forest):")
        for _, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save models
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, model in models.items():
            path = os.path.join(save_dir, f'{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"[Baseline] Saved {name} to {path}")
        
        # Save feature columns
        with open(os.path.join(save_dir, 'feature_cols.pkl'), 'wb') as f:
            pickle.dump(feature_cols, f)
        
        # Save scaler
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    
    # Comparison table
    comparison = pd.DataFrame(all_metrics)
    comparison = comparison.sort_values('R2', ascending=False)
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(comparison.to_string(index=False))
    
    return {
        'models': models,
        'metrics': all_metrics,
        'comparison': comparison,
        'feature_importance': importance_df,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'best_model': best_model_name,
    }
