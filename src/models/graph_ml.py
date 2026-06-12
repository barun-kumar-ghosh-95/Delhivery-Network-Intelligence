"""
Graph ML Models
===============
Node2Vec + XGBoost (primary approach for limited data).
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor


GRAPH_FEATURE_PREFIXES = ['src_emb_', 'dst_emb_', 'src_graph_', 'dst_graph_']


def get_graph_enhanced_features(df, base_feature_cols):
    """Get combined base + graph features."""
    graph_cols = [c for c in df.columns if any(c.startswith(p) for p in GRAPH_FEATURE_PREFIXES)]
    all_cols = base_feature_cols + graph_cols
    available = [c for c in all_cols if c in df.columns]
    return available


def train_graph_ml_models(train_df, test_df, base_feature_cols, save_dir=None):
    """Train graph-enhanced ML models."""
    from src.models.evaluation import compute_all_metrics, print_metrics
    
    print("\n" + "="*80)
    print("PHASE 8: GRAPH ML MODELS")
    print("="*80)
    
    all_feature_cols = get_graph_enhanced_features(train_df, base_feature_cols)
    graph_only_cols = [c for c in all_feature_cols if any(c.startswith(p) for p in GRAPH_FEATURE_PREFIXES)]
    
    print(f"[GraphML] Base features: {len(base_feature_cols)}")
    print(f"[GraphML] Graph features: {len(graph_only_cols)}")
    print(f"[GraphML] Total features: {len(all_feature_cols)}")
    
    X_train = train_df[all_feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['actual_time']
    X_test = test_df[all_feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df['actual_time']
    
    all_metrics = []
    models = {}
    
    # --- Node2Vec + XGBoost ---
    print("\n[GraphML] Training Graph-Enhanced XGBoost...")
    try:
        import xgboost as xgb
        graph_xgb = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0,
        )
        graph_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        pred = graph_xgb.predict(X_test).clip(min=0)
        metrics = compute_all_metrics(y_test, pred, "Graph+XGBoost")
        print_metrics(metrics)
        all_metrics.append(metrics)
        models['graph_xgboost'] = graph_xgb
    except ImportError:
        print("[GraphML] XGBoost not available, using GradientBoosting")
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test).clip(min=0)
        metrics = compute_all_metrics(y_test, pred, "Graph+GradientBoosting")
        print_metrics(metrics)
        all_metrics.append(metrics)
        models['graph_gradient_boosting'] = gb
    
    # --- Graph + LightGBM ---
    print("\n[GraphML] Training Graph-Enhanced LightGBM...")
    try:
        import lightgbm as lgb
        graph_lgb = lgb.LGBMRegressor(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        graph_lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        pred = graph_lgb.predict(X_test).clip(min=0)
        metrics = compute_all_metrics(y_test, pred, "Graph+LightGBM")
        print_metrics(metrics)
        all_metrics.append(metrics)
        models['graph_lightgbm'] = graph_lgb
    except ImportError:
        print("[GraphML] LightGBM not available, skipping")
    
    # --- Graph + CatBoost ---
    print("\n[GraphML] Training Graph-Enhanced CatBoost...")
    try:
        from catboost import CatBoostRegressor
        graph_cat = CatBoostRegressor(
            iterations=500, depth=8, learning_rate=0.05,
            random_seed=42, verbose=0,
        )
        graph_cat.fit(X_train, y_train, eval_set=(X_test, y_test))
        pred = graph_cat.predict(X_test).clip(min=0)
        metrics = compute_all_metrics(y_test, pred, "Graph+CatBoost")
        print_metrics(metrics)
        all_metrics.append(metrics)
        models['graph_catboost'] = graph_cat
    except ImportError:
        print("[GraphML] CatBoost not available, skipping")
    
    # --- Residual Model (predict actual - osrm correction) ---
    print("\n[GraphML] Training Residual Model (predict OSRM correction)...")
    if 'osrm_time' in train_df.columns:
        y_train_residual = y_train - train_df['osrm_time']
        y_test_residual = y_test - test_df['osrm_time']
        
        try:
            import xgboost as xgb
            residual_model = xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                n_jobs=-1, random_state=42, verbosity=0,
            )
            residual_model.fit(X_train, y_train_residual, eval_set=[(X_test, y_test_residual)], verbose=False)
            residual_pred = residual_model.predict(X_test)
            final_pred = (test_df['osrm_time'] + residual_pred).clip(lower=0)
            metrics = compute_all_metrics(y_test, final_pred, "Residual+XGBoost")
            print_metrics(metrics)
            all_metrics.append(metrics)
            models['residual_xgboost'] = residual_model
        except ImportError:
            gb = GradientBoostingRegressor(n_estimators=300, max_depth=6, random_state=42)
            gb.fit(X_train, y_train_residual)
            residual_pred = gb.predict(X_test)
            final_pred = (test_df['osrm_time'] + residual_pred).clip(lower=0)
            metrics = compute_all_metrics(y_test, final_pred, "Residual+GB")
            print_metrics(metrics)
            all_metrics.append(metrics)
            models['residual_gb'] = gb
    
    # Save models
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, model in models.items():
            path = os.path.join(save_dir, f'{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        with open(os.path.join(save_dir, 'graph_feature_cols.pkl'), 'wb') as f:
            pickle.dump(all_feature_cols, f)
    
    comparison = pd.DataFrame(all_metrics).sort_values('R2', ascending=False)
    print("\n" + "="*80)
    print("GRAPH ML MODEL COMPARISON")
    print("="*80)
    print(comparison.to_string(index=False))
    
    return {
        'models': models,
        'metrics': all_metrics,
        'comparison': comparison,
        'feature_cols': all_feature_cols,
    }
