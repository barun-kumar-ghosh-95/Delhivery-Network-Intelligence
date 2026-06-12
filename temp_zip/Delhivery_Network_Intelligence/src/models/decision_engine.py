"""
Decision Engine Module
======================
FTL vs Carting recommendation, bottleneck impact scoring, business metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os


def train_ftl_carting_classifier(train_df, test_df, feature_cols, save_dir=None):
    """Train FTL vs Carting route type classifier."""
    print("\n" + "="*80)
    print("PHASE 11: FTL vs CARTING DECISION ENGINE")
    print("="*80)
    
    available = [c for c in feature_cols if c in train_df.columns and c != 'is_ftl']
    
    X_train = train_df[available].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['is_ftl'] if 'is_ftl' in train_df.columns else (train_df['route_type'] == 'FTL').astype(int)
    X_test = test_df[available].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df['is_ftl'] if 'is_ftl' in test_df.columns else (test_df['route_type'] == 'FTL').astype(int)
    
    # Train classifier
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbosity=0,
            use_label_encoder=False, eval_metric='logloss',
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        model_name = "XGBoost Classifier"
    except ImportError:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        model_name = "GB Classifier"
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n[DecisionEngine] {model_name} Accuracy: {acc:.4f}")
    print(f"\n[DecisionEngine] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Carting', 'FTL']))
    
    # Feature importance
    if hasattr(clf, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'feature': available,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n[DecisionEngine] Top 10 features for FTL/Carting decision:")
        for _, row in imp_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'ftl_carting_classifier.pkl'), 'wb') as f:
            pickle.dump(clf, f)
    
    return {
        'classifier': clf,
        'accuracy': acc,
        'feature_cols': available,
    }


def compute_bottleneck_scores(facility_impact_df, corridor_stats):
    """Compute bottleneck impact scores for hubs and corridors."""
    print("\n" + "="*80)
    print("PHASE 10: BOTTLENECK IMPACT ENGINE")
    print("="*80)
    
    # Hub bottleneck scores (already computed in analysis)
    hub_scores = facility_impact_df.copy()
    hub_scores = hub_scores.sort_values('impact_score_normalized', ascending=False)
    
    print("\n[BottleneckEngine] Top 10 Bottleneck Hubs:")
    for _, row in hub_scores.head(10).iterrows():
        print(f"  {row['facility_id']} ({row['type']}): "
              f"score={row['impact_score_normalized']:.1f}, "
              f"throughput={row['throughput']:.0f}, "
              f"SLA breach={row['sla_breach_pct']:.2%}")
    
    # Corridor bottleneck scores
    corr = corridor_stats.copy()
    corr['corridor_impact'] = (
        corr['sla_breach_pct'].fillna(0) * 0.4 +
        (corr['delay_ratio_mean'] / corr['delay_ratio_mean'].max()) * 0.3 +
        (corr['trip_count'] / corr['trip_count'].max()) * 0.3
    ) * 100
    corr = corr.sort_values('corridor_impact', ascending=False)
    
    print("\n[BottleneckEngine] Top 10 Bottleneck Corridors:")
    for _, row in corr.head(10).iterrows():
        print(f"  {row['source_center']} → {row['destination_center']}: "
              f"impact={row['corridor_impact']:.1f}, "
              f"delay_ratio={row['delay_ratio_mean']:.2f}, "
              f"SLA breach={row['sla_breach_pct']:.2%}")
    
    return {
        'hub_scores': hub_scores,
        'corridor_scores': corr,
    }


def compute_business_metrics(y_true, y_pred, osrm_pred=None):
    """Compute business-relevant metrics."""
    print("\n" + "="*80)
    print("PHASE 9: BUSINESS METRICS")
    print("="*80)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    nonzero = y_true > 0
    errors = np.abs(y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
    
    metrics = {
        'accuracy_at_10': (errors <= 0.10).mean() * 100,
        'accuracy_at_15': (errors <= 0.15).mean() * 100,
        'accuracy_at_20': (errors <= 0.20).mean() * 100,
        'accuracy_at_30': (errors <= 0.30).mean() * 100,
        'late_delivery_rate': ((y_pred[nonzero] < y_true[nonzero] * 0.85).mean() * 100),
        'avg_prediction_error_min': np.mean(np.abs(y_true - y_pred)),
        'median_prediction_error_min': np.median(np.abs(y_true - y_pred)),
    }
    
    # Compare vs OSRM
    if osrm_pred is not None:
        osrm_pred = np.array(osrm_pred)
        osrm_errors = np.abs(y_true[nonzero] - osrm_pred[nonzero]) / y_true[nonzero]
        metrics['osrm_accuracy_at_15'] = (osrm_errors <= 0.15).mean() * 100
        metrics['improvement_over_osrm'] = metrics['accuracy_at_15'] - metrics['osrm_accuracy_at_15']
        metrics['osrm_avg_error_min'] = np.mean(np.abs(y_true - osrm_pred))
    
    # Revenue-at-risk estimation (illustrative)
    # Assume avg shipment value = ₹500, penalty for late = 10%
    avg_shipment_value = 500
    penalty_rate = 0.10
    late_count = (y_pred < y_true * 0.85).sum()
    metrics['estimated_revenue_at_risk'] = late_count * avg_shipment_value * penalty_rate
    
    print("[BusinessMetrics] Results:")
    for k, v in metrics.items():
        if 'accuracy' in k or 'rate' in k or 'improvement' in k:
            print(f"  {k}: {v:.2f}%")
        elif 'revenue' in k:
            print(f"  {k}: ₹{v:,.0f}")
        else:
            print(f"  {k}: {v:.2f} min")
    
    return metrics
