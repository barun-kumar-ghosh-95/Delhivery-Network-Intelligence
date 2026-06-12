"""
Model Evaluation Module
=======================
All evaluation metrics for ETA prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_all_metrics(y_true, y_pred, model_name="Model"):
    """Compute all evaluation metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs(y_true[nonzero] - y_pred[nonzero]) / np.abs(y_true[nonzero])) * 100
    else:
        mape = float('inf')
    
    # Accuracy@15% (within 15% of actual)
    if nonzero.sum() > 0:
        within_15 = np.abs(y_true[nonzero] - y_pred[nonzero]) / np.abs(y_true[nonzero]) <= 0.15
        acc_15 = within_15.mean() * 100
    else:
        acc_15 = 0
    
    # Accuracy@10% and @20%
    if nonzero.sum() > 0:
        acc_10 = (np.abs(y_true[nonzero] - y_pred[nonzero]) / np.abs(y_true[nonzero]) <= 0.10).mean() * 100
        acc_20 = (np.abs(y_true[nonzero] - y_pred[nonzero]) / np.abs(y_true[nonzero]) <= 0.20).mean() * 100
    else:
        acc_10 = acc_20 = 0
    
    metrics = {
        'model': model_name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'R2': round(r2, 4),
        'Acc@10%': round(acc_10, 2),
        'Acc@15%': round(acc_15, 2),
        'Acc@20%': round(acc_20, 2),
    }
    
    return metrics


def print_metrics(metrics):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"  Model: {metrics['model']}")
    print(f"{'='*60}")
    print(f"  MAE:      {metrics['MAE']:.2f} minutes")
    print(f"  RMSE:     {metrics['RMSE']:.2f} minutes")
    print(f"  MAPE:     {metrics['MAPE']:.2f}%")
    print(f"  R²:       {metrics['R2']:.4f}")
    print(f"  Acc@10%:  {metrics['Acc@10%']:.2f}%")
    print(f"  Acc@15%:  {metrics['Acc@15%']:.2f}%")
    print(f"  Acc@20%:  {metrics['Acc@20%']:.2f}%")
    print(f"{'='*60}")


def compare_models(all_metrics):
    """Create comparison table from list of metrics dicts."""
    df = pd.DataFrame(all_metrics)
    df = df.sort_values('R2', ascending=False)
    return df
