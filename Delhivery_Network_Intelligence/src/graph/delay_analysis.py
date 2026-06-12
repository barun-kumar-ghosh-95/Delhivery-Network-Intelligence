"""
Delay Analysis Module
=====================
Corridor delay classification, SLA breach analysis, root-cause framework.
"""

import pandas as pd
import numpy as np


def classify_corridors(corridor_stats):
    """Classify corridors into delay categories."""
    print("[DelayAnalysis] Classifying corridors...")
    
    df = corridor_stats.copy()
    
    # Ensure delay_class exists
    df['delay_class'] = pd.cut(
        df['delay_ratio_mean'],
        bins=[0, 1.3, 2.0, 3.0, float('inf')],
        labels=['Healthy', 'Moderate', 'Severe', 'Critical']
    )
    
    # Distribution
    dist = df['delay_class'].value_counts()
    print("[DelayAnalysis] Corridor delay distribution:")
    for cls, cnt in dist.items():
        print(f"  {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")
    
    return df


def get_top_delayed_corridors(corridor_stats, top_n=20):
    """Get the most delayed corridors."""
    df = corridor_stats.sort_values('delay_ratio_mean', ascending=False).head(top_n)
    
    print(f"[DelayAnalysis] Top {top_n} delayed corridors:")
    for _, row in df.head(10).iterrows():
        print(f"  {row['source_center']} → {row['destination_center']}: "
              f"ratio={row['delay_ratio_mean']:.2f}, trips={row.get('trip_count', 0)}")
    
    return df


def compute_sla_analysis(corridor_stats, facility_stats):
    """Compute SLA breach contribution analysis."""
    print("[DelayAnalysis] Computing SLA breach analysis...")
    
    results = {
        'corridor_sla': corridor_stats[['source_center', 'destination_center', 
                                         'sla_breach_pct', 'trip_count', 'delay_ratio_mean']].copy(),
        'total_corridors': len(corridor_stats),
        'breaching_corridors': (corridor_stats['sla_breach_pct'] > 0.3).sum(),
        'critical_corridors': (corridor_stats['delay_class'] == 'Critical').sum() if 'delay_class' in corridor_stats.columns else 0,
    }
    
    # Weighted SLA contribution
    results['corridor_sla']['sla_contribution'] = (
        results['corridor_sla']['sla_breach_pct'] * results['corridor_sla']['trip_count']
    )
    total_contribution = results['corridor_sla']['sla_contribution'].sum()
    if total_contribution > 0:
        results['corridor_sla']['sla_contribution_pct'] = (
            results['corridor_sla']['sla_contribution'] / total_contribution * 100
        )
    else:
        results['corridor_sla']['sla_contribution_pct'] = 0
    
    results['corridor_sla'] = results['corridor_sla'].sort_values('sla_contribution', ascending=False)
    
    print(f"[DelayAnalysis] {results['breaching_corridors']}/{results['total_corridors']} corridors with >30% SLA breach")
    
    return results


def run_delay_analysis(corridor_stats, facility_stats):
    """Run complete delay analysis pipeline."""
    results = {}
    
    # Classify corridors
    corridor_stats = classify_corridors(corridor_stats)
    results['classified_corridors'] = corridor_stats
    
    # Top delayed corridors
    results['top_delayed'] = get_top_delayed_corridors(corridor_stats)
    
    # SLA analysis
    results['sla_analysis'] = compute_sla_analysis(corridor_stats, facility_stats)
    
    # Summary stats
    results['summary'] = {
        'total_corridors': len(corridor_stats),
        'healthy_pct': (corridor_stats['delay_class'] == 'Healthy').mean() * 100,
        'moderate_pct': (corridor_stats['delay_class'] == 'Moderate').mean() * 100,
        'severe_pct': (corridor_stats['delay_class'] == 'Severe').mean() * 100,
        'critical_pct': (corridor_stats['delay_class'] == 'Critical').mean() * 100,
        'avg_delay_ratio': corridor_stats['delay_ratio_mean'].mean(),
        'median_delay_ratio': corridor_stats['delay_ratio_mean'].median(),
    }
    
    return results
