"""
Network Science Analysis Module
================================
Computes graph-theoretic metrics for bottleneck detection and facility ranking.
"""

import networkx as nx
import pandas as pd
import numpy as np


def compute_centrality_metrics(G):
    """Compute all centrality metrics for the facility graph."""
    print("[Analysis] Computing centrality metrics...")
    metrics = {}
    
    # Degree centrality
    metrics['degree_centrality'] = nx.degree_centrality(G)
    metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
    metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
    
    # Betweenness centrality (bottleneck indicator)
    metrics['betweenness_centrality'] = nx.betweenness_centrality(G, weight='weight')
    
    # Closeness centrality (accessibility)
    try:
        metrics['closeness_centrality'] = nx.closeness_centrality(G)
    except Exception:
        metrics['closeness_centrality'] = {n: 0.0 for n in G.nodes()}
    
    # PageRank (importance by connection quality)
    try:
        metrics['pagerank'] = nx.pagerank(G, weight='weight', max_iter=200)
    except Exception:
        metrics['pagerank'] = {n: 1.0/G.number_of_nodes() for n in G.nodes()}
    
    # Clustering coefficient (on undirected version)
    G_undirected = G.to_undirected()
    metrics['clustering_coefficient'] = nx.clustering(G_undirected)
    
    print(f"[Analysis] Computed 7 centrality metrics for {G.number_of_nodes()} nodes")
    return metrics


def detect_communities(G):
    """Detect communities using greedy modularity."""
    print("[Analysis] Detecting communities...")
    G_undirected = G.to_undirected()
    
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G_undirected, weight='weight'))
        
        # Create node → community mapping
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
        
        print(f"[Analysis] Found {len(communities)} communities")
        return community_map, communities
    except Exception as e:
        print(f"[Analysis] Community detection failed: {e}")
        return {n: 0 for n in G.nodes()}, [set(G.nodes())]


def find_critical_infrastructure(G):
    """Find bridge edges and articulation points."""
    print("[Analysis] Finding critical infrastructure...")
    G_undirected = G.to_undirected()
    
    # Articulation points (single points of failure)
    try:
        artic_points = list(nx.articulation_points(G_undirected))
    except Exception:
        artic_points = []
    
    # Bridge edges (critical corridors)
    try:
        bridges = list(nx.bridges(G_undirected))
    except Exception:
        bridges = []
    
    print(f"[Analysis] Found {len(artic_points)} articulation points, {len(bridges)} bridge edges")
    return artic_points, bridges


def compute_facility_impact_scores(G, centrality_metrics, facility_stats):
    """
    Compute composite impact score for each facility.
    Score = betweenness × avg_delay × throughput (normalized)
    """
    print("[Analysis] Computing facility impact scores...")
    
    betweenness = centrality_metrics['betweenness_centrality']
    pagerank = centrality_metrics['pagerank']
    
    scores = []
    for node in G.nodes():
        node_data = G.nodes[node]
        b = betweenness.get(node, 0)
        pr = pagerank.get(node, 0)
        throughput = node_data.get('throughput', 0)
        avg_delay = node_data.get('avg_delay', 0)
        sla = node_data.get('sla_breach_pct', 0)
        
        # Composite impact score
        impact = (b * 0.3 + pr * 0.2) * (avg_delay * 0.25 + sla * 0.15 + np.log1p(throughput) * 0.1)
        
        scores.append({
            'facility_id': node,
            'name': node_data.get('name', ''),
            'type': node_data.get('type', ''),
            'betweenness': b,
            'pagerank': pr,
            'throughput': throughput,
            'avg_delay': avg_delay,
            'sla_breach_pct': sla,
            'impact_score': impact,
        })
    
    impact_df = pd.DataFrame(scores).sort_values('impact_score', ascending=False)
    
    # Normalize impact score to 0-100
    if impact_df['impact_score'].max() > 0:
        impact_df['impact_score_normalized'] = (
            impact_df['impact_score'] / impact_df['impact_score'].max() * 100
        )
    else:
        impact_df['impact_score_normalized'] = 0
    
    print(f"[Analysis] Top 5 bottleneck facilities:")
    for _, row in impact_df.head(5).iterrows():
        print(f"  {row['facility_id']} ({row['type']}): score={row['impact_score_normalized']:.1f}")
    
    return impact_df


def run_full_analysis(G, facility_stats):
    """Run complete network science analysis pipeline."""
    results = {}
    
    # Centrality metrics
    results['centrality'] = compute_centrality_metrics(G)
    
    # Community detection
    results['community_map'], results['communities'] = detect_communities(G)
    
    # Critical infrastructure
    results['articulation_points'], results['bridges'] = find_critical_infrastructure(G)
    
    # Impact scores
    results['impact_scores'] = compute_facility_impact_scores(
        G, results['centrality'], facility_stats
    )
    
    # Add centrality to graph nodes
    for metric_name, metric_values in results['centrality'].items():
        nx.set_node_attributes(G, metric_values, metric_name)
    nx.set_node_attributes(G, results['community_map'], 'community')
    
    # Network summary
    results['summary'] = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'num_communities': len(results['communities']),
        'num_articulation_points': len(results['articulation_points']),
        'num_bridges': len(results['bridges']),
        'avg_clustering': np.mean(list(results['centrality']['clustering_coefficient'].values())),
    }
    
    print(f"\n[Analysis] Network Summary:")
    for k, v in results['summary'].items():
        print(f"  {k}: {v}")
    
    return results
