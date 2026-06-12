"""
Graph Embeddings Module
=======================
Node2Vec and graph statistics for feature engineering.
"""

import numpy as np
import pandas as pd
import networkx as nx


def compute_node2vec_embeddings(G, dimensions=64, walk_length=30, num_walks=80, p=1.0, q=2.0, seed=42):
    """
    Compute Node2Vec-style embeddings using random walks + simple averaging.
    Uses a lightweight approach without requiring gensim for speed.
    """
    print(f"[Embeddings] Computing graph embeddings (dim={dimensions})...")
    
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    
    if n == 0:
        return pd.DataFrame()
    
    np.random.seed(seed)
    
    # Build adjacency info
    neighbors = {}
    weights = {}
    for node in nodes:
        nbrs = list(G.successors(node)) if G.is_directed() else list(G.neighbors(node))
        if nbrs:
            w = [G[node][nbr].get('weight', 1.0) for nbr in nbrs]
            total = sum(w)
            neighbors[node] = nbrs
            weights[node] = [wi/total for wi in w]
        else:
            neighbors[node] = []
            weights[node] = []
    
    # Random walks
    walk_counts = np.zeros((n, n), dtype=np.float32)
    
    for _ in range(num_walks):
        for start_node in nodes:
            current = start_node
            for _ in range(walk_length):
                nbrs = neighbors.get(current, [])
                if not nbrs:
                    break
                # Weighted random choice
                current = np.random.choice(nbrs, p=weights[current])
                walk_counts[node_to_idx[start_node], node_to_idx[current]] += 1
    
    # Normalize
    row_sums = walk_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    walk_probs = walk_counts / row_sums
    
    # SVD for dimensionality reduction
    from numpy.linalg import svd
    
    # Log transform for better embeddings
    walk_probs_log = np.log1p(walk_probs * 1000)
    
    actual_dim = min(dimensions, n - 1, walk_probs_log.shape[1])
    if actual_dim <= 0:
        actual_dim = min(8, n)
    
    try:
        U, S, Vt = svd(walk_probs_log, full_matrices=False)
        embeddings = U[:, :actual_dim] * np.sqrt(S[:actual_dim])
    except Exception:
        # Fallback: use raw adjacency features
        embeddings = walk_probs[:, :actual_dim] if walk_probs.shape[1] >= actual_dim else walk_probs
    
    # Pad if needed
    if embeddings.shape[1] < dimensions:
        padding = np.zeros((n, dimensions - embeddings.shape[1]))
        embeddings = np.concatenate([embeddings, padding], axis=1)
    
    # Create dataframe
    col_names = [f'emb_{i}' for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=col_names)
    emb_df['facility_id'] = nodes
    
    print(f"[Embeddings] Generated {embeddings.shape[1]}-dim embeddings for {n} nodes")
    return emb_df


def compute_graph_statistics(G):
    """Compute structural graph features for each node."""
    print("[Embeddings] Computing graph structural features...")
    
    nodes = list(G.nodes())
    stats = []
    
    # Precompute centralities
    degree_cent = nx.degree_centrality(G)
    in_degree_cent = nx.in_degree_centrality(G) if G.is_directed() else degree_cent
    out_degree_cent = nx.out_degree_centrality(G) if G.is_directed() else degree_cent
    betweenness = nx.betweenness_centrality(G, weight='weight')
    try:
        pagerank = nx.pagerank(G, weight='weight', max_iter=200)
    except:
        pagerank = {n: 1.0/len(nodes) for n in nodes}
    
    G_und = G.to_undirected()
    clustering = nx.clustering(G_und)
    
    for node in nodes:
        # Neighbor aggregations
        nbrs = list(G.successors(node)) if G.is_directed() else list(G.neighbors(node))
        
        nbr_delays = [G.nodes[n].get('avg_delay', 0) for n in nbrs if 'avg_delay' in G.nodes[n]]
        nbr_throughputs = [G.nodes[n].get('throughput', 0) for n in nbrs if 'throughput' in G.nodes[n]]
        
        edge_weights = [G[node][n].get('weight', 1) for n in G.successors(node)] if G.is_directed() else []
        edge_distances = [G[node][n].get('distance', 0) for n in G.successors(node)] if G.is_directed() else []
        
        stats.append({
            'facility_id': node,
            'graph_degree': degree_cent.get(node, 0),
            'graph_in_degree': in_degree_cent.get(node, 0),
            'graph_out_degree': out_degree_cent.get(node, 0),
            'graph_betweenness': betweenness.get(node, 0),
            'graph_pagerank': pagerank.get(node, 0),
            'graph_clustering': clustering.get(node, 0),
            'graph_nbr_avg_delay': np.mean(nbr_delays) if nbr_delays else 0,
            'graph_nbr_max_delay': np.max(nbr_delays) if nbr_delays else 0,
            'graph_nbr_avg_throughput': np.mean(nbr_throughputs) if nbr_throughputs else 0,
            'graph_avg_edge_weight': np.mean(edge_weights) if edge_weights else 0,
            'graph_avg_edge_distance': np.mean(edge_distances) if edge_distances else 0,
            'graph_num_neighbors': len(nbrs),
        })
    
    stats_df = pd.DataFrame(stats)
    print(f"[Embeddings] Computed {len(stats_df.columns)-1} graph features for {len(nodes)} nodes")
    return stats_df
