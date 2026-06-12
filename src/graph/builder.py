"""
Graph Builder Module
====================
Constructs 4 graph types from delivery data:
1. Facility Graph - all facilities as nodes, corridors as edges
2. Corridor Graph - weighted by delay/reliability
3. Time-Aware Graph - temporal edge attributes
4. Route-Type Graph - separate FTL/Carting subgraphs
"""

import networkx as nx
import pandas as pd
import numpy as np
import pickle
import os


def build_facility_graph(df, corridor_stats, facility_stats):
    """Build the main facility graph with node and edge features."""
    print("[GraphBuilder] Building Facility Graph...")
    G = nx.DiGraph()
    
    # Add nodes with features
    for _, row in facility_stats.iterrows():
        attrs = {
            'name': row.get('facility_name', ''),
            'type': row.get('facility_type', 'Unknown'),
            'throughput': row.get('total_throughput', 0),
            'in_degree_custom': row.get('in_degree', 0),
            'out_degree_custom': row.get('out_degree', 0),
            'avg_delay': row.get('avg_delay', 0),
            'sla_breach_pct': row.get('sla_breach_pct', 0),
            'total_degree': row.get('total_degree', 0),
        }
        G.add_node(row['facility_id'], **attrs)
    
    # Add edges with features
    for _, row in corridor_stats.iterrows():
        attrs = {
            'distance': row.get('osrm_distance_mean', 0),
            'osrm_time': row.get('osrm_time_mean', 0),
            'actual_time_mean': row.get('actual_time_mean', 0),
            'delay_ratio': row.get('delay_ratio_mean', 1),
            'frequency': row.get('trip_count', 0),
            'reliability': row.get('corridor_reliability', 0),
            'sla_breach_pct': row.get('sla_breach_pct', 0),
            'delay_class': str(row.get('delay_class', 'Unknown')),
            'route_type': row.get('primary_route_type', 'Unknown'),
            'weight': row.get('delay_ratio_mean', 1),
        }
        G.add_edge(row['source_center'], row['destination_center'], **attrs)
    
    print(f"[GraphBuilder] Facility Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_route_type_graphs(df, corridor_stats):
    """Build separate subgraphs for FTL and Carting networks."""
    print("[GraphBuilder] Building Route-Type Graphs...")
    graphs = {}
    
    for route_type in ['FTL', 'Carting']:
        rt_df = df[df['route_type'] == route_type]
        G = nx.DiGraph()
        
        # Get corridors for this route type
        rt_corridors = rt_df.groupby(['source_center', 'destination_center']).agg({
            'actual_time': 'mean',
            'osrm_time': 'mean',
            'osrm_distance': 'mean',
            'factor': 'mean',
            'trip_uuid': 'nunique',
        }).reset_index()
        
        # Add all facilities involved
        all_facilities = set(rt_df['source_center'].unique()) | set(rt_df['destination_center'].unique())
        for f in all_facilities:
            G.add_node(f)
        
        for _, row in rt_corridors.iterrows():
            G.add_edge(
                row['source_center'], row['destination_center'],
                actual_time=row['actual_time'],
                osrm_time=row['osrm_time'],
                distance=row['osrm_distance'],
                delay_ratio=row['factor'],
                frequency=row['trip_uuid'],
                weight=row['factor'],
            )
        
        graphs[route_type] = G
        print(f"[GraphBuilder]   {route_type}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return graphs


def build_all_graphs(df, corridor_stats, facility_stats):
    """Build all graph types and return as dict."""
    graphs = {}
    graphs['facility'] = build_facility_graph(df, corridor_stats, facility_stats)
    route_graphs = build_route_type_graphs(df, corridor_stats)
    graphs['ftl'] = route_graphs.get('FTL')
    graphs['carting'] = route_graphs.get('Carting')
    return graphs


def save_graphs(graphs, output_dir):
    """Save graphs to disk as pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, G in graphs.items():
        if G is not None:
            path = os.path.join(output_dir, f'{name}_graph.pkl')
            with open(path, 'wb') as f:
                pickle.dump(G, f)
            print(f"[GraphBuilder] Saved {name} graph to {path}")


def load_graph(path):
    """Load a graph from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)
