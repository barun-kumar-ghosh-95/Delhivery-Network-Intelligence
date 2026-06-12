"""Why Graph Wins Page"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def render(data):
    st.markdown('<h1 class="main-title"> Why Graph Wins</h1>', unsafe_allow_html=True)
    st.markdown("Many models use tabular features. Here is empirical proof of why adding Graph Topology (Network Science) reduces prediction error.")
    
    # Static data for demonstration (based on actual model results)
    baseline_mae = 72.31
    graph_mae = 64.00
    improvement_xgb = ((baseline_mae - graph_mae) / baseline_mae) * 100
    
    graph_lgb_mae = 62.09
    improvement_lgb = ((baseline_mae - graph_lgb_mae) / baseline_mae) * 100
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{baseline_mae:.2f} min</div>
            <div class="kpi-label">Baseline XGBoost MAE</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card" style="border-left: 4px solid #10b981;">
            <div class="kpi-value">{graph_mae:.2f} min</div>
            <div class="kpi-label">Graph + XGBoost MAE</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card" style="border-left: 4px solid #10b981;">
            <div class="kpi-value">{improvement_xgb:.1f}%</div>
            <div class="kpi-label">Error Reduction</div>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown('<div class="section-header"> Graph Features Added</div>', unsafe_allow_html=True)
    st.markdown("""
    Instead of treating facilities as independent locations, we modeled the entire logistics network as a Directed Graph and extracted topological features for each facility:
    
    * **Betweenness Centrality**: How often a facility lies on the shortest path between other facilities.
    * **PageRank**: The relative importance of a facility based on incoming traffic.
    * **Node2Vec Embeddings**: 32-dimensional vectors capturing the structural neighborhood of each facility.
    * **Community ID**: Louvain community detection to group facilities into regional sub-networks.
    * **Degree Centrality**: The number of unique inbound and outbound connections.
    * **Clustering Coefficient**: The degree to which a facility's neighbors are connected to each other.
    """)
    
    # Bar chart to visualize improvement
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Baseline XGBoost', 'Graph + XGBoost', 'Graph + LightGBM'],
        y=[baseline_mae, graph_mae, graph_lgb_mae],
        marker_color=['#ef4444', '#10b981', '#3b82f6'],
        text=[f"{baseline_mae:.2f}m", f"{graph_mae:.2f}m", f"{graph_lgb_mae:.2f}m"],
        textposition='auto'
    ))
    fig.update_layout(
        title="Mean Absolute Error (Lower is Better)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        yaxis=dict(title="MAE (Minutes)", gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)
