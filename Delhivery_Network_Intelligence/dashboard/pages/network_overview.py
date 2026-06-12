"""Network Overview Page"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render(data):
    st.markdown('<h1 class="main-title"> Network Overview</h1>', unsafe_allow_html=True)
    
    summary = data.get('network_summary', {})
    delay = data.get('delay_summary', {})
    biz = data.get('summary', {}).get('business_metrics', {})
    
    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{summary.get('num_nodes', 1657):,}</div>
            <div class="kpi-label">Facilities</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{summary.get('num_edges', 2783):,}</div>
            <div class="kpi-label">Corridors</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{summary.get('num_communities', 0)}</div>
            <div class="kpi-label">Communities</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        density = summary.get('density', 0.001)
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{density:.4f}</div>
            <div class="kpi-label">Network Density</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        artic = summary.get('num_articulation_points', 0)
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{artic}</div>
            <div class="kpi-label">Critical Nodes</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Network Topology Map
    st.markdown('<div class="section-header"> Network Topology Map</div>', unsafe_allow_html=True)
    st.markdown("Visualizing the core logistics network (nodes = facilities, edges = corridors). Red nodes indicate high bottleneck risk.")
    
    import networkx as nx
    import os
    import pickle
    
    @st.cache_data
    def load_and_plot_graph():
        graph_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'graphs', 'facility_graph.pkl')
        if not os.path.exists(graph_path):
            return None
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
            
        # For performance, if graph is too large, take the largest connected component or a sample
        if len(G.nodes) > 1000:
            # Get nodes with highest degree to show the core network
            degrees = dict(G.degree())
            core_nodes = sorted(degrees, key=degrees.get, reverse=True)[:500]
            G = G.subgraph(core_nodes)
            
        pos = nx.spring_layout(G, seed=42, k=0.15)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#4b5563'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        # Load impact scores to color nodes
        impact_df = data.get('impact_scores')
        impact_dict = {}
        if impact_df is not None:
            impact_dict = dict(zip(impact_df['facility_id'], impact_df['impact_score_normalized']))
            
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            score = impact_dict.get(node, 0)
            node_color.append(score)
            node_text.append(f"Facility: {node}<br>Bottleneck Score: {score:.1f}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='RdYlGn_r',  # Red for high bottleneck, Green for healthy
                reversescale=False,
                color=node_color,
                size=8,
                colorbar=dict(
                    thickness=15,
                    title='Bottleneck Score',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=1))
                
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=600,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig
        
    graph_fig = load_and_plot_graph()
    if graph_fig:
        st.plotly_chart(graph_fig, use_container_width=True)
    else:
        st.info("Graph data not found. Please run the data pipeline first.")
        
    st.markdown("---")
    st.markdown('<div class="section-header"> Model Performance Comparison</div>', unsafe_allow_html=True)
    
    model_comp = data.get('model_comparison')
    if model_comp is not None and len(model_comp) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=model_comp['model'],
                y=model_comp['R2'],
                marker=dict(
                    color=model_comp['R2'],
                    colorscale=[[0, '#6366f1'], [0.5, '#818cf8'], [1, '#a78bfa']],
                ),
                text=model_comp['R2'].round(4),
                textposition='outside',
            ))
            fig.update_layout(
                title="R² Score by Model",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
                xaxis=dict(tickangle=-45),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=model_comp['model'],
                y=model_comp['Acc@15%'],
                marker=dict(
                    color=model_comp['Acc@15%'],
                    colorscale=[[0, '#10b981'], [0.5, '#34d399'], [1, '#6ee7b7']],
                ),
                text=model_comp['Acc@15%'].round(1).astype(str) + '%',
                textposition='outside',
            ))
            fig.update_layout(
                title="Accuracy@15% by Model",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
                xaxis=dict(tickangle=-45),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(model_comp.style.format({
            'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'MAPE': '{:.2f}',
            'R2': '{:.4f}', 'Acc@10%': '{:.1f}%', 'Acc@15%': '{:.1f}%', 'Acc@20%': '{:.1f}%',
        }), use_container_width=True, height=300)
    
    # Delay Distribution
    st.markdown('<div class="section-header">️ Corridor Health Distribution</div>', unsafe_allow_html=True)
    
    if delay:
        labels = ['Healthy', 'Moderate', 'Severe', 'Critical']
        values = [delay.get('healthy_pct', 0), delay.get('moderate_pct', 0), 
                  delay.get('severe_pct', 0), delay.get('critical_pct', 0)]
        colors = ['#10b981', '#f59e0b', '#ef4444', '#dc2626']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors),
            hole=0.5,
            textinfo='label+percent',
            textfont=dict(size=14),
        )])
        fig.update_layout(
            title="Corridor Delay Classification",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Facility Impact
    st.markdown('<div class="section-header"> Top Impact Facilities</div>', unsafe_allow_html=True)
    
    impact = data.get('impact_scores')
    if impact is not None and len(impact) > 0:
        top = impact.head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top['facility_id'],
            x=top['impact_score_normalized'],
            orientation='h',
            marker=dict(
                color=top['impact_score_normalized'],
                colorscale=[[0, '#6366f1'], [0.5, '#ef4444'], [1, '#dc2626']],
            ),
            text=top['impact_score_normalized'].round(1),
            textposition='outside',
        ))
        fig.update_layout(
            title="Facility Impact Score (Bottleneck Risk)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Impact Score (0-100)"),
        )
        st.plotly_chart(fig, use_container_width=True)
