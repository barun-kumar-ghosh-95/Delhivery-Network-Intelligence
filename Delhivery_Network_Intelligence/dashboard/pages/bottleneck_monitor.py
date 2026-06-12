"""Bottleneck Monitor Page"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def render(data):
    st.markdown('<h1 class="main-title"> Bottleneck Monitor</h1>', unsafe_allow_html=True)
    
    hubs = data.get('bottleneck_hubs')
    corridors = data.get('bottleneck_corridors')
    
    if hubs is None:
        st.warning("Bottleneck data not available. Run the pipeline first.")
        return
    
    # KPI Row
    c1, c2, c3 = st.columns(3)
    with c1:
        critical = len(hubs[hubs['impact_score_normalized'] > 70]) if 'impact_score_normalized' in hubs.columns else 0
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color: #ef4444;">{critical}</div>
            <div class="kpi-label">Critical Bottleneck Hubs</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        if corridors is not None and 'corridor_impact' in corridors.columns:
            corr_critical = len(corridors[corridors['corridor_impact'] > 60])
        else:
            corr_critical = 0
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color: #f59e0b;">{corr_critical}</div>
            <div class="kpi-label">High-Impact Corridors</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        avg_sla = hubs['sla_breach_pct'].mean() * 100 if 'sla_breach_pct' in hubs.columns else 0
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{avg_sla:.1f}%</div>
            <div class="kpi-label">Avg SLA Breach Rate</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs([" Hub Bottlenecks", "️ Corridor Bottlenecks"])
    
    with tab1:
        top = hubs.head(20)
        
        fig = go.Figure()
        colors = ['#dc2626' if v > 70 else '#f59e0b' if v > 40 else '#10b981' 
                  for v in top['impact_score_normalized']]
        
        fig.add_trace(go.Bar(
            y=top['facility_id'],
            x=top['impact_score_normalized'],
            orientation='h',
            marker_color=colors,
            text=top['impact_score_normalized'].round(1),
            textposition='outside',
        ))
        fig.update_layout(
            title="Hub Bottleneck Impact Score",
            xaxis_title="Impact Score (0-100)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=600,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.markdown("#### Detailed Hub Metrics")
        display_cols = ['facility_id', 'type', 'impact_score_normalized', 
                       'throughput', 'betweenness', 'pagerank', 'sla_breach_pct', 'avg_delay']
        avail = [c for c in display_cols if c in hubs.columns]
        st.dataframe(hubs[avail].head(20), use_container_width=True, height=400)
    
    with tab2:
        if corridors is not None and 'corridor_impact' in corridors.columns:
            top_corr = corridors.head(20)
            
            top_corr['corridor'] = top_corr['source_center'] + ' → ' + top_corr['destination_center']
            
            fig = go.Figure()
            colors = ['#dc2626' if v > 60 else '#f59e0b' if v > 30 else '#10b981'
                      for v in top_corr['corridor_impact']]
            
            fig.add_trace(go.Bar(
                y=top_corr['corridor'],
                x=top_corr['corridor_impact'],
                orientation='h',
                marker_color=colors,
                text=top_corr['corridor_impact'].round(1),
                textposition='outside',
            ))
            fig.update_layout(
                title="Corridor Bottleneck Impact Score",
                xaxis_title="Impact Score",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=600,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            display_cols = ['source_center', 'destination_center', 'corridor_impact',
                           'delay_ratio_mean', 'sla_breach_pct', 'trip_count']
            avail = [c for c in display_cols if c in corridors.columns]
            st.dataframe(corridors[avail].head(20), use_container_width=True, height=400)
