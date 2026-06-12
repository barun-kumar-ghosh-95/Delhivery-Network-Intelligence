"""Corridor Intelligence Page"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def render(data):
    st.markdown('<h1 class="main-title">️ Corridor Intelligence</h1>', unsafe_allow_html=True)
    
    corridors = data.get('classified_corridors')
    if corridors is None:
        corridors = data.get('corridor_stats')
    
    if corridors is None:
        st.warning("Corridor data not available. Run the pipeline first.")
        return
    
    # Summary KPIs
    if 'delay_class' in corridors.columns:
        dist = corridors['delay_class'].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            v = dist.get('Healthy', 0)
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color: #10b981;">{v}</div>
                <div class="kpi-label">Healthy Corridors</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            v = dist.get('Moderate', 0)
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color: #f59e0b;">{v}</div>
                <div class="kpi-label">Moderate Delay</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            v = dist.get('Severe', 0)
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color: #ef4444;">{v}</div>
                <div class="kpi-label">Severe Delay</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            v = dist.get('Critical', 0)
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color: #dc2626;">{v}</div>
                <div class="kpi-label">Critical Delay</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Delay ratio distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'delay_ratio_mean' in corridors.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=corridors['delay_ratio_mean'].clip(upper=10),
                nbinsx=50,
                marker_color='#818cf8',
            ))
            fig.add_vline(x=1.0, line_color='#10b981', line_dash='dash',
                          annotation_text="On-time")
            fig.add_vline(x=2.0, line_color='#f59e0b', line_dash='dash',
                          annotation_text="2x delay")
            fig.add_vline(x=3.0, line_color='#ef4444', line_dash='dash',
                          annotation_text="3x delay")
            fig.update_layout(
                title="Corridor Delay Ratio Distribution",
                xaxis_title="Delay Ratio (Actual/OSRM)",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'delay_class' in corridors.columns:
            colors_map = {'Healthy': '#10b981', 'Moderate': '#f59e0b', 
                          'Severe': '#ef4444', 'Critical': '#dc2626'}
            fig = go.Figure(data=[go.Pie(
                labels=dist.index.tolist(),
                values=dist.values.tolist(),
                marker=dict(colors=[colors_map.get(l, '#6366f1') for l in dist.index]),
                hole=0.5,
                textinfo='label+percent+value',
            )])
            fig.update_layout(
                title="Corridor Health Distribution",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Delay vs Volume scatter
    if 'delay_ratio_mean' in corridors.columns and 'trip_count' in corridors.columns:
        st.markdown('<div class="section-header"> Delay vs Volume Analysis</div>', unsafe_allow_html=True)
        
        color_col = 'delay_class' if 'delay_class' in corridors.columns else None
        fig = px.scatter(
            corridors,
            x='trip_count',
            y='delay_ratio_mean',
            color=color_col,
            color_discrete_map={'Healthy': '#10b981', 'Moderate': '#f59e0b', 
                                'Severe': '#ef4444', 'Critical': '#dc2626'},
            hover_data=['source_center', 'destination_center'],
            title="Corridor Delay vs Traffic Volume",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            xaxis_title="Trip Count",
            yaxis_title="Delay Ratio (Actual/OSRM)",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top delayed corridors table
    st.markdown('<div class="section-header"> Most Delayed Corridors</div>', unsafe_allow_html=True)
    
    top_delayed = corridors.sort_values('delay_ratio_mean', ascending=False).head(20)
    display_cols = ['source_center', 'destination_center', 'delay_ratio_mean', 
                    'delay_class', 'trip_count', 'sla_breach_pct', 'corridor_reliability']
    avail = [c for c in display_cols if c in top_delayed.columns]
    st.dataframe(top_delayed[avail], use_container_width=True, height=500)
    
    # Filter by delay class
    st.markdown('<div class="section-header"> Filter Corridors</div>', unsafe_allow_html=True)
    if 'delay_class' in corridors.columns:
        selected_class = st.multiselect(
            "Filter by Delay Class",
            ['Healthy', 'Moderate', 'Severe', 'Critical'],
            default=['Severe', 'Critical'],
        )
        filtered = corridors[corridors['delay_class'].isin(selected_class)]
        st.dataframe(filtered[avail].sort_values('delay_ratio_mean', ascending=False), 
                     use_container_width=True, height=400)
