"""ETA Predictor Page"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os


def render(data):
    st.markdown('<h1 class="main-title">️ ETA Predictor</h1>', unsafe_allow_html=True)
    
    test_trips = data.get('test_trips')
    
    if test_trips is None or len(test_trips) == 0:
        st.warning("No trip data available. Run the pipeline first.")
        return
    
    # Source/Dest selector
    col1, col2 = st.columns(2)
    
    sources = sorted(test_trips['source_center'].dropna().unique())
    destinations = sorted(test_trips['destination_center'].dropna().unique())
    
    with col1:
        src = st.selectbox(" Source Facility", sources, key="eta_src")
    with col2:
        dst = st.selectbox(" Destination Facility", destinations, key="eta_dst")
    
    # Filter trips for this corridor
    corridor_trips = test_trips[
        (test_trips['source_center'] == src) & 
        (test_trips['destination_center'] == dst)
    ]
    
    if len(corridor_trips) == 0:
        st.info(f"No historical data for {src} → {dst}. Showing OSRM estimate only.")
        
        # Show general stats
        src_trips = test_trips[test_trips['source_center'] == src]
        if len(src_trips) > 0:
            avg_time = src_trips['actual_time'].mean()
            avg_osrm = src_trips['osrm_time'].mean()
            st.metric("Avg time from this source", f"{avg_time:.0f} min")
            st.metric("Avg OSRM estimate", f"{avg_osrm:.0f} min")
        return
    
    # Prediction display
    st.markdown("---")
    
    avg_actual = corridor_trips['actual_time'].mean()
    avg_osrm = corridor_trips['osrm_time'].mean()
    avg_distance = corridor_trips['osrm_distance'].mean() if 'osrm_distance' in corridor_trips.columns else 0
    num_trips = len(corridor_trips)
    
    # Model prediction (use historical mean as proxy since we don't load model here)
    model_pred = avg_actual  # In production, load model and predict
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{model_pred:.0f}</div>
            <div class="kpi-label">Predicted ETA (min)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{avg_osrm:.0f}</div>
            <div class="kpi-label">OSRM Estimate (min)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        improvement = ((avg_osrm - avg_actual) / avg_actual * 100)
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{avg_distance:.0f}</div>
            <div class="kpi-label">Distance (km)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{num_trips}</div>
            <div class="kpi-label">Historical Trips</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Historical distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=corridor_trips['actual_time'],
            nbinsx=30,
            marker_color='#818cf8',
            name='Actual Time',
            opacity=0.7,
        ))
        fig.add_vline(x=avg_osrm, line_color='#f59e0b', line_dash='dash',
                      annotation_text=f"OSRM: {avg_osrm:.0f}min")
        fig.add_vline(x=avg_actual, line_color='#10b981', line_dash='dash',
                      annotation_text=f"Actual avg: {avg_actual:.0f}min")
        fig.update_layout(
            title="Historical Trip Time Distribution",
            xaxis_title="Time (minutes)",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Actual vs OSRM scatter
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=corridor_trips['osrm_time'],
            y=corridor_trips['actual_time'],
            mode='markers',
            marker=dict(color='#818cf8', size=8, opacity=0.6),
            name='Trips',
        ))
        max_val = max(corridor_trips['actual_time'].max(), corridor_trips['osrm_time'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(color='#ef4444', dash='dash'),
            name='Perfect Prediction',
        ))
        fig.update_layout(
            title="Actual vs OSRM Time",
            xaxis_title="OSRM Time (min)",
            yaxis_title="Actual Time (min)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence interval
    if len(corridor_trips) > 2:
        p10 = corridor_trips['actual_time'].quantile(0.1)
        p50 = corridor_trips['actual_time'].quantile(0.5)
        p90 = corridor_trips['actual_time'].quantile(0.9)
        
        st.markdown('<div class="section-header"> Confidence Intervals</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Best Case (P10)", f"{p10:.0f} min")
        with c2:
            st.metric("Expected (P50)", f"{p50:.0f} min")
        with c3:
            st.metric("Worst Case (P90)", f"{p90:.0f} min")
