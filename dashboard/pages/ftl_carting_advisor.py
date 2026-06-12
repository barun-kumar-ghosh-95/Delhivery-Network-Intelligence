"""FTL vs Carting Advisor Page"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render(data):
    st.markdown('<h1 class="main-title"> FTL vs Carting Advisor</h1>', unsafe_allow_html=True)
    
    summary = data.get('summary', {})
    corridors = data.get('classified_corridors') or data.get('corridor_stats')
    test_trips = data.get('test_trips')
    
    acc = summary.get('ftl_accuracy', 0)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{acc*100 if acc<1 else acc:.1f}%</div><div class="kpi-label">Classifier Accuracy</div></div>""", unsafe_allow_html=True)
    with c2:
        ftl_pct = 68.8
        if test_trips is not None and 'is_ftl' in test_trips.columns:
            ftl_pct = test_trips['is_ftl'].mean() * 100
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{ftl_pct:.1f}%</div><div class="kpi-label">FTL Share</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{100-ftl_pct:.1f}%</div><div class="kpi-label">Carting Share</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if test_trips is not None and 'actual_time' in test_trips.columns and 'is_ftl' in test_trips.columns:
        col1, col2 = st.columns(2)
        with col1:
            ftl = test_trips[test_trips['is_ftl']==1]['actual_time'].dropna()
            cart = test_trips[test_trips['is_ftl']==0]['actual_time'].dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ftl.clip(upper=2000), name='FTL', marker_color='#818cf8', opacity=0.7))
            fig.add_trace(go.Histogram(x=cart.clip(upper=2000), name='Carting', marker_color='#f59e0b', opacity=0.7))
            fig.update_layout(title="Delivery Time by Route Type", xaxis_title="Time (min)", barmode='overlay',
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            stats = test_trips.groupby('is_ftl').agg(
                avg_time=('actual_time','mean'), median_time=('actual_time','median'),
                avg_distance=('osrm_distance','mean'), count=('actual_time','count')
            ).reset_index()
            stats['route_type'] = stats['is_ftl'].map({1:'FTL', 0:'Carting'})
            fig = go.Figure(data=[
                go.Bar(name='Avg Time', x=stats['route_type'], y=stats['avg_time'], marker_color='#818cf8'),
                go.Bar(name='Median Time', x=stats['route_type'], y=stats['median_time'], marker_color='#a78bfa'),
            ])
            fig.update_layout(title="FTL vs Carting Performance", barmode='group',
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="section-header"> Route Type Comparison</div>', unsafe_allow_html=True)
        comp = test_trips.groupby('is_ftl').agg({
            'actual_time': ['mean','median','std'],
            'osrm_time': 'mean',
            'osrm_distance': 'mean',
            'num_segments': 'mean',
        }).round(2)
        comp.index = ['Carting', 'FTL']
        st.dataframe(comp, use_container_width=True)
    
    st.markdown('<div class="section-header"> Recommendation Engine</div>', unsafe_allow_html=True)
    st.info("""
    **FTL is recommended when:**
    - Distance > 200 km (long-haul)
    - Time-sensitive shipments
    - High-volume corridors with reliable schedules
    
    **Carting is recommended when:**
    - Distance < 100 km (last-mile)
    - Multi-stop deliveries in urban areas
    - Cost optimization over speed
    """)
