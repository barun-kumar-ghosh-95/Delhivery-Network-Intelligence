"""
Delhivery Network Intelligence Dashboard
==========================================
Production-grade Streamlit dashboard with 6 pages.
"""

import streamlit as st
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f2e 0%, #1a1a4a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #818cf8;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .main-title {
        background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .kpi-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    .section-header {
        color: #e2e8f0;
        font-size: 1.4rem;
        font-weight: 700;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .badge-healthy { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-moderate { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-severe { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-critical { background: rgba(220, 38, 38, 0.3); color: #fca5a5; border: 1px solid rgba(220, 38, 38, 0.4); }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stSelectbox > div > div {
        background: rgba(30, 30, 60, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 15, 40, 0.5);
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: #818cf8;
        background: rgba(99, 102, 241, 0.15);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
from streamlit_option_menu import option_menu

with st.sidebar:
    st.markdown('<h1 style="text-align: center; color: #818cf8; margin-bottom: 2rem;">Delhivery Intelligence</h1>', unsafe_allow_html=True)
    
    page = option_menu(
        menu_title=None,
        options=[
            "Network Overview",
            "Why Graph Wins",
            "ETA Predictor",
            "Bottleneck Monitor",
            "Corridor Intelligence",
            "FTL vs Carting",
            "Network Operations Command Center"
        ],
        icons=[
            'globe', 
            'diagram-3', 
            'clock-history', 
            'exclamation-triangle', 
            'signpost-split', 
            'truck', 
            'briefcase'
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#818cf8", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin": "0px", 
                "--hover-color": "rgba(99, 102, 241, 0.1)",
                "color": "#cbd5e1"
            },
            "nav-link-selected": {
                "background-color": "rgba(99, 102, 241, 0.2)", 
                "color": "#ffffff",
                "border-left": "4px solid #818cf8",
                "font-weight": "600"
            },
        }
    )

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.75rem;'>
    <p>Delhivery Network Intelligence v1.0</p>
    <p>Graph-Based ETA Optimization</p>
</div>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def load_dashboard_data():
    """Load all pre-computed data for dashboard."""
    import json
    data = {}
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    analysis_dir = os.path.join(processed_dir, 'analysis')
    
    # Dashboard summary
    try:
        with open(os.path.join(processed_dir, 'dashboard_data.json'), 'r') as f:
            data['summary'] = json.load(f)
    except:
        data['summary'] = {}
    
    # Model comparison
    try:
        import pandas as pd
        data['model_comparison'] = pd.read_csv(os.path.join(processed_dir, 'model_comparison.csv'))
    except:
        data['model_comparison'] = None
    
    # Corridor stats
    try:
        data['corridor_stats'] = pd.read_csv(os.path.join(processed_dir, 'corridor_stats.csv'))
    except:
        data['corridor_stats'] = None
    
    # Facility stats
    try:
        data['facility_stats'] = pd.read_csv(os.path.join(processed_dir, 'facility_stats.csv'))
    except:
        data['facility_stats'] = None
    
    # Impact scores
    try:
        data['impact_scores'] = pd.read_csv(os.path.join(analysis_dir, 'facility_impact_scores.csv'))
    except:
        data['impact_scores'] = None
    
    # Classified corridors
    try:
        data['classified_corridors'] = pd.read_csv(os.path.join(analysis_dir, 'classified_corridors.csv'))
    except:
        data['classified_corridors'] = None
    
    # Bottleneck data
    try:
        data['bottleneck_hubs'] = pd.read_csv(os.path.join(analysis_dir, 'bottleneck_hubs.csv'))
        data['bottleneck_corridors'] = pd.read_csv(os.path.join(analysis_dir, 'bottleneck_corridors.csv'))
    except:
        data['bottleneck_hubs'] = None
        data['bottleneck_corridors'] = None
    
    # Network summary
    try:
        with open(os.path.join(analysis_dir, 'network_summary.json'), 'r') as f:
            data['network_summary'] = json.load(f)
    except:
        data['network_summary'] = {}
    
    # Delay summary
    try:
        with open(os.path.join(analysis_dir, 'delay_summary.json'), 'r') as f:
            data['delay_summary'] = json.load(f)
    except:
        data['delay_summary'] = {}
    
    # Trip data for predictor
    try:
        data['test_trips'] = pd.read_csv(os.path.join(processed_dir, 'test_trips_featured.csv'))
    except:
        data['test_trips'] = None
    
    return data

data = load_dashboard_data()

# ============================================================
# PAGE ROUTING
# ============================================================

if page == "Network Overview":
    from dashboard.pages.network_overview import render
    render(data)
elif page == "Why Graph Wins":
    from dashboard.pages.why_graph_wins import render
    render(data)
elif page == "ETA Predictor":
    from dashboard.pages.eta_predictor import render
    render(data)
elif page == "Bottleneck Monitor":
    from dashboard.pages.bottleneck_monitor import render
    render(data)
elif page == "Corridor Intelligence":
    from dashboard.pages.corridor_intelligence import render
    render(data)
elif page == "FTL vs Carting":
    from dashboard.pages.ftl_carting_advisor import render
    render(data)
elif page == "Network Operations Command Center":
    from dashboard.pages.executive_insights import render
    render(data)
