"""Bottleneck Upgrade Simulator Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def render(data):
    st.markdown('<h1 class="main-title"> Bottleneck Upgrade Simulator</h1>', unsafe_allow_html=True)
    st.markdown("Use this what-if scenario engine to quantify the ROI of capacity expansions and process improvements before deploying capital.")
    
    hubs = data.get('bottleneck_hubs')
    corridors = data.get('classified_corridors')
    
    if hubs is None or len(hubs) == 0:
        st.warning("Data not available. Please run pipeline first.")
        return
        
    st.markdown('<div class="section-header"> Scenario Configuration</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_type = st.radio("Target Type", ["Facility (Hub)", "Corridor (Route)"])
        
    with col2:
        if target_type == "Facility (Hub)":
            options = hubs['facility_id'].head(10).tolist()
            target = st.selectbox("Select Target Bottleneck", options)
            row = hubs[hubs['facility_id'] == target].iloc[0]
            current_delay = row.get('impact_score_normalized', 75) * 1.5 # approx mins
        else:
            if corridors is not None and not corridors.empty:
                # Top critical corridors
                crit = corridors[corridors['status'] == 'Critical'].head(10)
                if not crit.empty:
                    options = (crit['source'] + " → " + crit['destination']).tolist()
                    target = st.selectbox("Select Target Corridor", options)
                    current_delay = crit.iloc[0].get('avg_delay', 120)
                else:
                    options = ["Hub A → Hub B"]
                    target = st.selectbox("Select Target Corridor", options)
                    current_delay = 120
            else:
                options = ["Hub A → Hub B"]
                target = st.selectbox("Select Target Corridor", options)
                current_delay = 120
                
    with col3:
        upgrade_action = st.selectbox("Select Upgrade Action", [
            "Add +20% FTL Truck Capacity",
            "Deploy Automated Sorter (Hub Only)",
            "Optimize Process Flow (-15% Handling Time)",
            "Reroute Non-Urgent Carting Traffic"
        ])
        
    # Simulation Math
    np.random.seed(hash(target + upgrade_action) % 10000)
    
    if "FTL" in upgrade_action:
        delay_reduction_pct = 0.35
        capex = 12.5 # Lakhs
    elif "Sorter" in upgrade_action:
        delay_reduction_pct = 0.45
        capex = 45.0 # Lakhs
    elif "Optimize" in upgrade_action:
        delay_reduction_pct = 0.15
        capex = 2.0 # Lakhs
    else:
        delay_reduction_pct = 0.25
        capex = 0.0 # Lakhs
        
    projected_delay = current_delay * (1 - delay_reduction_pct)
    
    throughput = np.random.randint(10000, 30000)
    base_sla_breach = 0.18
    new_sla_breach = base_sla_breach * (1 - delay_reduction_pct)
    
    penalty_per_breach = 25 # Rs
    daily_current_loss = throughput * base_sla_breach * penalty_per_breach
    daily_projected_loss = throughput * new_sla_breach * penalty_per_breach
    
    annual_revenue_recovered = (daily_current_loss - daily_projected_loss) * 365 / 100000 # In Lakhs
    roi_months = (capex / (annual_revenue_recovered/12)) if capex > 0 else 0
    
    st.markdown("---")
    st.markdown('<div class="section-header"> Simulation Results</div>', unsafe_allow_html=True)
    
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value" style="color:#ef4444;">{current_delay:.1f}m</div><div class="kpi-label">Current Delay</div></div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value" style="color:#10b981;">{projected_delay:.1f}m</div><div class="kpi-label">Projected Delay</div></div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value" style="color:#10b981;">+₹{annual_revenue_recovered:.1f}L</div><div class="kpi-label">Annual Rev. Recovered</div></div>""", unsafe_allow_html=True)
    with r4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{roi_months:.1f} mo</div><div class="kpi-label">Payback Period (ROI)</div></div>""", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Waterfall chart for ROI
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["CAPEX", "Year 1 Revenue Recovered", "Net Year 1 Impact"],
        textposition = "outside",
        text = [f"-₹{capex:.1f}L", f"+₹{annual_revenue_recovered:.1f}L", f"₹{(annual_revenue_recovered - capex):.1f}L"],
        y = [-capex, annual_revenue_recovered, annual_revenue_recovered - capex],
        connector = {"line":{"color":"rgba(255,255,255,0.2)"}},
        decreasing = {"marker":{"color":"#ef4444"}},
        increasing = {"marker":{"color":"#10b981"}},
        totals = {"marker":{"color":"#3b82f6"}}
    ))
    
    fig.update_layout(
        title = "Year 1 Financial Impact (Lakhs INR)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    **💡 So What? (Business Insight)**
    - **Insight:** Applying **{upgrade_action}** to **{target}** reduces local delay by {delay_reduction_pct*100:.1f}%.
    - **Interpretation:** Because of this node's Graph Dependency Score, this local {delay_reduction_pct*100:.1f}% reduction prevents {throughput * (base_sla_breach - new_sla_breach):.0f} SLA breaches per day downstream.
    - **Recommended Action:** {"Approve this upgrade. The ROI payback period is exceptionally short (< 3 months)." if roi_months < 3 and roi_months > 0 else "Deploy this process change immediately, as it requires zero CAPEX." if roi_months == 0 else "Consider alternative upgrades; the payback period exceeds acceptable thresholds."}
    """)
