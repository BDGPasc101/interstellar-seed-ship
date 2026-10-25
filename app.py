import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from physics.relativistic import travel_times, kinetic_energy_joules, energy_to_tnt_megatons, lorentz_factor
from physics.impacts import mission_survival_probability, ALLOY_PROPERTIES

st.set_page_config(
    page_title="Relativistic Seed Ship Simulator",
    page_icon="??",
    layout="wide"
)

@st.cache_data
def load_target_data():
    return pd.read_csv("data/seed_ship_targets.csv")

st.title("?? Relativistic Seed Ship: Interstellar Impact & Survival Simulator")
st.markdown(
    "Model interstellar transit hazards, relativistic kinetic energy transfers, "
    "and hull breach probabilities against dust in the Interstellar Medium (ISM)."
)

try:
    df_targets = load_target_data()
except FileNotFoundError:
    st.error("Target data not found. Run `python fetch_relativistic_targets.py` first.")
    st.stop()

# --- SIDEBAR: MISSION CONFIGURATION ---
st.sidebar.header("Mission Architecture")

# 1. Target Selection
target_names = df_targets['pl_name'].tolist()
selected_target_name = st.sidebar.selectbox("?? Target Destination", target_names)
target = df_targets[df_targets['pl_name'] == selected_target_name].iloc[0]

# 2. Velocity Configuration
beta = st.sidebar.slider(
    "Cruise Velocity (? = v/c)",
    min_value=0.01,
    max_value=0.95,
    value=0.20,
    step=0.01,
    help="Fraction of light speed. Higher speeds increase time dilation but exponentially elevate impact kinetic energies."
)

st.sidebar.header("Vessel & Armor Configuration")

# 3. Ship Geometry
ship_radius = st.sidebar.slider(
    "Bussard Shield / Nose Cone Radius (m)",
    min_value=0.5,
    max_value=25.0,
    value=4.0,
    step=0.5
)
frontal_area_m2 = np.pi * (ship_radius ** 2)

# 4. Shield Materials & Thickness
alloy_options = list(ALLOY_PROPERTIES.keys())
selected_alloy = st.sidebar.selectbox("Whipple Shield Alloy", alloy_options)

armor_thickness_cm = st.sidebar.slider(
    "Shield Thickness (cm)",
    min_value=0.5,
    max_value=50.0,
    value=5.0,
    step=0.5
)
armor_thickness_m = armor_thickness_cm / 100.0

# --- CORE CALCULATIONS ---
times = travel_times(target['dist_ly'], beta)
survival_metrics = mission_survival_probability(
    dist_ly=target['dist_ly'],
    beta=beta,
    ship_frontal_area_m2=frontal_area_m2,
    alloy_name=selected_alloy,
    thickness_m=armor_thickness_m,
    ism_density_multiplier=target['ism_density_multiplier']
)

# --- KEY METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Distance", f"{target['dist_ly']:.2f} ly")
col2.metric("Earth Frame Time", f"{times['coord_time_years']:.1f} yrs")
col3.metric("Ship Frame Time (Proper)", f"{times['proper_time_years']:.1f} yrs", delta=f"-{(times['coord_time_years'] - times['proper_time_years']):.1f} yrs")
col4.metric("Lorentz Factor (?)", f"{times['gamma']:.4f}")

st.divider()

# --- SURVIVAL & IMPACT BANNER ---
col_surv_metric, col_shield_metric = st.columns(2)

with col_surv_metric:
    st.subheader("Mission Survival Verdict")
    p_surv = survival_metrics['survival_prob']
    st.markdown(f"**Survival Probability:** `{p_surv * 100:.2f}%`")
    st.progress(max(0.0, min(1.0, p_surv)))
    
    if p_surv >= 0.90:
        st.success("Nominal flight path: Shielding is adequate for expected ISM density.")
    elif p_surv >= 0.50:
        st.warning("High hazard corridor: Significant risk of a catastrophic micro-impact breach.")
    else:
        st.error("Fatal trajectory: Statistically guaranteed shield breach and vehicle loss.")

with col_shield_metric:
    st.subheader("Target & Shield Telemetry")
    st.markdown(f"**Galactic Latitude (b):** `{target['gal_b_deg']:.2f}°`")
    st.markdown(f"**ISM Dust Multiplier:** `{target['ism_density_multiplier']:.2f}x baseline`")
    st.markdown(f"**Expected Lethal Impacts (?):** `{survival_metrics['lambda_fatal']:.3e}`")
    st.markdown(f"**Minimum Lethal Grain Radius:** `{survival_metrics['r_lethal_microns']:.2f} µm`")

st.divider()

# --- ANALYTICAL PLOTS ---
tab_curves, tab_comparison = st.tabs(["?? Velocity Trade-Off Analysis", "?? Target System Comparison"])

with tab_curves:
    # Generate beta sweep from 0.05 to 0.90
    beta_range = np.linspace(0.05, 0.90, 85)
    curve_data = []

    for b in beta_range:
        t_res = travel_times(target['dist_ly'], b)
        s_res = mission_survival_probability(
            target['dist_ly'], b, frontal_area_m2, selected_alloy, armor_thickness_m, target['ism_density_multiplier']
        )
        curve_data.append({
            "beta": b,
            "ship_years": t_res["proper_time_years"],
            "earth_years": t_res["coord_time_years"],
            "survival_prob": s_res["survival_prob"] * 100.0
        })

    df_curves = pd.DataFrame(curve_data)

    fig_tradeoff = go.Figure()

    # Left Y-axis: Survival Probability
    fig_tradeoff.add_trace(go.Scatter(
        x=df_curves["beta"],
        y=df_curves["survival_prob"],
        name="Survival Probability (%)",
        line=dict(color="#00FFAA", width=3)
    ))

    # Right Y-axis: Transit Time (Ship Frame)
    fig_tradeoff.add_trace(go.Scatter(
        x=df_curves["beta"],
        y=df_curves["ship_years"],
        name="Ship Frame Transit Time (Years)",
        line=dict(color="#FF8800", width=2, dash="dash"),
        yaxis="y2"
    ))

    fig_tradeoff.update_layout(
        title="The Relativistic Dilemma: Transit Duration vs. Survival Probability",
        xaxis=dict(title="Velocity (? = v/c)"),
        yaxis=dict(title="Survival Probability (%)", range=[0, 105]),
        yaxis2=dict(
            title="Ship Time (Years)",
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0.05, y=0.95),
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)

with tab_comparison:
    st.subheader("Cross-Target Trajectory Risk")
    st.markdown("Comparison across all 10 target destinations given current vessel shielding and velocity settings.")
    
    comp_list = []
    for _, row in df_targets.iterrows():
        s_res = mission_survival_probability(
            row['dist_ly'], beta, frontal_area_m2, selected_alloy, armor_thickness_m, row['ism_density_multiplier']
        )
        t_res = travel_times(row['dist_ly'], beta)
        comp_list.append({
            "Target": row['pl_name'],
            "Distance (ly)": row['dist_ly'],
            "Galactic Lat (b°)": row['gal_b_deg'],
            "ISM Mult": row['ism_density_multiplier'],
            "Survival (%)": s_res['survival_prob'] * 100.0,
            "Ship Travel Time (yrs)": t_res['proper_time_years']
        })
    
    df_comp = pd.DataFrame(comp_list).sort_values("Survival (%)", ascending=False)
    
    fig_bar = px.bar(
        df_comp,
        x="Target",
        y="Survival (%)",
        color="ISM Mult",
        color_continuous_scale="Viridis_r",
        hover_data=["Distance (ly)", "Galactic Lat (b°)", "Ship Travel Time (yrs)"],
        labels={"ISM Mult": "ISM Multiplier"}
    )
    fig_bar.update_layout(template="plotly_dark", yaxis_range=[0, 105], height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.dataframe(df_comp.style.format({
        "Distance (ly)": "{:.2f}",
        "Galactic Lat (b°)": "{:.2f}",
        "ISM Mult": "{:.2f}",
        "Survival (%)": "{:.2f}",
        "Ship Travel Time (yrs)": "{:.1f}"
    }))