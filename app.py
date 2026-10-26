import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from physics.relativistic import travel_times, kinetic_energy_joules, energy_to_tnt_megatons, lorentz_factor
from physics.impacts import (
    mission_survival_probability,
    ALLOY_PROPERTIES,
    SHIELD_ARCHITECTURES,
    calculate_shield_geometry,
    calculate_economic_costs
)

st.set_page_config(
    page_title="Relativistic Seed Ship Simulator",
    layout="wide"
)

@st.cache_data
def load_target_data():
    return pd.read_csv("data/seed_ship_targets.csv")

st.title("Relativistic Seed Ship: Interstellar Impact & Survival Simulator")
st.markdown(
    "Model interstellar transit hazards, relativistic kinetic energy transfers, "
    "and hull breach probabilities against dust in the Interstellar Medium (ISM)."
)

try:
    df_targets = load_target_data()
except FileNotFoundError:
    st.error("Target data not found. Run python fetch_relativistic_targets.py first.")
    st.stop()

# --- SIDEBAR: MISSION CONFIGURATION ---
st.sidebar.header("Mission Architecture")

# 1. Target Selection
target_names = df_targets['pl_name'].tolist()
selected_target_name = st.sidebar.selectbox("Target Destination", target_names)
target = df_targets[df_targets['pl_name'] == selected_target_name].iloc[0]

# 2. Velocity Configuration
beta = st.sidebar.slider(
    "Cruise Velocity (beta = v/c)",
    min_value=0.01,
    max_value=0.95,
    value=0.14,
    step=0.01,
    help="Fraction of light speed. Higher speeds increase time dilation but exponentially elevate impact kinetic energies."
)

st.sidebar.header("Shield Architecture & Materials")

# Architecture Selection
selected_architecture = st.sidebar.selectbox(
    "Shield Construction",
    SHIELD_ARCHITECTURES,
    help="Uniform: constant shell. Graded: reinforced apex/rim. Solid: solid cone plug."
)

# Ship Geometry
ship_radius = st.sidebar.slider(
    "Bussard Shield / Nose Cone Radius (m)",
    min_value=0.5,
    max_value=25.0,
    value=4.5,
    step=0.5
)
frontal_area_m2 = np.pi * (ship_radius ** 2)

cone_half_angle = st.sidebar.slider(
    "Cone Half-Angle (alpha in deg)",
    min_value=5.0,
    max_value=90.0,
    value=10.0,
    step=1.0,
    help="Half-angle of the conical nosecone. Sharper angles (smaller values) deflect relativistic kinetic energy: E_normal = E_k * sin^2(alpha)."
)

alloy_options = list(ALLOY_PROPERTIES.keys())
selected_alloy = st.sidebar.selectbox("Whipple Shield Alloy", alloy_options)

armor_thickness_cm = st.sidebar.slider(
    "Base Shell Thickness (cm)",
    min_value=0.5,
    max_value=100.0,
    value=50.0,
    step=0.5,
    disabled=(selected_architecture == "Solid Nosecone (Machined Slug)"),
    help="Disabled for Solid Nosecone because thickness is determined entirely by cone radius and angle."
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
    ism_density_multiplier=target['ism_density_multiplier'],
    cone_half_angle_deg=cone_half_angle,
    architecture=selected_architecture
)

# --- KEY METRICS ROW ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Distance", f"{target['dist_ly']:.2f} ly")
col2.metric("Earth Frame Time", f"{times['coord_time_years']:.1f} yrs")
col3.metric(
    "Ship Frame Time",
    f"{times['proper_time_years']:.1f} yrs",
    delta=f"-{(times['coord_time_years'] - times['proper_time_years']):.1f} yrs"
)
col4.metric("Lorentz Factor (gamma)", f"{times['gamma']:.4f}")
col5.metric("Shield Mass", f"{survival_metrics['shield_mass_tons']:,.1f} t")

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
    st.markdown(f"**Galactic Latitude (b):** `{target['gal_b_deg']:.2f} deg`")
    st.markdown(f"**ISM Dust Multiplier:** `{target['ism_density_multiplier']:.2f}x baseline`")
    st.markdown(f"**Axial Cone Length:** `{survival_metrics['axial_length_m']:.2f} m`")
    st.markdown(f"**Effective Mean Thickness:** `{survival_metrics['effective_thickness_cm']:.2f} cm`")
    st.markdown(f"**Expected Lethal Impacts (lambda):** `{survival_metrics['lambda_fatal']:.3e}`")
    st.markdown(f"**Minimum Lethal Grain Radius:** `{survival_metrics['r_lethal_microns']:.2f} um`")

st.divider()

# --- ANALYTICAL PLOTS ---
tab_curves, tab_comparison, tab_shield_design = st.tabs([
    "Velocity Trade-Off Analysis",
    "Target System Comparison",
    "Shield Geometry & Mass Budget"
])

with tab_curves:
    beta_range = np.linspace(0.02, 0.85, 80)
    curve_data = []

    for b in beta_range:
        t_res = travel_times(target['dist_ly'], b)
        s_res = mission_survival_probability(
            dist_ly=target['dist_ly'],
            beta=b,
            ship_frontal_area_m2=frontal_area_m2,
            alloy_name=selected_alloy,
            thickness_m=armor_thickness_m,
            ism_density_multiplier=target['ism_density_multiplier'],
            cone_half_angle_deg=cone_half_angle,
            architecture=selected_architecture
        )
        curve_data.append({
            "beta": b,
            "ship_years": t_res["proper_time_years"],
            "survival_prob": s_res["survival_prob"] * 100.0
        })

    df_curves = pd.DataFrame(curve_data)

    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(go.Scatter(
        x=df_curves["beta"],
        y=df_curves["survival_prob"],
        name="Survival Probability (%)",
        line=dict(color="#00FFAA", width=3)
    ))
    fig_tradeoff.add_trace(go.Scatter(
        x=df_curves["beta"],
        y=df_curves["ship_years"],
        name="Ship Frame Transit Time (Years)",
        line=dict(color="#FF8800", width=2, dash="dash"),
        yaxis="y2"
    ))

    fig_tradeoff.update_layout(
        title="The Relativistic Dilemma: Transit Duration vs. Survival Probability",
        xaxis=dict(title="Velocity (beta = v/c)"),
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
    comp_list = []
    for _, row in df_targets.iterrows():
        s_res = mission_survival_probability(
            dist_ly=row['dist_ly'],
            beta=beta,
            ship_frontal_area_m2=frontal_area_m2,
            alloy_name=selected_alloy,
            thickness_m=armor_thickness_m,
            ism_density_multiplier=row['ism_density_multiplier'],
            cone_half_angle_deg=cone_half_angle,
            architecture=selected_architecture
        )
        t_res = travel_times(row['dist_ly'], beta)
        comp_list.append({
            "Target": row['pl_name'],
            "Distance (ly)": row['dist_ly'],
            "Galactic Lat (deg)": row['gal_b_deg'],
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
        hover_data=["Distance (ly)", "Galactic Lat (deg)", "Ship Travel Time (yrs)"],
        labels={"ISM Mult": "ISM Multiplier"}
    )
    fig_bar.update_layout(template="plotly_dark", yaxis_range=[0, 105], height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.dataframe(df_comp.style.format({
        "Distance (ly)": "{:.2f}",
        "Galactic Lat (deg)": "{:.2f}",
        "ISM Mult": "{:.2f}",
        "Survival (%)": "{:.2f}",
        "Ship Travel Time (yrs)": "{:.1f}"
    }))

with tab_shield_design:
    st.subheader("Needle Hull Profile & Shield Cross-Section")
    
    geom = calculate_shield_geometry(
        ship_radius, cone_half_angle, armor_thickness_m, selected_architecture, selected_alloy
    )

    cone_x = [geom["axial_length_m"], 0.0, geom["axial_length_m"]]
    cone_y = [-ship_radius, 0.0, ship_radius]

    fig_geom = go.Figure()

    fig_geom.add_trace(go.Scatter(
        x=cone_x,
        y=cone_y,
        mode="lines",
        name="Outer Deflection Surface",
        line=dict(color="#00FFAA", width=3),
        fill="toself",
        fillcolor="rgba(0, 255, 170, 0.15)"
    ))

    fig_geom.add_trace(go.Scatter(
        x=[-geom["axial_length_m"] * 0.2, geom["axial_length_m"] * 1.2],
        y=[0, 0],
        mode="lines",
        name="Velocity Vector (Cruise Axis)",
        line=dict(color="#FF4444", width=1, dash="dot")
    ))

    fig_geom.update_layout(
        title=f"Shield Profile: {selected_architecture} (Aspect Ratio 1:1)",
        xaxis=dict(title="Axial Length (meters)", autorange="reversed"),
        yaxis=dict(title="Radial Radius (meters)", scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_geom, use_container_width=True)

    alloy_masses = []
    for alloy in ALLOY_PROPERTIES.keys():
        g = calculate_shield_geometry(ship_radius, cone_half_angle, armor_thickness_m, selected_architecture, alloy)
        alloy_masses.append({
            "Alloy": alloy,
            "Mass (Metric Tons)": g["mass_metric_tons"],
            "Density (kg/m3)": ALLOY_PROPERTIES[alloy]["density"]
        })
    df_masses = pd.DataFrame(alloy_masses)

    col_mass_plot, col_mass_info = st.columns([2, 1])
    with col_mass_plot:
        fig_mass = px.bar(
            df_masses,
            x="Alloy",
            y="Mass (Metric Tons)",
            color="Alloy",
            title="Total Shield Deadweight Mass by Material"
        )
        fig_mass.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig_mass, use_container_width=True)

    with col_mass_info:
        st.markdown("#### Dimensional Breakdown")
        st.write(f"- **Axial Length:** `{geom['axial_length_m']:.2f} m`")
        st.write(f"- **Slant Height:** `{geom['slant_height_m']:.2f} m`")
        st.write(f"- **Displaced Volume:** `{geom['volume_m3']:.2f} m^3`")
        st.write(f"- **Apex Thickness:** `{geom['apex_thickness_m'] * 100:.1f} cm`")
        st.write(f"- **Base Thickness:** `{geom['base_thickness_m'] * 100:.1f} cm`")

    st.divider()
    st.subheader("Shield Fabrication & Gravity Well Escape Fuel")

    costs = calculate_economic_costs(geom["mass_kg"], selected_alloy)

    col_cost1, col_cost2, col_cost3 = st.columns(3)
    col_cost1.metric("Cone Material Fabrication", f"${costs['cone_fab_cost_usd'] / 1e6:,.2f} M")
    col_cost2.metric("Earth Escape Propellant Mass", f"{costs['escape_propellant_metric_tons']:,.1f} t")
    col_cost3.metric("Escape Propellant Fuel Cost", f"${costs['escape_fuel_cost_usd'] / 1e3:,.1f} k")

    st.info(
        f"Fabricating a `{survival_metrics['shield_mass_tons']:,.1f} metric ton` shield from **{selected_alloy}** "
        f"requires an estimated `${costs['cone_fab_cost_usd'] / 1e6:,.2f} Million USD` in material processing. "
        f"Pushing this shield mass from Low Earth Orbit out of Earth's gravity well requires "
        f"`{costs['escape_propellant_metric_tons']:,.1f} metric tons` of Methalox propellant."
    )