import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from physics.relativistic import (
    travel_times,
    kinetic_energy_joules,
    energy_to_tnt_megatons,
    lorentz_factor,
    optimize_cruise_velocity
)
from physics.impacts import (
    mission_survival_probability,
    mission_survival_with_degradation,
    ALLOY_PROPERTIES,
    SHIELD_ARCHITECTURES,
    calculate_shield_geometry,
    calculate_economic_costs,
    run_monte_carlo_simulation
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

st.sidebar.header("Mission Architecture")

target_names = df_targets['pl_name'].tolist()
selected_target_name = st.sidebar.selectbox("Target Destination", target_names)
target = df_targets[df_targets['pl_name'] == selected_target_name].iloc[0]

beta = st.sidebar.slider(
    "Cruise Velocity (beta = v/c)",
    min_value=0.01,
    max_value=0.95,
    value=0.14,
    step=0.01
)

st.sidebar.header("Shield Architecture & Materials")

selected_architecture = st.sidebar.selectbox(
    "Shield Construction",
    SHIELD_ARCHITECTURES
)

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
    step=1.0
)

alloy_options = list(ALLOY_PROPERTIES.keys())
selected_alloy = st.sidebar.selectbox("Whipple Shield Alloy", alloy_options)

armor_thickness_cm = st.sidebar.slider(
    "Base Shell Thickness (cm)",
    min_value=0.5,
    max_value=100.0,
    value=50.0,
    step=0.5,
    disabled=(selected_architecture == "Solid Nosecone (Machined Slug)")
)
armor_thickness_m = armor_thickness_cm / 100.0

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

deg_metrics = mission_survival_with_degradation(
    dist_ly=target['dist_ly'],
    beta=beta,
    ship_frontal_area_m2=frontal_area_m2,
    alloy_name=selected_alloy,
    thickness_m=armor_thickness_m,
    ism_density_multiplier=target['ism_density_multiplier'],
    cone_half_angle_deg=cone_half_angle,
    architecture=selected_architecture
)

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

col_surv_metric, col_shield_metric = st.columns(2)

with col_surv_metric:
    st.subheader("Mission Survival Verdict")
    p_surv = deg_metrics['survival_prob_degraded']
    st.markdown(f"**Degraded Survival Probability:** `{p_surv * 100:.2f}%`")
    st.progress(max(0.0, min(1.0, p_surv)))
    
    if p_surv >= 0.90:
        st.success("Nominal flight path: Shielding is adequate for expected ISM density.")
    elif p_surv >= 0.50:
        st.warning("High hazard corridor: Significant risk of a catastrophic micro-impact breach.")
    else:
        st.error("Fatal trajectory: Statistically guaranteed shield breach and vehicle loss.")

with col_shield_metric:
    st.subheader("Dynamic Shield Ablation Telemetry")
    col_ab1, col_ab2 = st.columns(2)
    col_ab1.metric("Initial Armor", f"{deg_metrics['initial_thickness_cm']:.1f} cm")
    col_ab2.metric("Armor Lost to Dust", f"-{deg_metrics['total_erosion_cm']:.1f} cm")
    
    col_ab3, col_ab4 = st.columns(2)
    col_ab3.metric("Final Armor Remaining", f"{deg_metrics['final_thickness_cm']:.1f} cm")
    col_ab4.metric("Degradation Loss", f"{deg_metrics['armor_loss_pct']:.1f}%")

st.divider()

tab_curves, tab_comparison, tab_shield_design, tab_monte_carlo = st.tabs([
    "Velocity Trade-Off Analysis",
    "Target System Comparison",
    "Shield Geometry & Mass Budget",
    "Monte Carlo Hazard Simulation"
])

with tab_curves:
    beta_range = np.linspace(0.02, 0.85, 80)
    curve_data = []

    for b in beta_range:
        t_res = travel_times(target['dist_ly'], b)
        s_res = mission_survival_with_degradation(
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
            "survival_prob": s_res["survival_prob_degraded"] * 100.0
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
        title="The Relativistic Dilemma: Transit Duration vs. Survival Probability (w/ Erosion)",
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

    st.divider()
    st.subheader("SciPy Optimal Trajectory Solver")
    col_opt_w1, col_opt_w2 = st.columns(2)
    w_time = col_opt_w1.slider("Weight: Transit Time Minimization", 0.1, 10.0, 1.0, 0.1)
    w_risk = col_opt_w2.slider("Weight: Breach Risk Minimization", 0.1, 10.0, 5.0, 0.1)

    if st.button("Calculate Optimal Cruise Velocity"):
        with st.spinner("Executing SciPy bounded scalar minimization..."):
            opt = optimize_cruise_velocity(
                dist_ly=target['dist_ly'],
                ship_frontal_area_m2=frontal_area_m2,
                alloy_name=selected_alloy,
                thickness_m=armor_thickness_m,
                ism_density_multiplier=target['ism_density_multiplier'],
                cone_half_angle_deg=cone_half_angle,
                architecture=selected_architecture,
                weight_time=w_time,
                weight_risk=w_risk
            )
        
        if opt["success"]:
            st.success(f"Optimal Cruise Velocity Found: **beta = {opt['optimal_beta']:.4f}c**")
            col_o1, col_o2, col_o3 = st.columns(3)
            col_o1.metric("Optimal Ship Time", f"{opt['optimal_ship_years']:.1f} yrs")
            col_o2.metric("Optimal Earth Time", f"{opt['optimal_earth_years']:.1f} yrs")
            col_o3.metric("Optimal Survival Prob", f"{opt['optimal_survival_prob'] * 100:.2f}%")

with tab_comparison:
    st.subheader("Cross-Target Trajectory Risk")
    comp_list = []
    for _, row in df_targets.iterrows():
        s_res = mission_survival_with_degradation(
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
            "Survival (%)": s_res['survival_prob_degraded'] * 100.0,
            "Ship Travel Time (yrs)": t_res['proper_time_years']
        })
    
    df_comp = pd.DataFrame(comp_list).sort_values("Survival (%)", ascending=False)
    fig_bar = px.bar(
        df_comp,
        x="Target",
        y="Survival (%)",
        color="ISM Mult",
        color_continuous_scale="Viridis_r",
        hover_data=["Distance (ly)", "Galactic Lat (deg)", "Ship Travel Time (yrs)"]
    )
    fig_bar.update_layout(template="plotly_dark", yaxis_range=[0, 105], height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_shield_design:
    st.subheader("Needle Hull Profile & Shield Cross-Section")
    geom = calculate_shield_geometry(
        ship_radius, cone_half_angle, armor_thickness_m, selected_architecture, selected_alloy
    )
    cone_x = [geom["axial_length_m"], 0.0, geom["axial_length_m"]]
    cone_y = [-ship_radius, 0.0, ship_radius]

    fig_geom = go.Figure()
    fig_geom.add_trace(go.Scatter(
        x=cone_x, y=cone_y, mode="lines", name="Outer Deflection Surface",
        line=dict(color="#00FFAA", width=3), fill="toself", fillcolor="rgba(0, 255, 170, 0.15)"
    ))
    fig_geom.add_trace(go.Scatter(
        x=[-geom["axial_length_m"] * 0.2, geom["axial_length_m"] * 1.2],
        y=[0, 0], mode="lines", name="Velocity Vector (Cruise Axis)",
        line=dict(color="#FF4444", width=1, dash="dot")
    ))
    fig_geom.update_layout(
        title=f"Shield Profile: {selected_architecture} (Aspect Ratio 1:1)",
        xaxis=dict(title="Axial Length (meters)", autorange="reversed"),
        yaxis=dict(title="Radial Radius (meters)", scaleanchor="x", scaleratio=1),
        template="plotly_dark", height=450
    )
    st.plotly_chart(fig_geom, use_container_width=True)

    alloy_masses = []
    for alloy in ALLOY_PROPERTIES.keys():
        g = calculate_shield_geometry(ship_radius, cone_half_angle, armor_thickness_m, selected_architecture, alloy)
        alloy_masses.append({
            "Alloy": alloy,
            "Mass (Metric Tons)": g["mass_metric_tons"]
        })
    
    col_mass_plot, col_mass_info = st.columns([2, 1])
    with col_mass_plot:
        fig_mass = px.bar(pd.DataFrame(alloy_masses), x="Alloy", y="Mass (Metric Tons)", color="Alloy")
        fig_mass.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig_mass, use_container_width=True)

    with col_mass_info:
        st.markdown("#### Dimensional Breakdown")
        st.write(f"- **Axial Length:** `{geom['axial_length_m']:.2f} m`")
        st.write(f"- **Displaced Volume:** `{geom['volume_m3']:.2f} m^3`")
        st.write(f"- **Apex Thickness:** `{geom['apex_thickness_m'] * 100:.1f} cm`")
        st.write(f"- **Base Thickness:** `{geom['base_thickness_m'] * 100:.1f} cm`")

with tab_monte_carlo:
    st.subheader("Stochastic Flight Pass Simulation (10,000 Independent Trials)")
    st.markdown("Uses distance-based ablation scaling to evaluate hazard survival dynamically across the trajectory.")

    if st.button("Run 10,000 Flight Passes"):
        with st.spinner("Executing Monte Carlo with continuous ablation mechanics..."):
            mc_res = run_monte_carlo_simulation(
                dist_ly=target['dist_ly'], beta=beta, ship_frontal_area_m2=frontal_area_m2,
                alloy_name=selected_alloy, thickness_m=armor_thickness_m,
                ism_density_multiplier=target['ism_density_multiplier'],
                cone_half_angle_deg=cone_half_angle, architecture=selected_architecture,
                n_simulations=10000
            )

        col_mc1, col_mc2, col_mc3, col_mc4 = st.columns(4)
        col_mc1.metric("Flawless Transits", f"{mc_res['flawless_pct']:.1f}%")
        col_mc2.metric("Pitting Damage Only", f"{mc_res['pitting_pct']:.1f}%")
        col_mc3.metric("Catastrophic Breaches", f"{mc_res['breach_pct']:.1f}%")
        col_mc4.metric("Empirical Survival Rate", f"{mc_res['mc_survival_rate']:.1f}%")

        st.divider()

        df_outcomes = pd.DataFrame([
            {"Outcome": "Flawless Transit", "Percentage": mc_res['flawless_pct']},
            {"Outcome": "Pitting Damage", "Percentage": mc_res['pitting_pct']},
            {"Outcome": "Catastrophic Breach", "Percentage": mc_res['breach_pct']}
        ])

        fig_pie = px.pie(
            df_outcomes, values="Percentage", names="Outcome",
            color="Outcome", color_discrete_map={
                "Flawless Transit": "#00FFAA", "Pitting Damage": "#FF8800", "Catastrophic Breach": "#FF3344"
            }
        )
        fig_pie.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Updated Histogram with clearer statistical labeling
        fig_hist = px.histogram(
            x=mc_res["max_depths_cm"], 
            nbins=50,
            title="Worst Single Crater Depth Per Mission (Distribution over 10,000 Flights)",
            labels={"x": "Deepest Crater on Hull (cm)"}
        )
        fig_hist.add_vline(
            x=mc_res["effective_thickness_cm"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Breach Threshold ({mc_res['effective_thickness_cm']:.1f} cm)"
        )
        fig_hist.update_layout(
            template="plotly_dark", 
            height=450,
            yaxis_title="Number of Missions (out of 10,000)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)