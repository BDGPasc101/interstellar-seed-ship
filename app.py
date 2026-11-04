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
    optimize_cruise_velocity,
    ENGINE_TYPES,
    solve_mass_area_feedback,
    SPEED_OF_LIGHT
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
    "hull breach probabilities against dust in the Interstellar Medium (ISM), and coupled propulsion mass budgets."
)

try:
    df_targets = load_target_data()
except FileNotFoundError:
    st.error("Target data not found. Run python fetch_relativistic_targets.py first.")
    st.stop()

# --- SIDEBAR: MISSION CONFIGURATION ---
st.sidebar.header("Mission Architecture")

target_names = df_targets['pl_name'].tolist()
selected_target_name = st.sidebar.selectbox("Target Destination", target_names)
target = df_targets[df_targets['pl_name'] == selected_target_name].iloc[0]

beta = st.sidebar.slider(
    "Cruise Speed (% of Light Speed)",
    min_value=0.01,
    max_value=0.95,
    value=0.14,
    step=0.01,
    help="Represented in physics as 'beta' (v/c). 0.14 is 14% the speed of light."
)

st.sidebar.header("Shield Architecture & Materials")

selected_architecture = st.sidebar.selectbox(
    "Shield Construction",
    SHIELD_ARCHITECTURES
)

ship_radius = st.sidebar.slider(
    "Base Minimum Radius (m)",
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

# --- KEY METRICS ROW ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Distance", f"{target['dist_ly']:.2f} ly")
col2.metric("Time Passed on Earth", f"{times['coord_time_years']:.1f} yrs")
col3.metric(
    "Time Experienced by Crew",
    f"{times['proper_time_years']:.1f} yrs",
    delta=f"Saved {times['coord_time_years'] - times['proper_time_years']:.1f} yrs",
    delta_color="normal"
)
col4.metric(
    "Time Dilation Factor (gamma)", 
    f"{times['gamma']:.4f}",
    help="At 2.0, the crew ages exactly half as fast as the universe around them."
)
col5.metric("Base Shield Mass", f"{survival_metrics['shield_mass_tons']:,.1f} t")

st.divider()

# --- SURVIVAL & IMPACT BANNER ---
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
    st.subheader("Dynamic Shield Health")
    health_pct = max(0.0, min(1.0, deg_metrics['final_thickness_cm'] / deg_metrics['initial_thickness_cm']))
    
    if health_pct > 0.75:
        st.progress(health_pct, text="🟢 Shield Integrity Optimal")
    elif health_pct > 0.25:
        st.progress(health_pct, text="🟡 Heavy Scouring Detected")
    else:
        st.progress(health_pct, text="🔴 CRITICAL: Hull Breaches Imminent")

    col_ab1, col_ab2, col_ab3 = st.columns(3)
    col_ab1.metric("Initial Armor", f"{deg_metrics['initial_thickness_cm']:.1f} cm")
    col_ab2.metric("Armor Ablated", f"-{deg_metrics['total_erosion_cm']:.1f} cm")
    col_ab3.metric("Armor Remaining", f"{deg_metrics['final_thickness_cm']:.1f} cm")

st.divider()

# --- TABS ---
tab_curves, tab_comparison, tab_shield_design, tab_propulsion, tab_monte_carlo = st.tabs([
    "Velocity Trade-Off Analysis",
    "Target System Comparison",
    "Shield Geometry & Mass Budget",
    "Propulsion & Fuel Budget",
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
        x=df_curves["beta"], y=df_curves["survival_prob"],
        name="Survival Probability (%)", line=dict(color="#00FFAA", width=3)
    ))
    fig_tradeoff.add_trace(go.Scatter(
        x=df_curves["beta"], y=df_curves["ship_years"],
        name="Ship Frame Transit Time (Years)", line=dict(color="#FF8800", width=2, dash="dash"),
        yaxis="y2"
    ))

    fig_tradeoff.update_layout(
        title="The Relativistic Dilemma: Transit Duration vs. Survival Probability",
        xaxis=dict(title="Cruise Speed (% of Light Speed)"),
        yaxis=dict(title="Survival Probability (%)", range=[0, 105]),
        yaxis2=dict(title="Crew Time (Years)", overlaying="y", side="right"),
        legend=dict(x=0.05, y=0.95), template="plotly_dark", height=500
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    with st.expander("🤔 What does this chart actually mean?"):
        st.markdown("""
        **The TL;DR:** Going faster means you arrive sooner, but hitting a speck of dust at 90% the speed of light is like being hit by a freight train. 
        
        This chart visualizes the exact "sweet spot" for your mission. You are looking for the point where the crew doesn't die of old age (the orange dashed line is low), but the ship doesn't get vaporized by space dust (the green line is high).
        """)

with tab_comparison:
    st.subheader("Cross-Target Trajectory Risk")
    comp_list = []
    for _, row in df_targets.iterrows():
        s_res = mission_survival_with_degradation(
            dist_ly=row['dist_ly'], beta=beta, ship_frontal_area_m2=frontal_area_m2,
            alloy_name=selected_alloy, thickness_m=armor_thickness_m,
            ism_density_multiplier=row['ism_density_multiplier'],
            cone_half_angle_deg=cone_half_angle, architecture=selected_architecture
        )
        t_res = travel_times(row['dist_ly'], beta)
        comp_list.append({
            "Target": row['pl_name'], "Distance (ly)": row['dist_ly'],
            "Galactic Lat (deg)": row['gal_b_deg'], "ISM Mult": row['ism_density_multiplier'],
            "Survival (%)": s_res['survival_prob_degraded'] * 100.0,
            "Ship Travel Time (yrs)": t_res['proper_time_years']
        })
    
    df_comp = pd.DataFrame(comp_list).sort_values("Survival (%)", ascending=False)
    fig_bar = px.bar(
        df_comp, x="Target", y="Survival (%)", color="ISM Mult",
        color_continuous_scale="Viridis_r", hover_data=["Distance (ly)", "Ship Travel Time (yrs)"]
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
        title=f"Shield Profile: {selected_architecture}",
        xaxis=dict(title="Axial Length (meters)", autorange="reversed"),
        yaxis=dict(title="Radial Radius (meters)", scaleanchor="x", scaleratio=1),
        template="plotly_dark", height=450
    )
    st.plotly_chart(fig_geom, use_container_width=True)

with tab_propulsion:
    st.subheader("The Rocket Equation: Mass & Fuel Convergence")
    st.markdown(
        "More fuel requires wider tanks &rarr; wider tanks require a wider shield "
        "&rarr; a heavier shield requires more fuel."
    )
    
    col_prop1, col_prop2, col_prop3 = st.columns(3)
    payload_mass_tons = col_prop1.slider("Base Payload Mass (Tons)", 100, 20000, 5000, 100)
    engine_choice = col_prop2.selectbox("Main Propulsion System", list(ENGINE_TYPES.keys()), index=1)
    mission_profile = col_prop3.selectbox("Mission Profile", ["Colony (Accel + Decel)", "Flyby (No Deceleration)"])
    
    st.info(f"**{engine_choice}:** {ENGINE_TYPES[engine_choice]['description']} "
            f"(Exhaust Velocity: {ENGINE_TYPES[engine_choice]['ve_c']}c)")
    
    prop_res = solve_mass_area_feedback(
        beta=beta,
        payload_mass_kg=payload_mass_tons * 1000.0,
        engine_type=engine_choice,
        alloy_name=selected_alloy,
        thickness_m=armor_thickness_m,
        cone_half_angle_deg=cone_half_angle,
        architecture=selected_architecture,
        mission_profile=mission_profile
    )
    
    if prop_res["converged"]:
        st.success(f"Structural Iteration Converged successfully in {prop_res['iterations']} passes.")
        col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
        col_pr1.metric("Required Mass Ratio", f"{prop_res['mass_ratio']:.1f}:1")
        col_pr2.metric("Coupled Shield Radius", f"{prop_res['final_radius_m']:.2f} m")
        
        col_pr3.metric("Converged Shield Mass", f"{prop_res['shield_mass_kg'] / 1e3:,.0f} t")
        eiffel_towers_shield = (prop_res['shield_mass_kg'] / 1e3) / 10100
        col_pr3.caption(f"⚖️ Roughly {eiffel_towers_shield:,.1f} Eiffel Towers of deadweight.")
        
        col_pr4.metric("Required Fuel Mass", f"{prop_res['fuel_mass_kg'] / 1e3:,.0f} t")
        eiffel_towers_fuel = (prop_res['fuel_mass_kg'] / 1e3) / 10100
        col_pr4.caption(f"⛽ Roughly {eiffel_towers_fuel:,.1f} Eiffel Towers of fuel.")
    else:
        st.error(
            "🚨 **Design Impossible: The Mass-Area Death Spiral**\n\n"
            "The required fuel mass forced the tanks to expand radially. To protect those massive tanks, "
            "the nosecone shield became so heavy that the engine couldn't move it, requiring even more fuel. "
            "The solver timed out as the ship's mass spiraled toward infinity. \n\n"
            "**Fix this by lowering your cruise speed, switching to a Flyby mission, or upgrading your engine.**"
        )
        col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
        col_pr1.metric("Required Mass Ratio", f"{prop_res['mass_ratio']:.1f}:1")
        col_pr2.metric("Coupled Shield Radius", "DIVERGED")
        col_pr3.metric("Converged Shield Mass", "DIVERGED")
        col_pr4.metric("Required Fuel Mass", "DIVERGED")

    # --- KARDASHEV SCALE ENERGY COST MODULE ---
    if engine_choice == "Antimatter Beam Core" and prop_res['converged']:
        st.divider()
        st.subheader("Megastructure Synthesis: Antimatter Production Cost")
        
        am_mass_kg = prop_res['fuel_mass_kg'] / 2.0
        energy_ideal_joules = 2.0 * am_mass_kg * (SPEED_OF_LIGHT ** 2)
        earth_annual_joules = 6.0e20
        earth_years_ideal = energy_ideal_joules / earth_annual_joules
        
        col_am1, col_am2, col_am3 = st.columns(3)
        col_am1.metric("Pure Antimatter Req.", f"{am_mass_kg / 1e3:,.0f} t")
        col_am2.metric("Synthesis Energy (Ideal)", f"{energy_ideal_joules:.2e} J")
        col_am3.metric("Earth Energy Equivalent", f"{earth_years_ideal:,.0f} yrs")
        
        st.caption("💥 Equivalent to the energy required to boil away all of Earth's oceans hundreds of times over.")

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

        df_outcomes = pd.DataFrame([
            {"Outcome": "Flawless Transit", "Percentage": mc_res['flawless_pct']},
            {"Outcome": "Pitting Damage", "Percentage": mc_res['pitting_pct']},
            {"Outcome": "Catastrophic Breach", "Percentage": mc_res['breach_pct']}
        ])

        fig_pie = px.pie(
            df_outcomes, values="Percentage", names="Outcome",
            title="Mission Outcome Distribution across 10,000 Passes",
            color="Outcome", color_discrete_map={
                "Flawless Transit": "#00FFAA", "Pitting Damage": "#FF8800", "Catastrophic Breach": "#FF3344"
            }
        )
        fig_pie.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Click **Run 10,000 Flight Passes** to trigger the vectorized Monte Carlo engine.")