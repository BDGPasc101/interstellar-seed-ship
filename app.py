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
    st.markdown("Automated loss-function minimization using `scipy.optimize.minimize_scalar`.")
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
        else:
            st.error("Optimization failed to converge within bounds.")

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

with tab_propulsion:
    st.subheader("Relativistic Rocket Equation & Mass-Area Convergence")
    st.markdown(
        "Solves the iterative feedback loop: more fuel requires wider tanks &rarr; "
        "wider tanks require a wider shield &rarr; heavier shield requires more fuel."
    )
    
    col_prop1, col_prop2 = st.columns(2)
    payload_mass_tons = col_prop1.slider("Base Payload Mass (Metric Tons)", 100, 20000, 5000, 100)
    engine_choice = col_prop2.selectbox("Main Propulsion System", list(ENGINE_TYPES.keys()), index=1)
    
    st.info(f"**{engine_choice}:** {ENGINE_TYPES[engine_choice]['description']} "
            f"(Exhaust Velocity: {ENGINE_TYPES[engine_choice]['ve_c']}c)")
    
    prop_res = solve_mass_area_feedback(
        beta=beta,
        payload_mass_kg=payload_mass_tons * 1000.0,
        engine_type=engine_choice,
        alloy_name=selected_alloy,
        thickness_m=armor_thickness_m,
        cone_half_angle_deg=cone_half_angle,
        architecture=selected_architecture
    )
    
    if prop_res["converged"]:
        st.success(f"Structural Iteration Converged successfully in {prop_res['iterations']} passes.")
        col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
        col_pr1.metric("Accel/Decel Mass Ratio", f"{prop_res['mass_ratio']:.1f}:1")
        col_pr2.metric("Coupled Shield Radius", f"{prop_res['final_radius_m']:.2f} m")
        col_pr3.metric("Converged Shield Mass", f"{prop_res['shield_mass_kg'] / 1e3:,.0f} t")
        col_pr4.metric("Required Fuel Mass", f"{prop_res['fuel_mass_kg'] / 1e3:,.0f} t")
    else:
        st.error(
            "?? **Design Impossible: The Mass-Area Death Spiral**\n\n"
            "The required fuel mass forced the tanks to expand radially. To protect those massive tanks, "
            "the nosecone shield became so heavy that the engine couldn't move it, requiring even more fuel. "
            "The solver timed out as the ship's mass spiraled toward infinity."
        )
        col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
        col_pr1.metric("Accel/Decel Mass Ratio", f"{prop_res['mass_ratio']:.1f}:1")
        col_pr2.metric("Coupled Shield Radius", "DIVERGED")
        col_pr3.metric("Converged Shield Mass", "DIVERGED")
        col_pr4.metric("Required Fuel Mass", "DIVERGED")
    
    if prop_res['fuel_mass_kg'] != float('inf'):
        col_pr3.metric("Converged Shield Mass", f"{prop_res['shield_mass_kg'] / 1e3:,.0f} t")
        col_pr4.metric("Required Fuel Mass", f"{prop_res['fuel_mass_kg'] / 1e3:,.0f} t")
    else:
        col_pr3.metric("Converged Shield Mass", "INFINITE")
        col_pr4.metric("Required Fuel Mass", "INFINITE")

    # --- KARDASHEV SCALE ENERGY COST MODULE ---
    if engine_choice == "Antimatter Beam Core" and prop_res['fuel_mass_kg'] != float('inf'):
        st.divider()
        st.subheader("Megastructure Synthesis: Antimatter Production Cost")
        
        # A beam core uses a 1:1 ratio of matter and antimatter for annihilation.
        am_mass_kg = prop_res['fuel_mass_kg'] / 2.0
        
        # Pair production requires 2 * m * c^2 to make the antimatter 
        # (creating an equal amount of normal matter as a byproduct).
        energy_ideal_joules = 2.0 * am_mass_kg * (SPEED_OF_LIGHT ** 2)
        
        earth_annual_joules = 6.0e20
        earth_years_ideal = energy_ideal_joules / earth_annual_joules
        
        col_am1, col_am2, col_am3 = st.columns(3)
        col_am1.metric("Pure Antimatter Req.", f"{am_mass_kg / 1e3:,.0f} t")
        col_am2.metric("Synthesis Energy (Ideal)", f"{energy_ideal_joules:.2e} J")
        col_am3.metric("Earth Energy Equivalent", f"{earth_years_ideal:,.0f} yrs")
        
        st.warning(
            f"**Kardashev Scale Requirements:** To produce `{am_mass_kg / 1e3:,.0f} tons` of antimatter "
            f"at 100% perfect efficiency requires `{earth_years_ideal:,.0f} years` of Earth's *entire* current "
            f"global energy output. At a realistic 1% synthesis efficiency, this jumps to "
            f"`{earth_years_ideal * 100:,.0f} years`. This fuel budget mandates a Circum-Solar Dyson Swarm."
        )

    st.divider()
    st.subheader(f"The Asymptote: Total Ship Mass vs Velocity ({engine_choice})")
    
    # Calculate the propulsion curve
    beta_range_prop = np.linspace(0.01, 0.85, 40)
    prop_data = []
    
    for b in beta_range_prop:
        res = solve_mass_area_feedback(
            beta=b, payload_mass_kg=payload_mass_tons * 1000.0, engine_type=engine_choice,
            alloy_name=selected_alloy, thickness_m=armor_thickness_m,
            cone_half_angle_deg=cone_half_angle, architecture=selected_architecture
        )
        if res["initial_mass_kg"] != float('inf'):
            prop_data.append({
                "beta": b,
                "Total Ship Mass (Tons)": res["initial_mass_kg"] / 1000.0,
                "Fuel Mass (Tons)": res["fuel_mass_kg"] / 1000.0
            })
            
    if prop_data:
        df_prop = pd.DataFrame(prop_data)
        fig_prop = px.line(
            df_prop, x="beta", y=["Total Ship Mass (Tons)", "Fuel Mass (Tons)"],
            log_y=True, title="Exponential Fuel Requirements (Log Scale)",
            labels={"value": "Mass (Metric Tons)", "beta": "Cruise Velocity (beta = v/c)"}
        )
        fig_prop.update_layout(template="plotly_dark", height=500, legend_title_text="Mass Component")
        st.plotly_chart(fig_prop, use_container_width=True)
    else:
        st.warning("No finite solutions exist for the selected engine in this velocity range.")

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
            title="Mission Outcome Distribution across 10,000 Passes",
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
    else:
        st.info("Click **Run 10,000 Flight Passes** to trigger the vectorized Monte Carlo engine.")