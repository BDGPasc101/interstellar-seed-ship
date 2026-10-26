import numpy as np
from physics.relativistic import kinetic_energy_joules, SPEED_OF_LIGHT, LIGHT_YEAR_METERS

DUST_BULK_DENSITY = 2500.0  # kg/m^3
BASE_DUST_NUMBER_DENSITY = 1e-11  # particles / m^3

# Add to ALLOY_PROPERTIES in physics/impacts.py:
ALLOY_PROPERTIES = {
    "Titanium Grade 5 (Ti-6Al-4V)": {
        "density": 4430.0,
        "vol_vap_energy_j_m3": 4.5e10,
        "fab_cost_per_kg": 120.0  # Finished aerospace grade
    },
    "Tungsten Heavy Alloy": {
        "density": 19300.0,
        "vol_vap_energy_j_m3": 9.2e10,
        "fab_cost_per_kg": 200.0
    },
    "Whipple Carbon-Silicon Carbide (C/SiC)": {
        "density": 2700.0,
        "vol_vap_energy_j_m3": 7.0e10,
        "fab_cost_per_kg": 350.0
    }
}

# In physics/impacts.py

def calculate_economic_costs(mass_kg: float, alloy_name: str) -> dict:
    """
    Computes material fabrication cost and cryogenic propellant mass/fuel 
    cost required to push the shield mass clear of Earth escape velocity.
    """
    alloy = ALLOY_PROPERTIES[alloy_name]
    
    # 1. Fabrication / Raw Material Cost
    cone_fab_cost_usd = mass_kg * alloy["fab_cost_per_kg"]
    
    # 2. Tsiolkovsky Earth Escape Stage Propellant (delta_v = 3,200 m/s from LEO)
    # Isp = 380s (Methalox vacuum), ve = 3,727 m/s
    v_e = 380.0 * 9.80665
    delta_v_escape = 3200.0
    propellant_mass_ratio = np.exp(delta_v_escape / v_e) - 1.0
    propellant_kg = mass_kg * propellant_mass_ratio
    
    # Commercial LOX/LCH4 propellant cost ~ $0.25 / kg
    escape_fuel_cost_usd = propellant_kg * 0.25

    return {
        "cone_fab_cost_usd": cone_fab_cost_usd,
        "escape_propellant_metric_tons": propellant_kg / 1000.0,
        "escape_fuel_cost_usd": escape_fuel_cost_usd,
    }

SHIELD_ARCHITECTURES = [
    "Uniform Shell (Hollow Whipple)",
    "Graded Shell (Apex & Spall Reinforced)",
    "Solid Nosecone (Machined Slug)"
]

def calculate_shield_geometry(
    radius_m: float,
    cone_half_angle_deg: float,
    thickness_m: float,
    architecture: str,
    alloy_name: str
) -> dict:
    """
    Computes axial length, shield volume, mass, and area-weighted effective thickness.
    """
    alpha_rad = np.radians(max(1.0, min(90.0, cone_half_angle_deg)))
    sin_a = np.sin(alpha_rad)
    tan_a = np.tan(alpha_rad)
    
    axial_length_m = radius_m / tan_a
    slant_height_m = radius_m / sin_a
    lateral_area_m2 = np.pi * radius_m * slant_height_m
    density = ALLOY_PROPERTIES[alloy_name]["density"]

    if architecture == "Solid Nosecone (Machined Slug)":
        volume_m3 = (1.0 / 3.0) * np.pi * (radius_m ** 2) * axial_length_m
        effective_thickness_m = (1.0 / 3.0) * axial_length_m
        apex_thickness_m = axial_length_m
        base_thickness_m = 0.0
    elif architecture == "Graded Shell (Apex & Spall Reinforced)":
        # Thickness tapers from 2*T at apex to 1*T at base: mean = 4/3 * T
        mean_normal_t = (4.0 / 3.0) * thickness_m
        volume_m3 = lateral_area_m2 * mean_normal_t
        effective_thickness_m = mean_normal_t
        apex_thickness_m = 2.0 * thickness_m
        base_thickness_m = thickness_m
    else:  # Uniform Shell
        volume_m3 = lateral_area_m2 * thickness_m
        effective_thickness_m = thickness_m
        apex_thickness_m = thickness_m
        base_thickness_m = thickness_m

    mass_kg = density * volume_m3

    return {
        "axial_length_m": axial_length_m,
        "slant_height_m": slant_height_m,
        "volume_m3": volume_m3,
        "mass_kg": mass_kg,
        "mass_metric_tons": mass_kg / 1000.0,
        "effective_thickness_m": effective_thickness_m,
        "apex_thickness_m": apex_thickness_m,
        "base_thickness_m": base_thickness_m
    }

def grain_mass_from_radius(radius_meters: float) -> float:
    volume = (4.0 / 3.0) * np.pi * (radius_meters ** 3)
    return DUST_BULK_DENSITY * volume

def lethal_grain_threshold_radius(
    beta: float,
    alloy_name: str,
    thickness_m: float,
    cone_half_angle_deg: float = 90.0
) -> float:
    alloy = ALLOY_PROPERTIES[alloy_name]
    crater_volume_per_thickness = np.pi * (thickness_m ** 2) * thickness_m
    energy_to_breach = alloy["vol_vap_energy_j_m3"] * crater_volume_per_thickness

    alpha_rad = np.radians(max(1.0, min(90.0, cone_half_angle_deg)))
    sin_sq = np.sin(alpha_rad) ** 2
    required_ke = energy_to_breach / sin_sq

    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    min_mass_kg = required_ke / ((gamma - 1.0) * (SPEED_OF_LIGHT ** 2))

    r_meters = ((3.0 * min_mass_kg) / (4.0 * np.pi * DUST_BULK_DENSITY)) ** (1.0 / 3.0)
    return r_meters

def cumulative_lethal_density(r_min: float, ism_multiplier: float) -> float:
    r_reference = 1e-6
    r_eff = max(r_min, 1e-7)
    return BASE_DUST_NUMBER_DENSITY * ((r_reference / r_eff) ** 2.5) * ism_multiplier

def mission_survival_probability(
    dist_ly: float,
    beta: float,
    ship_frontal_area_m2: float,
    alloy_name: str,
    thickness_m: float,
    ism_density_multiplier: float,
    cone_half_angle_deg: float = 90.0,
    architecture: str = "Uniform Shell (Hollow Whipple)"
) -> dict:
    if beta <= 0.0:
        return {"lambda_fatal": 0.0, "survival_prob": 1.0, "r_lethal_microns": float("inf")}

    radius_m = np.sqrt(ship_frontal_area_m2 / np.pi)
    geom = calculate_shield_geometry(radius_m, cone_half_angle_deg, thickness_m, architecture, alloy_name)

    r_lethal = lethal_grain_threshold_radius(beta, alloy_name, geom["effective_thickness_m"], cone_half_angle_deg)
    n_lethal_m3 = cumulative_lethal_density(r_lethal, ism_density_multiplier)
    
    total_dist_meters = dist_ly * LIGHT_YEAR_METERS
    swept_volume = ship_frontal_area_m2 * total_dist_meters
    expected_fatal_impacts = n_lethal_m3 * swept_volume

    prob_survive = float(np.exp(-expected_fatal_impacts))

    return {
        "lambda_fatal": expected_fatal_impacts,
        "survival_prob": prob_survive,
        "r_lethal_microns": r_lethal * 1e6,
        "shield_mass_tons": geom["mass_metric_tons"],
        "effective_thickness_cm": geom["effective_thickness_m"] * 100.0,
        "axial_length_m": geom["axial_length_m"]
    }