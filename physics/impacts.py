import numpy as np
from physics.relativistic import SPEED_OF_LIGHT, LIGHT_YEAR_METERS

DUST_BULK_DENSITY = 2500.0  # kg/m^3
BASE_DUST_NUMBER_DENSITY = 1e-11  # particles / m^3

ALLOY_PROPERTIES = {
    "Titanium Grade 5 (Ti-6Al-4V)": {
        "density": 4430.0,
        "vol_vap_energy_j_m3": 4.5e10,
        "fab_cost_per_kg": 120.0
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

def calculate_economic_costs(mass_kg: float, alloy_name: str) -> dict:
    alloy = ALLOY_PROPERTIES[alloy_name]
    cone_fab_cost_usd = mass_kg * alloy["fab_cost_per_kg"]
    
    v_e = 380.0 * 9.80665
    delta_v_escape = 3200.0
    propellant_mass_ratio = np.exp(delta_v_escape / v_e) - 1.0
    propellant_kg = mass_kg * propellant_mass_ratio
    escape_fuel_cost_usd = propellant_kg * 0.25

    return {
        "cone_fab_cost_usd": cone_fab_cost_usd,
        "escape_propellant_metric_tons": propellant_kg / 1000.0,
        "escape_fuel_cost_usd": escape_fuel_cost_usd,
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
    if thickness_m <= 0.0:
        return 0.0
        
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

def calculate_shield_erosion_rate(
    beta: float,
    alloy_name: str,
    cone_half_angle_deg: float,
    ism_density_multiplier: float
) -> float:
    """
    Calculates total surface thickness lost per meter traveled (m_eroded / m_traveled)
    by integrating energy deposited across the continuous sub-micron dust spectrum.
    """
    if beta <= 0.0:
        return 0.0
        
    alloy = ALLOY_PROPERTIES[alloy_name]
    alpha_rad = np.radians(max(1.0, min(90.0, cone_half_angle_deg)))
    sin_sq = np.sin(alpha_rad) ** 2
    
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    specific_ke = (gamma - 1.0) * (SPEED_OF_LIGHT ** 2)
    
    base_dust_mass_density = 2.0e-23 * ism_density_multiplier
    normal_energy_density_j_m3 = base_dust_mass_density * specific_ke * sin_sq
    
    vol_vaporized_per_m3_swept = normal_energy_density_j_m3 / alloy["vol_vap_energy_j_m3"]
    erosion_rate_m_per_m = vol_vaporized_per_m3_swept * np.sin(alpha_rad)
    
    return erosion_rate_m_per_m

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
        return {"lambda_fatal": 0.0, "survival_prob": 1.0, "r_lethal_microns": float("inf"), "shield_mass_tons": 0.0, "effective_thickness_cm": 0.0, "axial_length_m": 0.0}

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

def mission_survival_with_degradation(
    dist_ly: float,
    beta: float,
    ship_frontal_area_m2: float,
    alloy_name: str,
    thickness_m: float,
    ism_density_multiplier: float,
    cone_half_angle_deg: float = 90.0,
    architecture: str = "Uniform Shell (Hollow Whipple)"
) -> dict:
    """
    Calculates analytical survival probability factoring in continuous shield ablation.
    """
    radius_m = np.sqrt(ship_frontal_area_m2 / np.pi)
    geom = calculate_shield_geometry(radius_m, cone_half_angle_deg, thickness_m, architecture, alloy_name)
    initial_t = geom["effective_thickness_m"]
    
    total_dist_m = dist_ly * LIGHT_YEAR_METERS
    erosion_rate = calculate_shield_erosion_rate(beta, alloy_name, cone_half_angle_deg, ism_density_multiplier)
    
    total_erosion_m = erosion_rate * total_dist_m
    final_t = max(0.0, initial_t - total_erosion_m)
    
    if total_erosion_m >= initial_t:
        return {
            "survival_prob_degraded": 0.0,
            "lambda_fatal_degraded": float('inf'),
            "total_erosion_cm": total_erosion_m * 100.0,
            "initial_thickness_cm": initial_t * 100.0,
            "final_thickness_cm": final_t * 100.0,
            "armor_loss_pct": 100.0
        }
    
    steps = 100
    dx = total_dist_m / steps
    lambda_fatal_degraded = 0.0
    
    for i in range(steps):
        current_x = (i + 0.5) * dx
        current_t = max(1e-5, initial_t - (erosion_rate * current_x))
        r_lethal_t = lethal_grain_threshold_radius(beta, alloy_name, current_t, cone_half_angle_deg)
        n_lethal_t = cumulative_lethal_density(r_lethal_t, ism_density_multiplier)
        lambda_fatal_degraded += n_lethal_t * ship_frontal_area_m2 * dx

    survival_prob = float(np.exp(-lambda_fatal_degraded))

    return {
        "survival_prob_degraded": survival_prob,
        "lambda_fatal_degraded": lambda_fatal_degraded,
        "total_erosion_cm": total_erosion_m * 100.0,
        "initial_thickness_cm": initial_t * 100.0,
        "final_thickness_cm": final_t * 100.0,
        "armor_loss_pct": min(100.0, (total_erosion_m / initial_t) * 100.0)
    }

def run_monte_carlo_simulation(
    dist_ly: float,
    beta: float,
    ship_frontal_area_m2: float,
    alloy_name: str,
    thickness_m: float,
    ism_density_multiplier: float,
    cone_half_angle_deg: float = 90.0,
    architecture: str = "Uniform Shell (Hollow Whipple)",
    n_simulations: int = 10000,
    seed: int = 42
) -> dict:
    np.random.seed(seed)
    
    radius_m = np.sqrt(ship_frontal_area_m2 / np.pi)
    geom = calculate_shield_geometry(radius_m, cone_half_angle_deg, thickness_m, architecture, alloy_name)
    effective_t = geom["effective_thickness_m"]
    
    r_lethal = lethal_grain_threshold_radius(beta, alloy_name, effective_t, cone_half_angle_deg)
    r_min_hazard = max(1e-7, r_lethal * 0.01)
    r_max = max(1e-3, r_lethal * 2.0)
    
    n_hazard_m3 = cumulative_lethal_density(r_min_hazard, ism_density_multiplier)
    total_dist_m = dist_ly * LIGHT_YEAR_METERS
    swept_vol = ship_frontal_area_m2 * total_dist_m
    lambda_hazard = n_hazard_m3 * swept_vol
    
    erosion_rate = calculate_shield_erosion_rate(beta, alloy_name, cone_half_angle_deg, ism_density_multiplier)

    if lambda_hazard > 1e5:
        counts_float = np.random.normal(loc=lambda_hazard, scale=np.sqrt(lambda_hazard), size=n_simulations)
        impact_counts = np.maximum(0, np.round(counts_float)).astype(int)
    else:
        impact_counts = np.random.poisson(lam=lambda_hazard, size=n_simulations)
    
    max_depths = np.zeros(n_simulations)
    catastrophic_breaches = np.zeros(n_simulations, dtype=bool)
    
    total_impacts = int(np.sum(impact_counts))
    max_batch_size = 5_000_000
    if total_impacts > max_batch_size:
        scale_factor = total_impacts / max_batch_size
        sampled_impact_counts = np.round(impact_counts / scale_factor).astype(int)
    else:
        sampled_impact_counts = impact_counts
        
    sampled_total = int(np.sum(sampled_impact_counts))

    if sampled_total > 0:
        u = np.random.uniform(0.0, 1.0, size=sampled_total)
        inv_pow = -2.5
        term_min = r_min_hazard ** inv_pow
        term_max = r_max ** inv_pow
        sampled_radii = (term_min + u * (term_max - term_min)) ** (1.0 / inv_pow)
        
        volumes = (4.0 / 3.0) * np.pi * (sampled_radii ** 3)
        masses = DUST_BULK_DENSITY * volumes
        
        gamma = 1.0 / np.sqrt(1.0 - beta**2)
        ke_joules = (gamma - 1.0) * masses * (SPEED_OF_LIGHT ** 2)
        
        alpha_rad = np.radians(max(1.0, min(90.0, cone_half_angle_deg)))
        e_normal = ke_joules * (np.sin(alpha_rad) ** 2)
        
        alloy_vap_energy = ALLOY_PROPERTIES[alloy_name]["vol_vap_energy_j_m3"]
        crater_depths = (e_normal / (np.pi * alloy_vap_energy)) ** (1.0 / 3.0)
        
        impact_distances_m = np.random.uniform(0.0, total_dist_m, size=sampled_total)
        thickness_at_impact = effective_t - (erosion_rate * impact_distances_m)
        thickness_at_impact = np.maximum(thickness_at_impact, 0.0)
        
        breaches = crater_depths >= thickness_at_impact
        
        split_indices = np.cumsum(sampled_impact_counts)[:-1]
        sim_depth_groups = np.split(crater_depths, split_indices)
        sim_breach_groups = np.split(breaches, split_indices)
        
        for idx, (depths, brchs) in enumerate(zip(sim_depth_groups, sim_breach_groups)):
            if len(depths) > 0:
                max_depths[idx] = np.max(depths)
                if np.any(brchs):
                    catastrophic_breaches[idx] = True

    if erosion_rate * total_dist_m >= effective_t:
        catastrophic_breaches[:] = True

    flawless_mask = (impact_counts == 0)
    breach_mask = catastrophic_breaches
    pitting_mask = (~flawless_mask) & (~breach_mask)
    
    n_flawless = int(np.sum(flawless_mask))
    n_pitting = int(np.sum(pitting_mask))
    n_breached = int(np.sum(breach_mask))
    
    return {
        "n_simulations": n_simulations,
        "lambda_total": lambda_hazard,
        "impact_counts": impact_counts,
        "max_depths_cm": max_depths * 100.0,
        "effective_thickness_cm": effective_t * 100.0,
        "flawless_pct": (n_flawless / n_simulations) * 100.0,
        "pitting_pct": (n_pitting / n_simulations) * 100.0,
        "breach_pct": (n_breached / n_simulations) * 100.0,
        "mc_survival_rate": ((n_simulations - n_breached) / n_simulations) * 100.0
    }