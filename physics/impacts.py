import numpy as np
from physics.relativistic import kinetic_energy_joules, SPEED_OF_LIGHT, LIGHT_YEAR_METERS

# Interstellar dust parameters (standard astronomical silicates)
DUST_BULK_DENSITY = 2500.0  # kg/m^3
# Baseline ISM grain density (>0.1 micron) in the solar neighborhood ~ 1e-12 grains/m^3
BASE_DUST_NUMBER_DENSITY = 1e-11  # particles / m^3

# Common aerospace/shielding alloys
ALLOY_PROPERTIES = {
    "Titanium Grade 5 (Ti-6Al-4V)": {
        "density": 4430.0,            # kg/m^3
        "vol_vap_energy_j_m3": 4.5e10 # J/m^3 (approx. energy to melt/vaporize crater core)
    },
    "Tungsten Heavy Alloy": {
        "density": 19300.0,
        "vol_vap_energy_j_m3": 9.2e10
    },
    "Whipple Carbon-Silicon Carbide (C/SiC)": {
        "density": 2700.0,
        "vol_vap_energy_j_m3": 7.0e10
    }
}

def grain_mass_from_radius(radius_meters: float) -> float:
    """Spherical grain mass from its radius."""
    volume = (4.0 / 3.0) * np.pi * (radius_meters ** 3)
    return DUST_BULK_DENSITY * volume

def lethal_grain_threshold_radius(beta: float, alloy_name: str, thickness_m: float) -> float:
    """
    Determine the minimum grain radius that can puncture the shield thickness T.
    Using energy balance: E_k(grain) >= vol_vap_energy * (crater volume formed)
    """
    alloy = ALLOY_PROPERTIES[alloy_name]
    # At relativistic speeds, crater diameter scales roughly ~ 3x to 5x grain diameter
    crater_volume_per_thickness = np.pi * (thickness_m ** 2) * thickness_m
    energy_to_breach = alloy["vol_vap_energy_j_m3"] * crater_volume_per_thickness

    # Solve for required grain mass: E_k = (gamma - 1) * m * c^2
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    min_mass_kg = energy_to_breach / ((gamma - 1.0) * (SPEED_OF_LIGHT ** 2))

    # Solve for radius from mass: m = 4/3 * pi * r^3 * rho
    r_meters = ((3.0 * min_mass_kg) / (4.0 * np.pi * DUST_BULK_DENSITY)) ** (1.0 / 3.0)
    return r_meters

def cumulative_lethal_density(r_min: float, ism_multiplier: float) -> float:
    """
    Calculates number density of particles with radius >= r_min using the MRN power law:
    N(> r) = N_0 * (r / r_0)^(-2.5)
    """
    r_reference = 1e-6  # 1 micrometer reference
    # Prevent divide-by-zero or excessively large values for ultra-small grains
    r_eff = max(r_min, 1e-7)
    
    n_grains_m3 = BASE_DUST_NUMBER_DENSITY * ((r_reference / r_eff) ** 2.5)
    return n_grains_m3 * ism_multiplier

def mission_survival_probability(
    dist_ly: float,
    beta: float,
    ship_frontal_area_m2: float,
    alloy_name: str,
    thickness_m: float,
    ism_density_multiplier: float
) -> dict:
    """
    Computes expected lethal impacts (lambda) and survival probability P(0) via Poisson.
    """
    if beta <= 0.0:
        return {"lambda_fatal": 0.0, "survival_prob": 1.0, "r_lethal_microns": float("inf")}

    r_lethal = lethal_grain_threshold_radius(beta, alloy_name, thickness_m)
    n_lethal_m3 = cumulative_lethal_density(r_lethal, ism_density_multiplier)
    
    total_dist_meters = dist_ly * LIGHT_YEAR_METERS
    swept_volume = ship_frontal_area_m2 * total_dist_meters
    expected_fatal_impacts = n_lethal_m3 * swept_volume

    # Poisson survival probability: P(k=0) = e^(-lambda)
    prob_survive = float(np.exp(-expected_fatal_impacts))

    return {
        "lambda_fatal": expected_fatal_impacts,
        "survival_prob": prob_survive,
        "r_lethal_microns": r_lethal * 1e6
    }