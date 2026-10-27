import numpy as np
from scipy.optimize import minimize_scalar
import numpy as np

# Physical constants (SI units)
SPEED_OF_LIGHT = 299_792_458.0          # m/s
LIGHT_YEAR_METERS = 9.4607304725808e15   # m
YEAR_SECONDS = 31_557_600.0             # seconds per Julian year
JOULES_PER_MEGATON_TNT = 4.184e15       # J per Mt TNT

def lorentz_factor(beta: float) -> float:
    """
    Calculate the relativistic Lorentz factor (gamma).
    beta = v / c, must satisfy 0 <= beta < 1.0
    """
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"Velocity fraction beta must satisfy 0 <= beta < 1, got {beta}")
    if beta == 0.0:
        return 1.0
    return 1.0 / np.sqrt(1.0 - beta**2)

def travel_times(dist_ly: float, beta: float) -> dict:
    """
    Compute mission duration from Earth's frame and the ship's frame.
    
    Returns:
        dict with:
          - coord_time_years: Earth frame duration (years)
          - proper_time_years: Ship onboard duration (years)
          - gamma: Lorentz factor
    """
    if beta <= 0.0:
        return {
            "coord_time_years": float("inf"),
            "proper_time_years": float("inf"),
            "gamma": 1.0
        }
    
    gamma = lorentz_factor(beta)
    coord_time_years = dist_ly / beta
    proper_time_years = coord_time_years / gamma

    return {
        "coord_time_years": coord_time_years,
        "proper_time_years": proper_time_years,
        "gamma": gamma
    }

def kinetic_energy_joules(mass_kg: float, beta: float) -> float:
    """
    Calculate the relativistic kinetic energy: E_k = (gamma - 1) * m * c^2
    """
    gamma = lorentz_factor(beta)
    return (gamma - 1.0) * mass_kg * (SPEED_OF_LIGHT ** 2)

def energy_to_tnt_megatons(energy_joules: float) -> float:
    """
    Convert energy from Joules to Megatons of TNT equivalent.
    """
    return energy_joules / JOULES_PER_MEGATON_TNT


def optimize_cruise_velocity(
    dist_ly: float,
    ship_frontal_area_m2: float,
    alloy_name: str,
    thickness_m: float,
    ism_density_multiplier: float,
    cone_half_angle_deg: float = 90.0,
    architecture: str = "Uniform Shell (Hollow Whipple)",
    weight_time: float = 1.0,
    weight_risk: float = 1.0,
    beta_bounds: tuple = (0.01, 0.95)
) -> dict:
    """
    Uses scipy.optimize.minimize_scalar to calculate the optimal cruise velocity (beta)
    that minimizes a composite cost function balancing transit duration and breach risk.
    """
    from physics.impacts import mission_survival_probability

    # Reference time at beta = 0.1c for normalization
    t_ref = travel_times(dist_ly, 0.10)["proper_time_years"]

    def loss_function(beta_candidate: float) -> float:
        if beta_candidate <= 0.0 or beta_candidate >= 1.0:
            return 1e9

        t_res = travel_times(dist_ly, beta_candidate)
        s_res = mission_survival_probability(
            dist_ly=dist_ly,
            beta=beta_candidate,
            ship_frontal_area_m2=ship_frontal_area_m2,
            alloy_name=alloy_name,
            thickness_m=thickness_m,
            ism_density_multiplier=ism_density_multiplier,
            cone_half_angle_deg=cone_half_angle_deg,
            architecture=architecture
        )

        time_penalty = (t_res["proper_time_years"] / t_ref)
        risk_penalty = 1.0 - s_res["survival_prob"]

        return (weight_time * time_penalty) + (weight_risk * risk_penalty)

    res = minimize_scalar(
        loss_function,
        bounds=beta_bounds,
        method="bounded"
    )

    optimal_beta = float(res.x)
    optimal_travel = travel_times(dist_ly, optimal_beta)
    optimal_survival = mission_survival_probability(
        dist_ly=dist_ly,
        beta=optimal_beta,
        ship_frontal_area_m2=ship_frontal_area_m2,
        alloy_name=alloy_name,
        thickness_m=thickness_m,
        ism_density_multiplier=ism_density_multiplier,
        cone_half_angle_deg=cone_half_angle_deg,
        architecture=architecture
    )

    return {
        "optimal_beta": optimal_beta,
        "optimal_ship_years": optimal_travel["proper_time_years"],
        "optimal_earth_years": optimal_travel["coord_time_years"],
        "optimal_survival_prob": optimal_survival["survival_prob"],
        "cost_value": float(res.fun),
        "success": res.success
    }

# Append to the bottom of physics/relativistic.py

ENGINE_TYPES = {
    "Nuclear Fusion (Project Daedalus)": {
        "ve_c": 0.03,              # 3% of light speed exhaust
        "fuel_density": 71.0,      # Liquid Hydrogen / He-3 kg/m^3
        "tank_length_m": 300.0,    # Max length of fuel tanks before expanding radially
        "description": "Deuterium/Helium-3 pellets ignited by inertial confinement."
    },
    "Antimatter Beam Core": {
        "ve_c": 0.33,              # 33% of light speed exhaust (charged pions)
        "fuel_density": 1000.0,    # Water/Tungsten containment medium kg/m^3
        "tank_length_m": 200.0,
        "description": "Proton-antiproton annihilation channeling charged pions via magnetic nozzles."
    },
    "Perfect Photon Drive (Theoretical)": {
        "ve_c": 1.00,              # 100% of light speed exhaust
        "fuel_density": 2000.0,    # Exotic dense matter / compact containment
        "tank_length_m": 100.0,
        "description": "Total matter-energy conversion emitting pure directed radiation."
    }
}

def solve_mass_area_feedback(
    beta: float,
    payload_mass_kg: float,
    engine_type: str,
    alloy_name: str,
    thickness_m: float,
    cone_half_angle_deg: float,
    architecture: str
) -> dict:
    """
    Iterative solver that resolves the coupled 'mass-area' feedback loop.
    Iterates until the fuel tank radius and the required shield radius converge.
    """
    from physics.impacts import calculate_shield_geometry
    
    engine = ENGINE_TYPES[engine_type]
    ve_c = engine["ve_c"]
    rho = engine["fuel_density"]
    l_tank = engine["tank_length_m"]
    
    if beta <= 0.0 or beta >= 1.0:
        return {"converged": False, "initial_mass_kg": float('inf')}
        
    # Relativistic Rocket Equation for accelerating AND decelerating
    # M_initial / M_final = ((1 + beta) / (1 - beta)) ^ (1 / ve_c)
    try:
        mass_ratio = ((1.0 + beta) / (1.0 - beta)) ** (1.0 / ve_c)
    except OverflowError:
        mass_ratio = float('inf')
        
    base_radius = 4.5  # Minimum radius to occlude the payload section
    current_r = base_radius
    converged = False
    
    m_shield = 0.0
    m_initial = float('inf')
    m_fuel = float('inf')
    m_dry = payload_mass_kg
    
    # Iterate to find structural convergence (max 50 loops)
    for i in range(50):
        geom = calculate_shield_geometry(current_r, cone_half_angle_deg, thickness_m, architecture, alloy_name)
        m_shield = geom["mass_kg"]
        m_dry = payload_mass_kg + m_shield
        
        if mass_ratio > 1e10:
            m_initial = float('inf')
            break
            
        m_initial = m_dry * mass_ratio
        if m_initial > 1e15:  # Diverged to > 1 trillion tons
            m_initial = float('inf')
            break
            
        m_fuel = m_initial - m_dry
        v_fuel = m_fuel / rho
        
        # Calculate tank radius needed to hold the fuel
        r_tank = np.sqrt(v_fuel / (np.pi * l_tank))
        r_new = max(base_radius, r_tank)
        
        # Check for convergence
        if abs(r_new - current_r) < 0.01:
            converged = True
            current_r = r_new
            break
            
        current_r = r_new
        
    return {
        "converged": converged,
        "mass_ratio": mass_ratio,
        "final_radius_m": current_r,
        "frontal_area_m2": np.pi * (current_r ** 2),
        "shield_mass_kg": m_shield,
        "dry_mass_kg": m_dry,
        "initial_mass_kg": m_initial,
        "fuel_mass_kg": m_fuel,
        "iterations": i + 1
    }