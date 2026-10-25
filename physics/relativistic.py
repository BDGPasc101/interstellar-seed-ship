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