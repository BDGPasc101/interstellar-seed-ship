import pytest
import numpy as np
import numpy as np
from physics.relativistic import optimize_cruise_velocity
from physics.relativistic import (
    lorentz_factor,
    travel_times,
    kinetic_energy_joules,
    SPEED_OF_LIGHT
)

def test_lorentz_factor_at_rest():
    assert lorentz_factor(0.0) == 1.0

def test_lorentz_factor_standard_values():
    # At beta = 0.6, gamma should be exactly 1.25
    assert np.isclose(lorentz_factor(0.6), 1.25)
    # At beta = 0.8, gamma should be ~1.6667
    assert np.isclose(lorentz_factor(0.8), 5.0 / 3.0)

def test_lorentz_factor_invalid_inputs():
    with pytest.raises(ValueError):
        lorentz_factor(1.0)  # Cannot reach light speed
    with pytest.raises(ValueError):
        lorentz_factor(1.2)  # FTL prohibited
    with pytest.raises(ValueError):
        lorentz_factor(-0.1)

def test_time_dilation():
    dist_ly = 10.0
    beta = 0.8
    result = travel_times(dist_ly, beta)
    
    # Earth frame time = 10 / 0.8 = 12.5 years
    assert np.isclose(result["coord_time_years"], 12.5)
    # Ship frame time = 12.5 / (5/3) = 7.5 years
    assert np.isclose(result["proper_time_years"], 7.5)
    assert result["proper_time_years"] < result["coord_time_years"]

def test_kinetic_energy_limit():
    mass = 1.0  # 1 kg
    # Classical limit: E_k ? 0.5 * m * v^2 for small beta
    beta_small = 1e-4
    v = beta_small * SPEED_OF_LIGHT
    classical_ke = 0.5 * mass * (v ** 2)
    rel_ke = kinetic_energy_joules(mass, beta_small)
    assert np.isclose(rel_ke, classical_ke, rtol=1e-3)

def test_optimizer_finds_bounded_beta():
    opt_res = optimize_cruise_velocity(
        dist_ly=10.0,
        ship_frontal_area_m2=15.0,
        alloy_name="Titanium Grade 5 (Ti-6Al-4V)",
        thickness_m=0.10,
        ism_density_multiplier=1.0,
        cone_half_angle_deg=30.0,
        weight_time=1.0,
        weight_risk=5.0
    )
    
    assert opt_res["success"] is True
    assert 0.01 <= opt_res["optimal_beta"] <= 0.95
    assert 0.0 <= opt_res["optimal_survival_prob"] <= 1.0