import pytest
import numpy as np
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