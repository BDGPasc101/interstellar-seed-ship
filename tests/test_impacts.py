import pytest
import numpy as np
from physics.impacts import (
    grain_mass_from_radius,
    lethal_grain_threshold_radius,
    mission_survival_probability,
    ALLOY_PROPERTIES
)

def test_grain_mass_scaling():
    r = 1e-6  # 1 micron
    mass = grain_mass_from_radius(r)
    assert mass > 0
    # Mass scales with r^3: doubling radius should octuple mass
    mass_double = grain_mass_from_radius(2 * r)
    assert np.isclose(mass_double / mass, 8.0)

def test_armor_scaling():
    alloy = "Titanium Grade 5 (Ti-6Al-4V)"
    beta = 0.5
    r_thin = lethal_grain_threshold_radius(beta, alloy, thickness_m=0.01)
    r_thick = lethal_grain_threshold_radius(beta, alloy, thickness_m=0.05)
    
    # Thicker armor requires a larger particle to breach
    assert r_thick > r_thin

def test_speed_increases_vulnerability():
    alloy = "Whipple Carbon-Silicon Carbide (C/SiC)"
    # Higher beta should lower the minimum lethal particle size
    r_slow = lethal_grain_threshold_radius(0.1, alloy, 0.02)
    r_fast = lethal_grain_threshold_radius(0.8, alloy, 0.02)
    assert r_fast < r_slow

def test_survival_probability_monotonicity():
    dist_ly = 40.0
    area = 25.0
    alloy = "Tungsten Heavy Alloy"
    multiplier = 1.0

    res_slow = mission_survival_probability(dist_ly, 0.1, area, alloy, 0.03, multiplier)
    res_fast = mission_survival_probability(dist_ly, 0.7, area, alloy, 0.03, multiplier)

    assert 0.0 <= res_slow["survival_prob"] <= 1.0
    assert 0.0 <= res_fast["survival_prob"] <= 1.0
    # Survival probability drops as speed increases
    assert res_slow["survival_prob"] >= res_fast["survival_prob"]
    assert res_slow["lambda_fatal"] <= res_fast["lambda_fatal"]