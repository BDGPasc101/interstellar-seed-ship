import pytest
import numpy as np
from physics.impacts import calculate_shield_geometry
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

def test_cone_deflection_benefits():
    alloy = "Titanium Grade 5 (Ti-6Al-4V)"
    beta = 0.5
    thickness = 0.02
    
    # A sharper cone (smaller half-angle) deflects more energy,
    # meaning only larger grains can penetrate it.
    r_blunt = lethal_grain_threshold_radius(beta, alloy, thickness, cone_half_angle_deg=90.0)
    r_sharp = lethal_grain_threshold_radius(beta, alloy, thickness, cone_half_angle_deg=15.0)
    
    assert r_sharp > r_blunt
    
    # 1. Compare expected lethal impacts (lambda_fatal), which avoids float underflow:
    res_blunt = mission_survival_probability(40.0, beta, 20.0, alloy, thickness, 1.0, cone_half_angle_deg=90.0)
    res_sharp = mission_survival_probability(40.0, beta, 20.0, alloy, thickness, 1.0, cone_half_angle_deg=15.0)
    assert res_sharp["lambda_fatal"] < res_blunt["lambda_fatal"]

    # 2. In a survivable regime (e.g. beta = 0.1, thicker shield), survival probability strictly increases:
    surv_blunt = mission_survival_probability(4.0, 0.1, 5.0, alloy, 0.10, 1.0, cone_half_angle_deg=90.0)
    surv_sharp = mission_survival_probability(4.0, 0.1, 5.0, alloy, 0.10, 1.0, cone_half_angle_deg=15.0)
    assert surv_sharp["survival_prob"] > surv_blunt["survival_prob"]


def test_shield_architectures_mass_and_thickness():
    r = 4.0
    alpha = 15.0
    t = 0.05
    alloy = "Titanium Grade 5 (Ti-6Al-4V)"

    geo_uniform = calculate_shield_geometry(r, alpha, t, "Uniform Shell (Hollow Whipple)", alloy)
    geo_graded = calculate_shield_geometry(r, alpha, t, "Graded Shell (Apex & Spall Reinforced)", alloy)
    geo_solid = calculate_shield_geometry(r, alpha, t, "Solid Nosecone (Machined Slug)", alloy)

    # Graded shell has 4/3 the mass and average thickness of uniform shell
    assert np.isclose(geo_graded["mass_kg"], (4.0 / 3.0) * geo_uniform["mass_kg"], rtol=1e-4)
    assert np.isclose(geo_graded["effective_thickness_m"], (4.0 / 3.0) * geo_uniform["effective_thickness_m"])

    # Solid nosecone mass is dramatically larger than a thin hollow shell
    assert geo_solid["mass_kg"] > geo_uniform["mass_kg"]

    # Solid nosecone effective thickness is L / 3
    expected_solid_t = (r / np.tan(np.radians(alpha))) / 3.0
    assert np.isclose(geo_solid["effective_thickness_m"], expected_solid_t)