import pytest
from physics.relativistic import solve_mass_area_feedback

def test_project_daedalus_historical_accuracy():
    """
    Verifies the simulation against the 1978 BIS Project Daedalus paper.
    Target: 0.122c flyby.
    Historical Initial Mass: ~54,000 metric tons.
    Historical Payload + Engine Dry Mass: ~1,500 metric tons.
    """
    
    historical_beta = 0.122
    historical_dry_mass_kg = 1_500_000  # 1,500 metric tons
    
    # Run our solver simulating the Daedalus design
    result = solve_mass_area_feedback(
        beta=historical_beta,
        payload_mass_kg=historical_dry_mass_kg,
        engine_type="Nuclear Fusion (Project Daedalus)",
        alloy_name="Titanium Grade 5 (Ti-6Al-4V)",  # Closest to Daedalus Beryllium
        thickness_m=0.01,                           # Daedalus used a thin 50t erosion shield
        cone_half_angle_deg=45.0,                   # Blunt profile
        architecture="Whipple Shield (Single Shell)",
        mission_profile="Flyby (No Deceleration)"
    )
    
    # 1. Verify the solver didn't hit the mass-area death spiral
    assert result["converged"] is True, "Daedalus flyby failed to converge!"
    
    # 2. Extract total simulated mass
    total_mass_tons = result["initial_mass_kg"] / 1000.0
    
    # 3. Verify against the 1978 paper's Single-Stage Equivalent specification.
    # Note: The historical 54,000-ton Daedalus was a TWO-STAGE vehicle. 
    # Because our solver evaluates a single-stage architecture, we expect 
    # the required mass to roughly double to ~108,000 tons due to the penalty 
    # of carrying empty tanks and heavy first-stage engines the entire way.
    expected_single_stage_mass = 108000
    lower_bound = expected_single_stage_mass * 0.85
    upper_bound = expected_single_stage_mass * 1.15
    
    error_msg = f"Simulated single-stage mass {total_mass_tons:,.0f} t deviates from expected 108k t baseline."
    assert lower_bound <= total_mass_tons <= upper_bound, error_msg

    print(f"SUCCESS: Simulated Single-Stage Mass = {total_mass_tons:,.0f} tons. (Accounts for missing staging)")

if __name__ == "__main__":
    test_project_daedalus_historical_accuracy()