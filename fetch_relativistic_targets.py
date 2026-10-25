import os
import io
import requests
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

def fetch_and_process_targets():
    print("Connecting to NASA Exoplanet Archive (TAP API)...")

    # Fetch confirmed exoplanets with celestial coordinates and stellar parameters
    query = """
    select pl_name, hostname, sy_dist, ra, dec,
           pl_rade, pl_masse, pl_orbper, pl_orbsmax,
           st_teff, st_rad, st_age, st_met
    from pscomppars
    where pl_masse is not null 
      and pl_rade is not null 
      and sy_dist is not null
      and ra is not null
      and dec is not null
    """
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv"
    
    response = requests.get(url)
    response.raise_for_status()
    print("Data downloaded successfully. Processing candidates...")

    df = pd.read_csv(io.StringIO(response.text))

    # 1. Calculate Earth Similarity Index (ESI)
    esi_radius = 1 - np.abs((df['pl_rade'] - 1) / (df['pl_rade'] + 1))
    esi_mass = 1 - np.abs((df['pl_masse'] - 1) / (df['pl_masse'] + 1))
    df['esi'] = esi_radius * esi_mass

    # 2. Stefan-Boltzmann Luminosity & Habitable Zone (HZ) boundaries
    df['st_lum_est'] = (df['st_rad'].fillna(1.0) ** 2) * ((df['st_teff'].fillna(5778) / 5778) ** 4)
    df['hz_inner'] = (df['st_lum_est'] / 1.1) ** 0.5
    df['hz_outer'] = (df['st_lum_est'] / 0.53) ** 0.5
    df['in_hz'] = (df['pl_orbsmax'] >= df['hz_inner']) & (df['pl_orbsmax'] <= df['hz_outer'])

    # 3. Exclude stars confirmed younger than 1 Gyr (keep mature or unmeasured)
    mature_mask = (df['st_age'] >= 1.0) | (df['st_age'].isna())

    # 4. Filter for terrestrial worlds
    candidates = df[
        (df['pl_masse'].between(0.3, 5.0)) &
        (df['pl_rade'].between(0.5, 1.8)) &
        mature_mask
    ].copy()

    # 5. Prioritize by: In Habitable Zone -> ESI descending -> Distance ascending
    candidates = candidates.sort_values(
        by=['in_hz', 'esi', 'sy_dist'], 
        ascending=[False, False, True]
    ).head(10).reset_index(drop=True)

    # 6. Coordinate Conversion: ICRS (RA, Dec) -> Galactic (l, b)
    sky_coords = SkyCoord(
        ra=candidates['ra'].values * u.degree, 
        dec=candidates['dec'].values * u.degree, 
        frame='icrs'
    )
    galactic = sky_coords.galactic
    candidates['gal_l_deg'] = galactic.l.degree
    candidates['gal_b_deg'] = galactic.b.degree

    # Convert distance from parsecs to light-years
    candidates['dist_ly'] = candidates['sy_dist'] * 3.26156

    # 7. Directional Interstellar Medium (ISM) Dust Density Multiplier
    # Milky Way dust concentrates in the galactic midplane (b = 0).
    # We model scale height decay: high latitudes (|b| -> 90) pass through cleaner space.
    candidates['ism_density_multiplier'] = 0.3 + 2.7 * np.exp(
        -np.abs(np.radians(candidates['gal_b_deg'])) / np.radians(15.0)
    )

    # Clean and export columns
    output_cols = [
        'pl_name', 'hostname', 'dist_ly', 'sy_dist',
        'gal_b_deg', 'gal_l_deg', 'ism_density_multiplier',
        'esi', 'in_hz', 'pl_masse', 'pl_rade', 'st_teff', 'st_age'
    ]
    candidates = candidates[output_cols]

    os.makedirs("data", exist_ok=True)
    output_path = "data/seed_ship_targets.csv"
    candidates.to_csv(output_path, index=False)

    print(f"\nSaved Top 10 target systems to {output_path}:")
    print(candidates[['pl_name', 'dist_ly', 'gal_b_deg', 'ism_density_multiplier', 'esi', 'in_hz']])

if __name__ == "__main__":
    fetch_and_process_targets()