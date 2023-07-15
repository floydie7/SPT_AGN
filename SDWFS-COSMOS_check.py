"""
SDWFS-COSMOS_check.py
Author: Benjamin Floyd

Makes cutouts in the COSMOS field in order to perform a check on the SDWFS IRAGN surface densities.
"""

from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

rng = np.random.default_rng(1234)

# Read in the COSMOS catalog
cosmos = Table.read('Data_Repository/Catalogs/COSMOS/COSMOS2015_Laigle+_v1.1.fits.gz')
cosmos = cosmos['ALPHA_J2000', 'DELTA_J2000',
                'SPLASH_1_FLUX', 'SPLASH_1_FLUX_ERR', 'SPLASH_1_MAG', 'SPLASH_1_MAGERR',
                'SPLASH_2_FLUX', 'SPLASH_2_FLUX_ERR', 'SPLASH_2_MAG', 'SPLASH_2_MAGERR',
                'SPLASH_3_FLUX', 'SPLASH_3_FLUX_ERR', 'SPLASH_3_MAG', 'SPLASH_3_MAGERR',
                'SPLASH_4_FLUX', 'SPLASH_4_FLUX_ERR', 'SPLASH_4_MAG', 'SPLASH_4_MAGERR']

# Filter the catalog to only include sources with galaxies meeting our magnitude requirements
cosmos = cosmos[(10.00 < cosmos['SPLASH_1_MAG']) & (cosmos['SPLASH_1_MAG'] >= 18.3) &
                (10.45 < cosmos['SPLASH_2_MAG']) & (cosmos['SPLASH_2_MAG'] >= 17.48) &
                (cosmos['SPLASH_3_FLUX'] / cosmos['SPLASH_3_FLUX_ERR'] >= 5.) &
                (cosmos['SPLASH_4_FLUX'] / cosmos['SPLASH_4_FLUX_ERR'] >= 5.)]

# Get the footprint boundaries
min_ra, max_ra = cosmos['ALPHA_J2000'].min(), cosmos['ALPHA_J2000'].max()
min_dec, max_dec = cosmos['DELTA_J2000'].min(), cosmos['DELTA_J2000'].max()

# Define the footprint corners
footprint_corners = SkyCoord([(min_ra, min_dec), (max_ra, max_dec)], unit=u.deg)

# Trim to avoid the edges
trimmed_min_corner = footprint_corners[0].spherical_offsets_by(5 * u.arcmin, 5 * u.arcmin)
trimmed_max_corner = footprint_corners[1].spherical_offsets_by(-5 * u.arcmin, -5 * u.arcmin)

# Pick cutout center coordinates
ra_coords = rng.uniform(low=trimmed_min_corner.ra.value, high=trimmed_max_corner.ra.value, size=28)
dec_coords = rng.uniform(low=trimmed_min_corner.dec.value, high=trimmed_max_corner.dec.value, size=28)
center_coords = SkyCoord(ra_coords, dec_coords, unit=u.deg)

# Cycle through the cutout centers and select the objects within a 5'x5' square
cutout_catalogs = []
for center_coord in center_coords:
    # Calculate the boundaries
    max_bound = center_coord.spherical_offsets_by(2.5 * u.arcmin, 2.5 * u.arcmin)
    min_bound = center_coord.spherical_offsets_by(-2.5 * u.arcmin, -2.5 * u.arcmin)

    # Select objects within the cutout
    cutout_catalog = cosmos[(min_bound.ra.value < cosmos['ALPHA_J2000']) & (cosmos['ALPHA_J2000'] < max_bound.ra.value) &
                            (min_bound.dec.value < cosmos['DELTA_J2000']) & (cosmos['DELTA_J2000'] < max_bound.dec.value)]
    cutout_catalogs.append(cutout_catalog)

# Restack the catalog
cosmos_cutout_catalog = vstack(cutout_catalogs)
cosmos_cutout_catalog.write('Data_Repository/Project_Data/SPT-IRAGN/Misc/COSMOS15_catalog_cutouts.fits')
