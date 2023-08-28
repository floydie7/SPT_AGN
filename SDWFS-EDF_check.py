"""
SDWFS-EDF_check.py
Author: Benjamin Floyd

Just like `SDWFS-COSMOS_check.py` this creates rough cutouts in the Euclid Deep Fields to perform a check on the
SDWFS AGN surface densities.
"""

import json

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from scipy.interpolate import interp1d

rng = np.random.default_rng(1234)

# Read in the Euclid Deep Field-South catalog (We will start with EDF-S as it is the newest and most rectangular).
edfs = Table.read('Data_Repository/Catalogs/Cosmic_Dawn_Survey/Euclid_Deep_Fields/EDFS_IRAC_merged_v2.1.1.fits')

# Convert the magnitudes to use the Vega system rather than AB
edfs['MAG_AUTO_CH1'] += -2.788
edfs['MAG_AUTO_CH2'] += -3.255

# Filter the catalog to only include sources with galaxies meeting our magnitude requirements
edfs = edfs[(10.00 < edfs['MAG_AUTO_CH1']) & (edfs['MAG_AUTO_CH1'] <= 18.3) &
            (10.45 < edfs['MAG_AUTO_CH2']) & (edfs['MAG_AUTO_CH2'] <= 17.48)]

# For this rough test, we will draw cutouts from a central square region of the field
min_ra, max_ra = 60., 64.
min_dec, max_dec = -49., -47.

# Define the footprint corners
footprint_corners = SkyCoord([(min_ra, min_dec), (max_ra, max_dec)], unit=u.deg)

# Trim to avoid the edges
trimmed_min_corner = footprint_corners[0].spherical_offsets_by(5 * u.arcmin, 5 * u.arcmin)
trimmed_max_corner = footprint_corners[1].spherical_offsets_by(-5 * u.arcmin, -5 * u.arcmin)

# Pick cutout center coordinates
ra_coords = rng.uniform(low=trimmed_min_corner.ra.value, high=trimmed_max_corner.ra.value, size=28)
dec_coords = rng.uniform(low=trimmed_min_corner.dec.value, high=trimmed_max_corner.dec.value, size=28)
center_coords = SkyCoord(ra_coords, dec_coords, unit=u.deg)

# For now, we will assume that the photometric completeness curve for EDF-S is the same as in SDWFS
with open('Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SDWFS/Results/'
          'I2_results_gaussian_fwhm172_corr005_mag02_final.json') as f:
    sdwfs_completeness = json.load(f)
completeness_mag_bins = sdwfs_completeness.pop('magnitude_bins', None)[:-1]
sdwfs_mean_completeness = np.mean(list(list(curve) for curve in sdwfs_completeness.values()), axis=0)
sdwfs_mean_comp_curve = interp1d(completeness_mag_bins, sdwfs_mean_completeness)

# Cycle through the cutout centers and select the objects within a 5'x5' square
cutout_catalogs = []
for i, center_coord in enumerate(center_coords):
    # Calculate the boundaries
    max_bound = center_coord.spherical_offsets_by(2.5 * u.arcmin, 2.5 * u.arcmin)
    min_bound = center_coord.spherical_offsets_by(-2.5 * u.arcmin, -2.5 * u.arcmin)

    # Select objects within the cutout
    cutout_catalog = edfs[(min_bound.ra.value < edfs['ALPHA_J2000']) & (edfs['ALPHA_J2000'] < max_bound.ra.value) &
                          (min_bound.dec.value < edfs['DELTA_J2000']) & (edfs['DELTA_J2000'] < max_bound.dec.value)]

    # Give the cutout an ID
    cutout_catalog['CUTOUT_ID'] = f'EDFS_cutout_{i:03}'

    cutout_catalogs.append(cutout_catalog)

# Restack the catalog
edfs_cutout_catalog = vstack(cutout_catalogs)

# Assuming that the photometric completeness of EDF-S is similar to the mean completeness in SDWFS
# compute the completeness corrections needed for the objects.
edfs_cutout_catalog['COMPLETENESS_CORRECTION'] = 1 / sdwfs_mean_comp_curve(edfs_cutout_catalog['MAG_AUTO_CH2'])

edfs_cutout_catalog.write('Data_Repository/Project_Data/SPT-IRAGN/Misc/EDFS_catalog_cutouts.fits', overwrite=True)
