"""
SDWFS-COSMOS_check.py
Author: Benjamin Floyd

Makes cutouts in the COSMOS field in order to perform a check on the SDWFS IRAGN surface densities.
"""
import json
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from scipy.interpolate import interp1d

rng = np.random.default_rng(1234)

# Read in the COSMOS catalog
with warnings.catch_warnings():
    warnings.simplefilter('ignore', u.UnitsWarning)
    cosmos = Table.read('Data_Repository/Catalogs/COSMOS/COSMOS2020_CLASSIC_R1_v2.1_p3.fits.gz')

cosmos = cosmos['ID', 'ALPHA_J2000', 'DELTA_J2000',
                'IRAC_CH1_FLUX', 'IRAC_CH1_FLUXERR', 'IRAC_CH1_MAG', 'IRAC_CH1_MAGERR',
                'IRAC_CH2_FLUX', 'IRAC_CH2_FLUXERR', 'IRAC_CH2_MAG', 'IRAC_CH2_MAGERR']

# Convert the IRAC magnitudes from AB to Vega (because I don't want to have to convert the selections to AB).
cosmos['IRAC_CH1_MAG'] += -2.788
cosmos['IRAC_CH2_MAG'] += -3.255

# Filter the catalog to only include sources with galaxies meeting our magnitude requirements
cosmos = cosmos[(10.00 < cosmos['IRAC_CH1_MAG']) & (cosmos['IRAC_CH1_MAG'] <= 18.3) &
                (10.45 < cosmos['IRAC_CH2_MAG']) & (cosmos['IRAC_CH2_MAG'] <= 17.48)]

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

# For now, we will assume that the photometric completeness curve for COSMOS is the same as in SDWFS
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
    cutout_catalog = cosmos[(min_bound.ra.value < cosmos['ALPHA_J2000']) & (cosmos['ALPHA_J2000'] < max_bound.ra.value) &
                            (min_bound.dec.value < cosmos['DELTA_J2000']) & (cosmos['DELTA_J2000'] < max_bound.dec.value)]

    # Give the cutout an ID
    cutout_catalog['CUTOUT_ID'] = f'COSMOS_cutout_{i:03}'

    cutout_catalogs.append(cutout_catalog)

# Restack the catalog
cosmos_cutout_catalog = vstack(cutout_catalogs)
cosmos_cutout_catalog = cosmos_cutout_catalog.filled()

# Assuming that the photometric completeness of COSMOS is similar to the mean completeness in SDWFS
# compute the completeness corrections needed for the objects.
cosmos_cutout_catalog['COMPLETENESS_CORRECTION'] = 1 / sdwfs_mean_comp_curve(cosmos_cutout_catalog['IRAC_CH2_MAG'])

cosmos_cutout_catalog.write('Data_Repository/Project_Data/SPT-IRAGN/Misc/COSMOS20_catalog_cutouts.fits', overwrite=True)
