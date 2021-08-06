"""
SDWFS_color-redshift_dist.py
Author: Benjamin Floyd

Saves a database for the SDWFS color-redshift distribution. We also compute color-color error trends.
"""

import numpy as np
from astropy.table import Table, join
import json

# Read in the SDWFS photometric catalog
SDWFS_main = Table.read(
    'Data_Repository/Catalogs/Bootes/SDWFS/ch2v33_sdwfs_2009mar3_apcorr_matched_ap4_Main_v0.4.cat.gz',
    names=['ID', 'IRAC_RA', 'IRAC_DEC', 'B_APFLUX4', 'R_APFLUX4', 'I_APFLUX4', 'B_APFLUXERR4',
           'R_APFLUXERR4', 'I_APFLUXERR4', 'B_APMAG4', 'R_APMAG4', 'I_APMAG4', 'B_APMAGERR4',
           'R_APMAGERR4', 'I_APMAGERR4', 'CH1_APFLUX4', 'CH2_APFLUX4', 'CH3_APFLUX4', 'CH4_APFLUX4',
           'CH1_APFLUXERR4', 'CH2_APFLUXERR4', 'CH3_APFLUXERR4', 'CH4_APFLUXERR4',
           'CH1_APFLUXERR4_BROWN', 'CH2_APFLUXERR4_BROWN', 'CH3_APFLUXERR4_BROWN',
           'CH4_APFLUXERR4_BROWN', 'CH1_APMAG4', 'CH2_APMAG4', 'CH3_APMAG4', 'CH4_APMAG4',
           'CH1_APMAGERR4', 'CH2_APMAGERR4', 'CH3_APMAGERR4', 'CH4_APMAGERR4',
           'CH1_APMAGERR4_BROWN', 'CH2_APMAGERR4_BROWN', 'CH3_APMAGERR4_BROWN',
           'CH4_APMAGERR4_BROWN', 'STARS_COLOR', 'STARS_MORPH', 'CLASS_STAR', 'MBZ_FLAG_4_4_4'],
    format='ascii')

# Read in the photometric redshift catalog
SDWFS_photz = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/mbz_v0.06_prior_bri12_18p8.cat.gz',
                         names=['ID', 'PHOT_Z', 'col3', 'col4', 'col5', 'col6', 'col7'],
                         format='ascii',
                         include_names=['ID', 'PHOT_Z'])

# Join the two catalogs together
SDWFS_cat = join(SDWFS_main, SDWFS_photz, keys='ID')

# Make the appropriate magnitude cuts
SDWFS_cat = SDWFS_cat[(10. < SDWFS_cat['CH1_APMAG4']) &
                      (10.45 < SDWFS_cat['CH2_APMAG4']) & (SDWFS_cat['CH2_APMAG4'] <= 17.46)]

# Select for Stern wedge AGN following the Stern+05 criteria
# Make the selections using the Stern wedge bounds
Stern_AGN = SDWFS_cat[(SDWFS_cat['CH3_APMAG4'] - SDWFS_cat['CH4_APMAG4'] > 0.6) &
                      (SDWFS_cat['CH1_APMAG4'] - SDWFS_cat['CH2_APMAG4']
                       > 0.2 * (SDWFS_cat['CH3_APMAG4'] - SDWFS_cat['CH4_APMAG4']) + 0.18) &
                      (SDWFS_cat['CH1_APMAG4'] - SDWFS_cat['CH2_APMAG4']
                       > 2.5 * (SDWFS_cat['CH3_APMAG4'] - SDWFS_cat['CH4_APMAG4']) - 3.5)]

# Make colors for both the full sample and for the AGN sample
SDWFS_color = SDWFS_cat['CH1_APMAG4'] - SDWFS_cat['CH2_APMAG4']
AGN_color = Stern_AGN['CH1_APMAG4'] - Stern_AGN['CH2_APMAG4']

# Set the color and redshift bins
color_bins = np.arange(0., 1.5, 0.05)
redshift_bins = np.arange(0., 1.7, 0.05)

# Create the histograms
SDWFS_hist, _, _ = np.histogram2d(SDWFS_cat['PHOT_Z'], SDWFS_color, bins=[redshift_bins, color_bins], density=True)
AGN_hist, _, _ = np.histogram2d(Stern_AGN['PHOT_Z'], AGN_color, bins=[redshift_bins, color_bins], density=True)

color_redshift_dists = {'color_bins': list(color_bins), 'redshift_bins': list(redshift_bins),
                        'SDWFS_hist': SDWFS_hist.tolist(), 'AGN_hist': AGN_hist.tolist()}

color_redshift_filename = 'Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_color_redshift_dist.json'
with open(color_redshift_filename, 'w') as f:
    json.dump(color_redshift_dists, f, ensure_ascii=False, indent=4)

# Compute the color errors for the two samples
SDWFS_color_err = np.sqrt((2.5 * SDWFS_cat['CH1_APFLUXERR4'] / (SDWFS_cat['CH1_APFLUX4'] * np.log(10)))**2 +
                          (2.5 * SDWFS_cat['CH2_APFLUXERR4'] / (SDWFS_cat['CH2_APFLUX4'] * np.log(10)))**2)
AGN_color_err = np.sqrt((2.5 * Stern_AGN['CH1_APFLUXERR4'] / (Stern_AGN['CH1_APFLUX4'] * np.log(10)))**2 +
                          (2.5 * Stern_AGN['CH2_APFLUXERR4'] / (Stern_AGN['CH2_APFLUX4'] * np.log(10)))**2)

# Store the color and color errors
SDWFS_color_table = Table([SDWFS_color, SDWFS_color_err], names=['COLOR', 'COLOR_ERR'])
AGN_color_table = Table([AGN_color, AGN_color_err], names=['COLOR', 'COLOR_ERR'])

# Write the tables
SDWFS_color_table.write('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_color-color_err.fits')
AGN_color_table.write('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_AGN_color-color_err.fits')
