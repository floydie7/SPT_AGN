"""
SDWFS_color-redshift_dist.py
Author: Benjamin Floyd

Saves a database for the SDWFS color-redshift distribution. We also compute color-color error trends.
"""

import pickle
import numpy as np
from astropy.table import Table, join
from scipy.stats import gaussian_kde

# Read in the SDWFS photometric catalog
# SDWFS_main = Table.read(
#     'Data_Repository/Catalogs/Bootes/SDWFS/ch2v33_sdwfs_2009mar3_apcorr_matched_ap4_Main_v0.4.cat.gz',
#     names=['ID', 'IRAC_RA', 'IRAC_DEC', 'B_APFLUX4', 'R_APFLUX4', 'I_APFLUX4', 'B_APFLUXERR4',
#            'R_APFLUXERR4', 'I_APFLUXERR4', 'B_APMAG4', 'R_APMAG4', 'I_APMAG4', 'B_APMAGERR4',
#            'R_APMAGERR4', 'I_APMAGERR4', 'CH1_APFLUX4', 'CH2_APFLUX4', 'CH3_APFLUX4', 'CH4_APFLUX4',
#            'CH1_APFLUXERR4', 'CH2_APFLUXERR4', 'CH3_APFLUXERR4', 'CH4_APFLUXERR4',
#            'CH1_APFLUXERR4_BROWN', 'CH2_APFLUXERR4_BROWN', 'CH3_APFLUXERR4_BROWN',
#            'CH4_APFLUXERR4_BROWN', 'CH1_APMAG4', 'CH2_APMAG4', 'CH3_APMAG4', 'CH4_APMAG4',
#            'CH1_APMAGERR4', 'CH2_APMAGERR4', 'CH3_APMAGERR4', 'CH4_APMAGERR4',
#            'CH1_APMAGERR4_BROWN', 'CH2_APMAGERR4_BROWN', 'CH3_APMAGERR4_BROWN',
#            'CH4_APMAGERR4_BROWN', 'STARS_COLOR', 'STARS_MORPH', 'CLASS_STAR', 'MBZ_FLAG_4_4_4'],
#     format='ascii')
#
# # Read in the photometric redshift catalog
# SDWFS_photz = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/mbz_v0.06_prior_bri12_18p8.cat.gz',
#                          names=['ID', 'PHOT_Z', 'col3', 'col4', 'col5', 'col6', 'col7'],
#                          format='ascii',
#                          include_names=['ID', 'PHOT_Z'])

# Join the two catalogs together
SDWFS_cat = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')

# Make the appropriate magnitude cuts
# SDWFS_cat = SDWFS_cat[(10. < SDWFS_cat['CH1_APMAG4']) &
#                       (10.45 < SDWFS_cat['CH2_APMAG4']) & (SDWFS_cat['CH2_APMAG4'] <= 17.46)]

# Select for Stern wedge AGN following the Stern+05 criteria
# Make the selections using the Stern wedge bounds
Stern_AGN = SDWFS_cat[(SDWFS_cat['I3_MAG_APER4'] - SDWFS_cat['I4_MAG_APER4'] > 0.6) &
                      (SDWFS_cat['I1_MAG_APER4'] - SDWFS_cat['I2_MAG_APER4']
                       > 0.2 * (SDWFS_cat['I3_MAG_APER4'] - SDWFS_cat['I4_MAG_APER4']) + 0.18) &
                      (SDWFS_cat['I1_MAG_APER4'] - SDWFS_cat['I2_MAG_APER4']
                       > 2.5 * (SDWFS_cat['I3_MAG_APER4'] - SDWFS_cat['I4_MAG_APER4']) - 3.5)]

# Make colors for both the full sample and for the AGN sample
SDWFS_color = SDWFS_cat['I1_MAG_APER4'] - SDWFS_cat['I2_MAG_APER4']
AGN_color = Stern_AGN['I1_MAG_APER4'] - Stern_AGN['I2_MAG_APER4']

# Create a Gaussian KDE of our color-redshift plane
SDWFS_values = np.vstack([SDWFS_cat['REDSHIFT'], SDWFS_color])
AGN_values = np.vstack([Stern_AGN['REDSHIFT'], AGN_color])

SDWFS_kde = gaussian_kde(SDWFS_values, weights=SDWFS_cat['COMPLETENESS_CORRECTION'] * SDWFS_cat['SELECTION_MEMBERSHIP'])
AGN_kde = gaussian_kde(AGN_values, weights=Stern_AGN['COMPLETENESS_CORRECTION'] * Stern_AGN['SELECTION_MEMBERSHIP'])

# Store the KDE objects in a pickle file
kde_dict = {'SDWFS_kde': SDWFS_kde, 'AGN_kde': AGN_kde}
sdwfs_background_dir = 'Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background'
with open(f'{sdwfs_background_dir}/SDWFS_color_redshift_kde_agn_weighted.pkl', 'wb') as f:
    pickle.dump(kde_dict, f, pickle.HIGHEST_PROTOCOL)

# Compute the color errors for the two samples
SDWFS_color_err = np.sqrt((2.5 * SDWFS_cat['I1_FLUXERR_APER4'] / (SDWFS_cat['I1_FLUX_APER4'] * np.log(10))) ** 2 +
                          (2.5 * SDWFS_cat['I2_FLUXERR_APER4'] / (SDWFS_cat['I2_FLUX_APER4'] * np.log(10))) ** 2)
AGN_color_err = np.sqrt((2.5 * Stern_AGN['I1_FLUXERR_APER4'] / (Stern_AGN['I1_FLUX_APER4'] * np.log(10))) ** 2 +
                        (2.5 * Stern_AGN['I2_FLUXERR_APER4'] / (Stern_AGN['I2_FLUX_APER4'] * np.log(10))) ** 2)

# Store the color and color errors
SDWFS_color_table = Table([SDWFS_color, SDWFS_color_err], names=['COLOR', 'COLOR_ERR'])
AGN_color_table = Table([AGN_color, AGN_color_err], names=['COLOR', 'COLOR_ERR'])

# Write the tables
SDWFS_color_table.write(f'{sdwfs_background_dir}/SDWFS_color-color_err_agn_weighted.fits', overwrite=True)
AGN_color_table.write(f'{sdwfs_background_dir}/SDWFS_AGN_color-color_err_agn_weighted.fits', overwrite=True)
