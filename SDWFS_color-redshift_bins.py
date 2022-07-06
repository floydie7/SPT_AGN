"""
SDWFS_color-redshift_bins.py
Author: Benjamin Floyd

Creates histograms from the color-redshift plane using SDWFS data.
"""

import numpy as np
from astropy.table import Table, join

# Read in the SDWFS photometric catalog
sdwfs_main = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/ch2v33_sdwfs_2009mar3_apcorr_matched_ap4_Main_v0.4.cat.gz',
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
sdwfs_photz = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/mbz_v0.06_prior_bri12_18p8.cat.gz',
                         names=['ID', 'PHOT_Z', 'col3', 'col4', 'col5', 'col6', 'col7'],
                         format='ascii',
                         include_names=['ID', 'PHOT_Z'])

# Join the two catalogs together
sdwfs_cat = join(sdwfs_main, sdwfs_photz, keys='ID')

# For convenience, add columns for the two colors
sdwfs_cat['CH1_CH2_COLOR'] = sdwfs_cat['CH1_APMAG4'] - sdwfs_cat['CH2_APMAG4']
sdwfs_cat['CH3_CH4_COLOR'] = sdwfs_cat['CH3_APMAG4'] - sdwfs_cat['CH4_APMAG4']

# Require SNR cuts of > 5 in all bands for a clean sample
sdwfs_cat = sdwfs_cat[(sdwfs_cat['CH1_APFLUX4'] / sdwfs_cat['CH1_APFLUXERR4'] >= 5) &
                      (sdwfs_cat['CH2_APFLUX4'] / sdwfs_cat['CH2_APFLUXERR4'] >= 5) &
                      (sdwfs_cat['CH3_APFLUX4'] / sdwfs_cat['CH3_APFLUXERR4'] >= 5) &
                      (sdwfs_cat['CH4_APFLUX4'] / sdwfs_cat['CH4_APFLUXERR4'] >= 5)]

# # Apply magnitude cuts to the catalog
# sdwfs_cat = sdwfs_cat[(sdwfs_cat['CH1_APMAG4'] > 10.0) &   # 3.6um bright-end
#                       (sdwfs_cat['CH2_APMAG4'] > 10.45) &  # 4.5um bright-end
#                       (sdwfs_cat['CH2_APMAG4'] <= 17.46)]  # 4.5um faint-end

# Make AGN selections using the Stern+05 wedge selection
stern_AGN = sdwfs_cat[(sdwfs_cat['CH3_CH4_COLOR'] > 0.6) &
                      (sdwfs_cat['CH1_CH2_COLOR'] > 0.2 * (sdwfs_cat['CH3_CH4_COLOR'] + 0.18)) &
                      (sdwfs_cat['CH1_CH2_COLOR'] > 2.5 * (sdwfs_cat['CH3_CH4_COLOR'] - 3.5))]

# Identify the IDs of the objects outside the Stern-wedge selection to create a sample of non-AGN.
stern_complement_ids = list(set(sdwfs_cat['ID']) - set(stern_AGN['ID']))
non_agn = sdwfs_cat[np.in1d(sdwfs_cat['ID'], stern_complement_ids)]

# Bin the data in both redshift and color
z_bins = np.arange(0., 1.7, 0.2)
color_bins = np.arange(0., 1.5, 0.05)
agn_binned, _, _ = np.histogram2d(stern_AGN['PHOT_Z'], stern_AGN['CH1_CH2_COLOR'], bins=(z_bins, color_bins))
non_agn_binned, _, _ = np.histogram2d(non_agn['PHOT_Z'], non_agn['CH1_CH2_COLOR'], bins=(z_bins, color_bins))

# Combine the two histograms for the total counts
total_binned = agn_binned + non_agn_binned

# Give bins as a fraction of AGN purity
agn_purity = agn_binned / total_binned