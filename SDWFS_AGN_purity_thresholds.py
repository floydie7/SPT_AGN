"""
SDWFS_AGN_purity_thresholds.py
Author: Benjamin Floyd

Computes the AGN purity color thresholds and outputs a file storing the color thresholds and curves.
"""

import json
from json import JSONEncoder
from typing import Any

import numpy as np
from astropy.table import Table, join
from scipy.interpolate import interp1d
from scipy.optimize import brentq


class NumpyArrayEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


# Read in the SDWFS photometric catalog
sdwfs_main = Table.read(
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
sdwfs_photz = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/mbz_v0.06_prior_bri12_18p8.cat.gz',
                         names=['ID', 'PHOT_Z', 'col3', 'col4', 'col5', 'col6', 'col7'],
                         format='ascii',
                         include_names=['ID', 'PHOT_Z'])

# Join the two catalogs together
sdwfs_cat = join(sdwfs_main, sdwfs_photz, keys='ID')

# For convenience, add columns for the two colors
sdwfs_cat['CH1_CH2_COLOR'] = sdwfs_cat['CH1_APMAG4'] - sdwfs_cat['CH2_APMAG4']
sdwfs_cat['CH3_CH4_COLOR'] = sdwfs_cat['CH3_APMAG4'] - sdwfs_cat['CH4_APMAG4']

# Impose our magnitude cuts used in AGN selection for 3.6 um and 4.5 um and require SNR > 5 for 5.8 um and 8.0 um.
sdwfs_cat = sdwfs_cat[(sdwfs_cat['CH1_APMAG4'] > 10.00) & (sdwfs_cat['CH1_APMAG4'] <= 18.30) &  # 10. < [3.6] <= 18.3
                     (sdwfs_cat['CH2_APMAG4'] > 10.45) & (sdwfs_cat['CH2_APMAG4'] <= 17.48) &   # 10.45 < [4.5] <= 17.48
                      (sdwfs_cat['CH3_APFLUX4'] / sdwfs_cat['CH3_APFLUXERR4'] >= 5) &           # SNR_5.8 >= 5
                      (sdwfs_cat['CH4_APFLUX4'] / sdwfs_cat['CH4_APFLUXERR4'] >= 5)]            # SNR_8.0 >= 5

# Make AGN selections using the Stern+05 wedge selection
stern_AGN = sdwfs_cat[(sdwfs_cat['CH3_CH4_COLOR'] > 0.6) &
                      (sdwfs_cat['CH1_CH2_COLOR'] > 0.2 * sdwfs_cat['CH3_CH4_COLOR'] + 0.18) &
                      (sdwfs_cat['CH1_CH2_COLOR'] > 2.5 * sdwfs_cat['CH3_CH4_COLOR'] - 3.5)]

# Identify the IDs of the objects outside the Stern-wedge selection to create a sample of non-AGN.
stern_complement_ids = list(set(sdwfs_cat['ID']) - set(stern_AGN['ID']))
non_agn = sdwfs_cat[np.in1d(sdwfs_cat['ID'], stern_complement_ids)]

# Bin the data in both redshift and color
z_bins = np.arange(0., 1.75 + 0.4, 0.2)
color_bins = np.arange(0., 1.5, 0.05)
agn_binned, _, _ = np.histogram2d(stern_AGN['PHOT_Z'], stern_AGN['CH1_CH2_COLOR'], bins=(z_bins, color_bins))
non_agn_binned, _, _ = np.histogram2d(non_agn['PHOT_Z'], non_agn['CH1_CH2_COLOR'], bins=(z_bins, color_bins))

# Combine the two histograms for the total counts
total_binned = agn_binned + non_agn_binned

# Loop over the color bins to compute the purities
contamination_ratios = []
for i, _ in enumerate(color_bins[:-1]):
    # Add all objects above our color cut
    non_agn_above_color = np.sum(non_agn_binned[:, i:], axis=1)
    total_above_color = np.sum(total_binned[:, i:], axis=1)

    # Compute the contamination ratio
    contamination = non_agn_above_color / total_above_color
    contamination_ratios.append(contamination)

# Combine the results and return to [redshift, color] order.
contamination_ratios = np.array(contamination_ratios).T

# Build interpolator
contam_interp = interp1d(color_bins[:-1], contamination_ratios, kind='linear', axis=1)

# Find the 90% purity colors
purity_90_color = []
purity_92_color = []
purity_95_color = []
for i in range(len(z_bins[:-1])):
    def inverse_contam_interp(color, contam_level):
        return contam_interp(color)[i] - contam_level


    color_90 = brentq(inverse_contam_interp, a=color_bins[0], b=color_bins[-2], args=(0.1,))
    color_92 = brentq(inverse_contam_interp, a=color_bins[0], b=color_bins[-2], args=(0.08,))
    color_95 = brentq(inverse_contam_interp, a=color_bins[0], b=color_bins[-2], args=(0.05,))
    purity_90_color.append(color_90)
    purity_92_color.append(color_92)
    purity_95_color.append(color_95)

# Write data to file
data = {'purity_90_colors': purity_90_color,
        'purity_92_colors': purity_92_color,
        'purity_95_colors': purity_95_color,
        'redshift_bins': z_bins,
        'color_bins': color_bins,
        'purity_ratios': 1 - contamination_ratios}
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'w') as f:
    json.dump(data, f, cls=NumpyArrayEncoder)
