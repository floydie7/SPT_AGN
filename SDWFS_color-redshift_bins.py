"""
SDWFS_color-redshift_bins.py
Author: Benjamin Floyd

Creates histograms from the color-redshift plane using SDWFS data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
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
z_bins = np.arange(0., 1.75+0.4, 0.2)
color_bins = np.arange(0., 1.5, 0.05)
agn_binned, _, _ = np.histogram2d(stern_AGN['PHOT_Z'], stern_AGN['CH1_CH2_COLOR'], bins=(z_bins, color_bins))
non_agn_binned, _, _ = np.histogram2d(non_agn['PHOT_Z'], non_agn['CH1_CH2_COLOR'], bins=(z_bins, color_bins))

# Combine the two histograms for the total counts
total_binned = agn_binned + non_agn_binned

# Compute the contamination of non-AGN as the complementary CDF
non_agn_ccdf = 1 - np.cumsum(non_agn_binned / np.sum(non_agn_binned, axis=1)[:, None], axis=1)


def purity_to_color_threshold(thresh: float) -> np.array:
    # Find the first occurrence of the purity above threshold
    purity_idx = np.argmax(non_agn_ccdf <= thresh, axis=1)

    # Give the corresponding color to the purity threshold
    color_threshold = color_bins[purity_idx]
    return color_threshold


# Compute the color thresholds for 90% and 80% purities
purity_90_color = purity_to_color_threshold(thresh=0.1)
purity_80_color = purity_to_color_threshold(thresh=0.2)

# Export data to file
# data = {z_bin: color_90 for z_bin, color_90 in zip(z_bins[:-1], purity_90_color)}
# with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_AGN_purity_90_color-redshift.json', 'w') as f:
#     json.dump(data, f)

# Report stats
for i, (color_90, color_80) in enumerate(zip(purity_90_color, purity_80_color)):
    if i + 1 >= len(z_bins):
        print(f'In redshift bin: z < {z_bins[i]:.1f}:\n'
              f'\tColor corresponding to 90% purity: {color_90:.2f}\n'
              f'\tColor corresponding to 80% purity: {color_80:.2f}')
    else:
        print(f'In redshift bin: {z_bins[i]:.1f} < z < {z_bins[i+1]:.1f}:\n'
              f'\tColor corresponding to 90% purity: {color_90:.2f}\n'
              f'\tColor corresponding to 80% purity: {color_80:.2f}')

#% make plot
z_bin_centers = np.diff(z_bins) + z_bins[:-1]
fig, ax = plt.subplots()
ax.hexbin(non_agn['PHOT_Z'], non_agn['CH1_CH2_COLOR'], gridsize=50, extent=(0., 1.7, 0., 1.5), cmap='Blues', bins=None, mincnt=1)
ax.hexbin(stern_AGN['PHOT_Z'], stern_AGN['CH1_CH2_COLOR'], gridsize=50, extent=(0., 1.7, 0., 1.5), cmap='Reds', bins=None, mincnt=1, alpha=0.5)
ax.step(z_bins[:-1], purity_90_color, color='tab:orange', lw=2, label='90% threshold')
ax.step(z_bins[:-1], purity_80_color, color='tab:green', lw=2, label='80% threshold')
ax.axhline(y=0.7, color='k', lw=2, ls='--', label=r'$[3.6] - [4.5] \geq 0.7$')
ax.legend()
ax.set(xlabel='Photometric Redshift', ylabel='[3.6] - [4.5] (Vega)', ylim=[0, 1.5], xlim=[0, 1.7])
plt.show()
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/Plots/SDWFS_color-redshift_AGN_purity_options.pdf')
