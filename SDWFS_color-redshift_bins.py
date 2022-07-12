"""
SDWFS_color-redshift_bins.py
Author: Benjamin Floyd

Creates histograms from the color-redshift plane using SDWFS data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from scipy.interpolate import interp1d
import scipy.optimize as op

# plt.style.use('tableau-colorblind10')

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

#%% Bin the data in both redshift and color
z_bins = np.arange(0., 1.75+0.4, 0.2)
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

#%% make color-contamination plots
fig, ax = plt.subplots()
color_range = np.linspace(color_bins[0], color_bins[-2], num=100)
for i, z in enumerate(z_bins[:-1]):
    # ax.step(color_bins[:-1], contamination_ratios[i, :], label=fr'${z:.1f} < z < {z_bins[i+1]:.1f}$')
    ax.plot(color_range, contam_interp(color_range)[i], label=fr'${z:.1f} < z < {z_bins[i+1]:.1f}$')
ax.axhline(0.1, ls='--', c='k')
ax.legend()
ax.set(xlabel='[3.6] - [4.5]', ylabel='Contamination')
plt.show()
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/Plots/SDWFS_AGN_contamination_curves.pdf')

#%% Find the 90% purity colors
purity_90_color = []
for i in range(len(z_bins[:-1])):
    inverse_contam_interp = lambda color, contam_level: contam_interp(color)[i] - contam_level
    color = op.brentq(inverse_contam_interp, a=color_bins[0], b=color_bins[-2], args=(0.1,))
    purity_90_color.append(color)

#%% Write data to file
data = {'purity_90_colors': purity_90_color}
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color.json', 'w') as f:
    json.dump(data, f)

#%% make plot
z_bin_centers = np.diff(z_bins) + z_bins[:-1]
fig, ax = plt.subplots()
ax.hexbin(non_agn['PHOT_Z'], non_agn['CH1_CH2_COLOR'], gridsize=50, extent=(0., 1.8, 0., 1.5),
          cmap='Blues', bins=None, mincnt=1)
ax.hexbin(stern_AGN['PHOT_Z'], stern_AGN['CH1_CH2_COLOR'], gridsize=50, extent=(0., 1.8, 0., 1.5), cmap='Reds',
          bins=None, mincnt=1, alpha=0.6)
ax.step(z_bins[:-1], purity_90_color, color='tab:green', lw=2, label='90% threshold')
# ax.step(z_bins[:-1], purity_80_color, color='tab:green', lw=2, label='80% threshold')
ax.axhline(y=0.7, color='k', lw=2, ls='--', label=r'$[3.6] - [4.5] \geq 0.7$')
ax.legend()
ax.set(xlabel='Photometric Redshift', ylabel='[3.6] - [4.5] (Vega)', ylim=[0, 1.5], xlim=[0, 1.8])
plt.show()
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/Plots/SDWFS_color-redshift_AGN_90_purity.pdf')
