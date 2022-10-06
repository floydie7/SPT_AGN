"""
SPT-SDWFS_SNR.py
Author: Benjamin Floyd

Estimates a signal-to-noise ratio of cluster-to-background AGN number counts.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

# Read in the catalogs
sdwfs_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Bin the cluster catalog by redshift
cluster_bins = np.digitize(sptcl_iragn['REDSHIFT'], z_bins)
sptcl_iragn_binned = sptcl_iragn.group_by(cluster_bins)

# Get the bin centers
z_bin_centers = np.diff(z_bins) / 2 + z_bins[:-1]

cl_bkg_snr = []
for z, cluster_bin in zip(z_bin_centers, sptcl_iragn_binned.groups):
    # Get the color threshold for this redshift bin
    color_threshold = agn_purity_color(z)

    # Select the probable AGN
    los_agn = cluster_bin[cluster_bin['SELECTION_MEMBERSHIP'] >= 0.5]
    field_agn = sdwfs_iragn[sdwfs_iragn[f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'] >= 0.5]

    # AGN number counts
    no_los_agn = np.sum(los_agn['COMPLETENESS_CORRECTION'])
    no_field_agn = np.sum(field_agn['COMPLETENESS_CORRECTION'])

    # Estimate the number of cluster AGN
    no_cluster_agn = np.abs(no_los_agn - no_field_agn)

    cl_bkg_snr.append(no_cluster_agn / no_los_agn)

fig, ax = plt.subplots()
ax.bar(z_bin_centers, cl_bkg_snr, width=np.diff(z_bins))
ax.set(xlabel='redshift', ylabel='SNR [# cluster AGN / # total AGN]')
plt.show()
