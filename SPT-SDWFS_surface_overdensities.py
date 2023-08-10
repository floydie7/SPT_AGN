"""
SPT-SDWFS_surface_overdensities.py
Author: Benjamin Floyd

Examines the relative surface densities of AGN in the two samples. Diagnostic for fixing SNR problem.
"""

import json

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import interp1d


def calculate_area(mask_files: list) -> u.Quantity:
    # Read in each mask file and calculate the allowable area
    areas = []
    for mask_file in mask_files:
        mask_img, mask_hdr = fits.getdata(mask_file, header=True)
        mask_wcs = WCS(mask_hdr)
        # Get the area of a pixel in angular units
        pixel_area = mask_wcs.proj_plane_pixel_area()
        # Find the total area of the image by adding all pixels and multiplying by the pixel area
        mask_area = np.count_nonzero(mask_img) * pixel_area
        areas.append(mask_area)
    # Compute total area in sample
    return u.Quantity(areas).sum()


# Read in the catalogs
sdwfs_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN_no_structure.fits')
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')
cosmos_cutouts = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Misc/COSMOS20_catalog_cutouts.fits')


# List all mask files
sdwfs_mask_files = [cutout['MASK_NAME'][0] for cutout in sdwfs_iragn.group_by('CUTOUT_ID').groups]
sdwfs_area = calculate_area(sdwfs_mask_files)

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Bin the cluster catalog by redshift
cluster_bins = np.digitize(sptcl_iragn['REDSHIFT'], z_bins)
sptcl_iragn_binned = sptcl_iragn.group_by(cluster_bins)

# Bin the field by redshift as well
field_bins = np.digitize(sdwfs_iragn['REDSHIFT'], z_bins)
sdwfs_iragn_binned = sdwfs_iragn.group_by(field_bins)

# Get the bin centers
z_bin_centers = np.diff(z_bins) / 2 + z_bins[:-1]

sdwfs_surf_den = []
for z in z_bin_centers:
    color_threshold = agn_purity_color(z)
    field_agn = sdwfs_iragn[sdwfs_iragn[f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'] >= 0.5]
    field_agn_grp = field_agn.group_by('CUTOUT_ID')

    cutout_surf_den = []
    for cutout in field_agn_grp.groups:
        cutout_mask_file = cutout['MASK_NAME'][0]
        cutout_area = calculate_area([cutout_mask_file])
        no_cutout_agn = cutout['COMPLETENESS_CORRECTION'].sum()
        cutout_surf_den.append(no_cutout_agn / cutout_area)
    sdwfs_surf_den.append(u.Quantity(cutout_surf_den))
sdwfs_surf_den_means = u.Quantity([surf_den.mean() for surf_den in sdwfs_surf_den])
sdwfs_surf_den_std = u.Quantity([surf_den.std() for surf_den in sdwfs_surf_den])

cosmos_surf_den = []
for z in z_bin_centers:
    color_threshold = agn_purity_color(z)
    field_agn = cosmos_cutouts[cosmos_cutouts['IRAC_CH1_MAG'] - cosmos_cutouts['IRAC_CH2_MAG'] >= color_threshold]
    field_agn_grp = field_agn.group_by('CUTOUT_ID')

    cutout_surf_den = []
    for cutout in field_agn_grp.groups:
        cutout_area = 25 * u.arcmin ** 2  # Assume all area within cutout bounds is good.
        no_cutout_agn = cutout['COMPLETENESS_CORRECTION'].sum()
        cutout_surf_den.append(no_cutout_agn / cutout_area)
    cosmos_surf_den.append(u.Quantity(cutout_surf_den))
cosmos_surf_den_means = u.Quantity([surf_den.mean() for surf_den in cosmos_surf_den])
cosmos_surf_den_std = u.Quantity([surf_den.std() for surf_den in cosmos_surf_den])

print('SPT LoS')
los_surf_den = []
for cluster_bin, z in zip(sptcl_iragn_binned.groups, z_bin_centers):
    los_agn = cluster_bin[cluster_bin['SELECTION_MEMBERSHIP'] >= 0.5]
    los_agn_grp = los_agn.group_by('SPT_ID')

    cluster_surf_den = []
    for cluster in los_agn_grp.groups:
        cluster_mask_file = cluster['MASK_NAME'][0]
        cluster_area = calculate_area([cluster_mask_file])
        no_los_agn = cluster['COMPLETENESS_CORRECTION'].sum()
        cluster_surf_den.append(no_los_agn / cluster_area)
    los_surf_den.append(u.Quantity(cluster_surf_den))
los_surf_den_means = u.Quantity([surf_den.mean() for surf_den in los_surf_den])
los_surf_den_std = u.Quantity([surf_den.std() for surf_den in los_surf_den])

#%%
# sdwfs_surf_den_medians = u.Quantity([np.median(surf_den) for surf_den in sdwfs_surf_den])
# cosmos_surf_den_medians = u.Quantity([np.median(surf_den) for surf_den in cosmos_surf_den])
# los_surf_den_medians = u.Quantity([np.median(surf_den) for surf_den in los_surf_den])

#%%
fig, ax1 = plt.subplots()
ax1.bar(z_bin_centers, los_surf_den_means.to_value(u.arcmin ** -2), width=np.diff(z_bins), label='SPT LoS',
        alpha=0.65)
ax1.bar(z_bin_centers, sdwfs_surf_den_means.to_value(u.arcmin ** -2), width=np.diff(z_bins), label='SDWFS (no struct)',
        alpha=0.45)
ax1.bar(z_bin_centers, cosmos_surf_den_means.to_value(u.arcmin ** -2), width=np.diff(z_bins), label='COSMOS2020',
        alpha=0.45)
ax1.errorbar(z_bin_centers, los_surf_den_means.to_value(u.arcmin ** -2), yerr=los_surf_den_std.to_value(u.arcmin ** -2),
             fmt='none', ecolor='tab:blue', capsize=5)
ax1.errorbar(z_bin_centers, sdwfs_surf_den_means.to_value(u.arcmin ** -2), yerr=sdwfs_surf_den_std.to_value(u.arcmin ** -2),
             fmt='none', ecolor='tab:orange', capsize=5)
ax1.errorbar(z_bin_centers, cosmos_surf_den_means.to_value(u.arcmin ** -2), yerr=cosmos_surf_den_std.to_value(u.arcmin ** -2),
             fmt='none', ecolor='tab:green', capsize=5)
ax1.legend()
ax1.set(xlabel=r'$z$', ylabel=r'$\Sigma_{AGN}$ [arcmin$^{-2}$]', xlim=[0., 1.8])
ax2 = ax1.twinx()
min_y, max_y = ax1.get_ylim()
ax2.set(ylabel=r'$\Sigma_{AGN}$ [per FoV ($\sim 25$ arcmin$^2$)]', ylim=[min_y * 25, max_y * 25])
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/SPT-SDWFS-no_struct-COSMOS20_surf_overdens_errorbars.pdf')
# plt.show()


#%% Looking closer at the 0.8 < z < 1.0 bin for SDWFS
# redshift = 0.9
# color_threshold = agn_purity_color(redshift)
# sdwfs_problem_bin = sdwfs_iragn[sdwfs_iragn[f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'] >= 0.5]
#
# fig, ax = plt.subplots()
# ax.hist(sdwfs_problem_bin['REDSHIFT'], bins=np.arange(0., 3.5, 0.1))
# ax.set(title=fr'SDWFS Galaxies with $[3.5] - [4.5] \geq {color_threshold:.2f}$',
#        xlabel='Photometric Redshift', ylabel='Number of Galaxies')
# # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/SDWFS-IRAGN_Photo-z_hist_ColorThresh{color_threshold:.2f}.pdf')
# plt.show()

#%%
sdwfs_surf_den_z08_1 = sdwfs_surf_den[4]
cosmos_surf_den_z08_1 = cosmos_surf_den[4]
los_surf_den_z08_1 = los_surf_den[4]
bins = np.arange(0, 2, 0.1)

fig, ax = plt.subplots()
ax.hist(los_surf_den_z08_1.to_value(u.arcmin**-2), bins=bins, alpha=0.65, label='SPTcl')
ax.hist(sdwfs_surf_den_z08_1.to_value(u.arcmin**-2), bins=bins, alpha=0.45, label='SDWFS')
ax.hist(cosmos_surf_den_z08_1.to_value(u.arcmin**-2), bins=bins, alpha=0.45, label='COSMOS2020')
ax.legend()
ax.set(xlabel=r'$\Sigma_{\rm AGN}$ [arcmin$^{-2}$]', ylabel='N')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/SPT-SDWFS-no_struct-COSMOS20_surf_den_z0.8_1_bin.pdf')
