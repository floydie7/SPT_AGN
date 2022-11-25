"""
SPT-SDWFS_SNR.py
Author: Benjamin Floyd

Estimates a signal-to-noise ratio of cluster-to-background AGN number counts.
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
    total_area = u.Quantity(areas).sum()
    return total_area


# Read in the catalogs
sdwfs_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

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

cl_bkg_snr, cl_bkg_snr_zbin = [], []
for z, cluster_bin, field_bin in zip(z_bin_centers, sptcl_iragn_binned.groups, sdwfs_iragn_binned.groups):
    # Get the area of all the clusters in the redshift bin
    spt_mask_files = [cluster['MASK_NAME'][0] for cluster in cluster_bin.group_by('SPT_ID').groups]
    cluster_bin_area = calculate_area(spt_mask_files)

    sdwfs_zbin_mask_files = [cutout['MASK_NAME'][0] for cutout in field_bin.group_by('CUTOUT_ID').groups]
    field_bin_area = calculate_area(sdwfs_zbin_mask_files)

    # Get the color threshold for this redshift bin
    color_threshold = agn_purity_color(z)

    # Select the probable AGN
    los_agn = cluster_bin[cluster_bin['SELECTION_MEMBERSHIP'] >= 0.5]
    field_agn = sdwfs_iragn[sdwfs_iragn[f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'] >= 0.5]
    field_agn_zbin = field_bin[field_bin[f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'] >= 0.5]

    # AGN number counts
    no_los_agn = np.sum(los_agn['COMPLETENESS_CORRECTION'])
    no_field_agn = np.sum(field_agn['COMPLETENESS_CORRECTION'])
    no_field_agn_zbin = np.sum(field_agn_zbin['COMPLETENESS_CORRECTION'])

    # Estimate the number of cluster AGN surface densities
    # cluster_surf_den = (no_los_agn - no_field_agn) / cluster_bin_area
    # cluster_surf_den_z_bin = (no_los_agn - no_field_agn_zbin) / cluster_bin_area
    cluster_surf_den = (no_los_agn / cluster_bin_area - no_field_agn / sdwfs_area)
    cluster_surf_den_z_bin = (no_los_agn / cluster_bin_area - no_field_agn_zbin / field_bin_area)

    # Field Surface Densities
    field_surf_den = no_field_agn / sdwfs_area
    field_surf_den_zbin = no_field_agn_zbin / field_bin_area

    cl_bkg_snr.append(cluster_surf_den / field_surf_den)
    cl_bkg_snr_zbin.append(cluster_surf_den_z_bin / field_surf_den_zbin)

# %%
fig, (ax, bx) = plt.subplots(nrows=2, sharex='col', figsize=(6.4, 4.8 * 2))
ax.bar(z_bin_centers, cl_bkg_snr, width=np.diff(z_bins))
ax.set(xlabel='redshift', ylabel=r'SNR [$\Sigma_{AGN,cl}$ / $\Sigma_{AGN,bkg}$]', title='Over all SDWFS')

bx.bar(z_bin_centers, cl_bkg_snr_zbin, width=np.diff(z_bins))
bx.set(xlabel='redshift', ylabel=r'SNR [$\Sigma_{AGN,cl}$ / $\Sigma_{AGN,bkg}$]', title='Over SDWFS @ z')
plt.show()

# %%
# # Narrow in on the wierd redshift bin
# sdwfs_iragn_z07 = sdwfs_iragn[(sdwfs_iragn['REDSHIFT'] > z_bins[3]) & (sdwfs_iragn['REDSHIFT'] <= z_bins[4])]
# sdwfs_iragn_z07_mu_cut = sdwfs_iragn_z07[
#     sdwfs_iragn_z07[f'SELECTION_MEMBERSHIP_{agn_purity_color(np.mean([z_bins[3], z_bins[4]])):.2f}'] >= 0.5]
#
# bins = np.arange(z_bins[3], z_bins[4] + 0.025, 0.025)
#
# fig, ax = plt.subplots()
# ax.hist(sdwfs_iragn_z07['REDSHIFT'], bins=bins)
# ax.hist(sdwfs_iragn_z07_mu_cut['REDSHIFT'], bins=bins, label=r'$\mu_{AGN} \geq 0.5$')
# ax.legend()
# ax.set(xlabel='redshift', ylabel=r'$N_{AGN}$', yscale='log')
# plt.show()
#
# fig, ax = plt.subplots()
# ax.scatter(sdwfs_iragn_z07['REDSHIFT'], sdwfs_iragn_z07['I1_MAG_APER4'] - sdwfs_iragn_z07['I2_MAG_APER4'])
# ax.axhline(agn_purity_color(np.mean([z_bins[3], z_bins[4]])))
# ax.set(xlabel='redshift', ylabel='[3.6] - [4.5]')
# plt.show()
