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
    return u.Quantity(areas).sum()


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

#%%
cl_bkg_snr, cl_bkg_snr_var = [], []
for cluster_bin, field_mean, field_std in zip(sptcl_iragn_binned.groups, sdwfs_surf_den_means, sdwfs_surf_den_std):
    # Select the probable AGN
    los_agn = cluster_bin[cluster_bin['SELECTION_MEMBERSHIP'] >= 0.5]

    # Group the clusters within the redshift bin
    los_agn_grp = los_agn.group_by('SPT_ID')

    # Get the area of all the clusters in the redshift bin
    spt_mask_files = [cluster['MASK_NAME'][0] for cluster in los_agn_grp.groups]
    cluster_bin_area = calculate_area(spt_mask_files)

    # AGN number counts
    no_los_agn = los_agn_grp['COMPLETENESS_CORRECTION'].sum()

    # Estimate the number of cluster AGN in excess of the field counts for the cluster area
    cluster_excess = (no_los_agn - (field_mean * cluster_bin_area).value)

    # Convert the field surface density error to a number error using the cluster area
    field_to_field_err = (field_std * cluster_bin_area).value
    poisson_err = np.sqrt(field_mean * cluster_bin_area).value

    # Combine errors in quadrature
    field_err = np.sqrt(field_to_field_err**2 + poisson_err**2)

    # Reduce the field error by the number of clusters present in the redshift bin
    field_err /= np.sqrt(len(los_agn_grp.groups))

    # The signal-to-noise ratio will then be the excess number counts over the field (number count) error
    cl_bkg_snr.append(cluster_excess / field_err)
    cl_bkg_snr_var.append(field_err ** 2)
cl_bkg_snr = np.asarray(cl_bkg_snr)
cl_bkg_snr_var = np.asarray(cl_bkg_snr_var)

# %% Use inverse variance weighting to combine all clusters' SNRs to form a combined SNR for the catalog
catalog_snr = np.sum(cl_bkg_snr / cl_bkg_snr_var) / np.sum(1 / cl_bkg_snr_var)
catalog_snr_err = np.sum(1 / cl_bkg_snr_var) ** -0.5
print(f'Catalog SNR = {catalog_snr:.2f} +/- {catalog_snr_err:.3f}')

# %%
# fig, (ax, bx) = plt.subplots(nrows=2, sharex='col', figsize=(6.4, 4.8 * 2))
# ax.bar(z_bin_centers, cl_bkg_snr, width=np.diff(z_bins))
# ax.set(xlabel='redshift', ylabel=r'SNR [$\Sigma_{AGN,cl}$ / $\Sigma_{AGN,bkg}$]', title='Over all SDWFS')
#
# bx.bar(z_bin_centers, cl_bkg_snr_zbin, width=np.diff(z_bins))
# bx.set(xlabel='redshift', ylabel=r'SNR [$\Sigma_{AGN,cl}$ / $\Sigma_{AGN,bkg}$]', title='Over SDWFS @ z', xlim=[0, 1.8])
# plt.show()
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/SPT_Data/Plots/SPT-SDWFS_SNR.pdf')

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
