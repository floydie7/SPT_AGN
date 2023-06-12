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

sdwfs_surf_den_binned = []
for z in z_bin_centers:
    color_threshold = agn_purity_color(z)
    field_agn = sdwfs_iragn[sdwfs_iragn[f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'] >= 0.5]
    no_field_agn = field_agn['COMPLETENESS_CORRECTION'].sum()

    sdwfs_surf_den_binned.append(no_field_agn / sdwfs_area)
sdwfs_surf_den_binned = u.Quantity(sdwfs_surf_den_binned)

los_surf_den_binned = []
for cluster_bin in sptcl_iragn_binned.groups:
    los_agn = cluster_bin[cluster_bin['SELECTION_MEMBERSHIP'] >= 0.5]
    # Group the clusters within the redshift bin
    los_agn_grp = los_agn.group_by('SPT_ID')
    # Get the area of all the clusters in the redshift bin
    spt_mask_files = [cluster['MASK_NAME'][0] for cluster in los_agn_grp.groups]
    cluster_bin_area = calculate_area(spt_mask_files)
    # AGN number counts
    no_los_agn = los_agn_grp['COMPLETENESS_CORRECTION'].sum()

    los_surf_den_binned.append(no_los_agn / cluster_bin_area)

los_surf_den_binned = u.Quantity(los_surf_den_binned)

fig, ax1 = plt.subplots()
ax1.bar(z_bin_centers, (los_surf_den_binned.to(u.arcmin ** -2)).value, width=np.diff(z_bins), label='SPT LoS',
        alpha=0.65)
ax1.bar(z_bin_centers, (sdwfs_surf_den_binned.to(u.arcmin ** -2)).value, width=np.diff(z_bins), label='SDWFS (binned)',
        alpha=0.45)
# ax1.bar(z_bin_centers, (sdwfs_surf_den_means.to(u.arcmin**-2)).value, width=np.diff(z_bins), label='SDWFS (means)', alpha=0.35)
ax1.legend()
ax1.set(xlabel=r'$z$', ylabel=r'$\Sigma_{AGN}$ [arcmin$^{-2}$]', xlim=[0., 1.8])
ax2 = ax1.twinx()
min_y, max_y = ax1.get_ylim()
ax2.set(ylabel=r'$\Sigma_{AGN}$ [per FoV ($\sim 25$ arcmin$^2$)]', ylim=[min_y * 25, max_y * 25])
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/SPT-SDWFS_surf_overdens.pdf')
plt.show()


#%% Looking closer at the 0.8 < z < 1.0 bin for SDWFS
redshift = 0.9
color_threshold = agn_purity_color(redshift)
sdwfs_problem_bin = sdwfs_iragn[sdwfs_iragn[f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'] >= 0.5]

fig, ax = plt.subplots()
ax.hist(sdwfs_problem_bin['REDSHIFT'], bins=np.arange(0., 3.5, 0.1))
ax.set(title=fr'SDWFS Galaxies with $[3.5] - [4.5] \geq {color_threshold:.2f}$',
       xlabel='Photometric Redshift', ylabel='Number of Galaxies')
fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/SDWFS-IRAGN_Photo-z_hist_ColorThresh{color_threshold:.2f}.pdf')
plt.show()
