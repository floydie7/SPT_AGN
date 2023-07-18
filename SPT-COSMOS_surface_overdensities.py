"""
SPT-COSMOS_surface_overdensities.py
Author: Benjamin Floyd

Replicates the same calculations as SPT-SDWFS_surface_overdensities.py but using COSMOS and adjusted that we don't have
as much information on-hand for COSMOS as we do for SDWFS.
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
cosmos_cutouts = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Misc/COSMOS15_catalog_cutouts.fits')
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Bin the cluster catalog by redshift
cluster_bins = np.digitize(sptcl_iragn['REDSHIFT'], z_bins)
sptcl_iragn_binned = sptcl_iragn.group_by(cluster_bins)

# Get the bin centers
z_bin_centers = np.diff(z_bins) / 2 + z_bins[:-1]

cosmos_surf_den = []
for z in z_bin_centers:
    color_threshold = agn_purity_color(z)
    field_agn = cosmos_cutouts[cosmos_cutouts['SPLASH_1_MAG'] - cosmos_cutouts['SPLASH_2_MAG'] >= color_threshold]
    try:
        field_agn_grp = field_agn.group_by('CUTOUT_ID')
    except IndexError:
        cosmos_surf_den.append(0 * u.arcmin**-2)
        continue
        
    cutout_surf_den = []
    for cutout in field_agn_grp.groups:
        cutout_area = 25 * u.arcmin**2  # Assume all area within cutout bounds is good.
        no_cutout_agn = cutout['COMPLETENESS_CORRECTION'].sum()
        cutout_surf_den.append(no_cutout_agn / cutout_area)
    cosmos_surf_den.append(u.Quantity(cutout_surf_den))
cosmos_surf_den_means = u.Quantity([surf_den.mean() for surf_den in cosmos_surf_den])
cosmos_surf_den_std = u.Quantity([surf_den.std() for surf_den in cosmos_surf_den])

los_surf_den_binned = []
for cluster_bin, z in zip(sptcl_iragn_binned.groups, z_bin_centers):
    los_agn = cluster_bin[cluster_bin['SELECTION_MEMBERSHIP'] >= 0.5]
    # Group the clusters within the redshift bin
    los_agn_grp = los_agn.group_by('SPT_ID')
    # Get the area of all the clusters in the redshift bin
    spt_mask_files = [cluster['MASK_NAME'][0] for cluster in los_agn_grp.groups]
    cluster_bin_area = calculate_area(spt_mask_files)
    # AGN number counts
    no_los_agn = los_agn_grp['COMPLETENESS_CORRECTION'].sum()
    print(f'{z = :.2f}: {no_los_agn = :.2f}, {cluster_bin_area = :.2f}, Num of FoVs: {len(los_agn_grp.groups.keys["SPT_ID"])}')

    los_surf_den_binned.append(no_los_agn / cluster_bin_area)

los_surf_den_binned = u.Quantity(los_surf_den_binned)

fig, ax = plt.subplots()
ax.bar(z_bin_centers, los_surf_den_binned.to_value(u.arcmin**-2), width=np.diff(z_bins), label='SPTcl LoS', alpha=0.65)
ax.bar(z_bin_centers, cosmos_surf_den_means.to_value(u.arcmin**-2), width=np.diff(z_bins), label='COSMOS', alpha=0.45)
ax.set(xlabel=r'$z$', ylabel=r'$\Sigma_{AGN}$ [arcmin$^{-2}$]', xlim=[0., 1.8])
plt.show()