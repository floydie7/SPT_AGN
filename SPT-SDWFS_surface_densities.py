"""
SPT-SDWFS_surface_densities.py
Benjamin Floyd

Computes the surface densities of both surveys of IR-bright AGN
"""

from astropy.table import QTable, vstack
import numpy as np
from scipy.interpolate import interp1d
import json
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

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

# Read in all catalogs
sptcl_iragn = QTable.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN_no-stars.fits')
sdwfs_iragn = QTable.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_full-field_IRAGN.fits')

# Read in the color selection function
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    color_threshold_data = json.load(f)
redshift_bins = color_threshold_data['redshift_bins']
color_thresholds = color_threshold_data['purity_90_colors']
color_redshift_threshold_function = interp1d(redshift_bins[:-1], color_thresholds, kind='previous')

# Limit SDWFS data to z ≤ 1.8
sdwfs_iragn = sdwfs_iragn[sdwfs_iragn['REDSHIFT'] <= redshift_bins[-2]]

# For SDWFS we need to find the true selection membership values for each galaxy based on their photo-z
sdwfs_color_thresholds = color_redshift_threshold_function(sdwfs_iragn['REDSHIFT'])
sdwfs_iragn['TRUE_SELECTION_MEMBERSHIP'] = [sdwfs_iragn[f'SELECTION_MEMBERSHIP_{color:.2f}'][i]
                                            for i, color in enumerate(sdwfs_color_thresholds)]

# Get the areas for all lines of sight
sptcl_iragn_grp = sptcl_iragn.group_by('SPT_ID')
spt_areas = []
for cluster in sptcl_iragn_grp.groups:
    area = calculate_area([cluster['MASK_NAME'][0]])
    spt_areas.append(area)
spt_areas = u.Quantity(spt_areas).sum()

sdwfs_iragn_area = calculate_area([sdwfs_iragn['MASK_NAME'][0]])

# Apply our µ > 0.5 cut
sptcl_iragn_agn = sptcl_iragn[sptcl_iragn['SELECTION_MEMBERSHIP'] >= 0.5]
sdwfs_iragn_agn = sdwfs_iragn[sdwfs_iragn['TRUE_SELECTION_MEMBERSHIP'] >= 0.5]

# Compute the surface densities
sptcl_iragn_agn_counts = sptcl_iragn_agn['COMPLETENESS_CORRECTION'].sum()
sdwfs_iragn_agn_counts = sdwfs_iragn_agn['COMPLETENESS_CORRECTION'].sum()

sptcl_iragn_agn_surf_den = sptcl_iragn_agn_counts / spt_areas
sdwfs_iragn_agn_surf_den = sdwfs_iragn_agn_counts / sdwfs_iragn_area

#%%
print(f"""---
SPTcl-IRAGN counts (corrected): {sptcl_iragn_agn_counts:g}
SDWFS AGN counts (corrected): {sdwfs_iragn_agn_counts:g}

Total SPT Survey Area: {spt_areas:.2f}
Total SDWFS Area: {sdwfs_iragn_area:.2f}

SPT surface density: {sptcl_iragn_agn_surf_den.to(u.arcmin**-2):.3f}
SDWFS surface density: {sdwfs_iragn_agn_surf_den.to(u.arcmin**-2):.3f}""")