"""
SDWFS_background_prior_dist.py
Author: Benjamin Floyd

Creates the ancillary file containing the background prior distributions for the different purity color thresholds used.
"""

import json
from typing import Any

import astropy.units as u
import numpy as np
from astro_compendium.utils.small_poisson import small_poisson
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


# Read in the SDWFS AGN catalog
sdwfs_agn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')

# List all mask files
mask_files = [cutout['MASK_NAME'][0] for cutout in sdwfs_agn.group_by('CUTOUT_ID').groups]

# Get the color thresholds used in catalog creation
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    color_threshold_data = json.load(f)
    color_thresholds = color_threshold_data['purity_90_colors']

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

# For each color selection threshold, add all objects applying the appropriate corrections
completeness_correction = sdwfs_agn['COMPLETENESS_CORRECTION']
selection_membership_columns = [colname for colname in sdwfs_agn.colnames if 'SELECTION_MEMBERSHIP' in colname]

# numbers_agn = [np.sum(completeness_correction * sdwfs_agn[selection_membership_column])
#                for selection_membership_column in selection_membership_columns]
numbers_agn = []
for color in color_thresholds[:-1]:
    # cat = sdwfs_agn[sdwfs_agn['I1_MAG_APER4'] - sdwfs_agn['I2_MAG_APER4'] >= color]
    cat = sdwfs_agn[sdwfs_agn[f'SELECTION_MEMBERSHIP_{color:.2f}'] >= 0.5]
    numbers_agn.append(np.sum(cat['COMPLETENESS_CORRECTION']))

# Create surface density array
agn_surf_den = np.array(numbers_agn) / total_area

# Also generate Poisson errors for the surface densities
upper_errors, lower_errors = small_poisson(numbers_agn)
agn_surf_den_uerr = upper_errors / total_area
agn_surf_den_lerr = lower_errors / total_area

# Additionally, calculate the symmetric Poisson errors for the surface densities
agn_surf_den_symm_errs = np.sqrt(numbers_agn) / total_area

# Store data in file (All arrays are in units of arcmin^-2)
agn_surf_den_data = {'agn_surf_den': agn_surf_den.to_value(u.arcmin ** -2),
                     'agn_surf_den_uerr': agn_surf_den_uerr.to_value(u.arcmin ** -2),
                     'agn_surf_den_lerr': agn_surf_den_lerr.to_value(u.arcmin ** -2),
                     'agn_surf_den_err': agn_surf_den_symm_errs.to_value(u.arcmin ** -2),
                     'color_thresholds': color_thresholds}
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
          'SDWFS_background_prior_distributions_mu_cut_updated_cuts.json', 'w') as f:
    json.dump(agn_surf_den_data, f, cls=NumpyArrayEncoder)
