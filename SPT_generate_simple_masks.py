"""
SPT_generate_simple_masks.py
Author: Benjamin Floyd

Creates (or updates) simple mask files based on the original masks for testing purposes.
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join, unique
from astropy.wcs import WCS

id_pattern = re.compile(r'SPT-CLJ\d+-\d+')


def make_quarter_mask(mask_name: str, cluster_center: SkyCoord):
    # Read in the original mask
    mask_img, mask_hdr = fits.getdata(mask_name, header=True)
    mask_wcs = WCS(mask_hdr)

    # Initialize the new mask
    new_mask = np.zeros_like(mask_img)

    # Get the pixel coordinates of the cluster center (as array indices)
    cluster_center_pix = np.floor(cluster_center.to_pixel(wcs=mask_wcs, origin=0, mode='wcs')).astype(int).reshape(2, )

    # Set all pixels in the first quadrant with the cluster center as the origin to `1`
    new_mask[cluster_center_pix[1]:, cluster_center_pix[0]:] = 1

    # Get the filename of the original mask
    path_name = Path(mask_name)
    new_path_name = path_name.parent / 'quarter_masks' / path_name.name
    new_mask_hdu = fits.PrimaryHDU(new_mask, header=mask_hdr)
    new_mask_hdu.writeto(new_path_name, overwrite=True)


# Read in the look-up table for SPT-SZ IDs
with open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_observed_to_official_ids.json', 'r') as f:
    obs_to_off = json.load(f)

# Collect all original masks
masks_files = [*glob.glob(f'Data_Repository/Project_Data/SPT-IRAGN/Masks/SPT-SZ_2500d/*.fits'),
               *glob.glob(f'Data_Repository/Project_Data/SPT-IRAGN/Masks/SPTpol_100d/*.fits')]

# Read in the SPT cluster catalog. We will use real data to source our mock cluster properties.
Bocquet = Table.read(f'Data_Repository/Catalogs/SPT/SPT_catalogs/2500d_cluster_sample_Bocquet18.fits')

# For the 20 common clusters between SPT-SZ 2500d and SPTpol 100d surveys we want to update the cluster information from
# the more recent survey. Thus, we will merge the SPT-SZ and SPTpol catalogs together.
Huang = Table.read(f'Data_Repository/Catalogs/SPT/SPT_catalogs/sptpol100d_catalog_huang19.fits')

# First we need to rename several columns in the SPTpol 100d catalog to match the format of the SPT-SZ catalog
Huang.rename_columns(['Dec', 'xi', 'theta_core', 'redshift', 'redshift_unc'],
                     ['DEC', 'XI', 'THETA_CORE', 'REDSHIFT', 'REDSHIFT_UNC'])

# Now, merge the two catalogs
SPTcl = join(Bocquet, Huang, join_type='outer')
SPTcl.sort(keys=['SPT_ID', 'field'])  # Sub-sorting by 'field' puts Huang entries first
SPTcl = unique(SPTcl, keys='SPT_ID', keep='first')  # Keeping Huang entries over Bocquet
SPTcl.sort(keys='SPT_ID')  # Resort by ID

# Make sure all the masks have matches in the catalog
masks_files = [f for f in masks_files if re.search(r'SPT-CLJ\d+-\d+', f).group(0) in SPTcl['SPT_ID']]

for file_name in masks_files:
    cluster_id = id_pattern.search(file_name).group(0)
    sz_center = SPTcl['RA', 'DEC'][SPTcl['SPT_ID'] == obs_to_off.get(cluster_id, cluster_id)]
    sz_center_skycoord = SkyCoord(sz_center['RA'], sz_center['DEC'], unit='deg')

    print(f'Working on {cluster_id}')
    make_quarter_mask(file_name, sz_center_skycoord)
