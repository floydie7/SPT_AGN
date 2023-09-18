"""
SDWFS_full_field_mask.py
Author: Benjamin Floyd

This script takes the Optical + IRAC mask that Mark Brodwin provided and converts it into the simpler binary mask that
we use in the IRAGN pipeline analyses.
"""
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

# Read in the original mask and get the WCS
orig_mask, orig_hdr = fits.getdata('Data_Repository/Images/Bootes/SDWFS/SDWFS_opt+irac.fits.gz', header=True)
mask_wcs = WCS(orig_hdr)

# The original mask has `0` as "good" pixel. Set these to `True` and all others to `False`
new_mask = (orig_mask == 0).astype(int)

# There are a couple of erroneous regions that are marked as good in the corners
new_mask[:790, :1130] = 0
new_mask[:790, 14730:] = 0

# We want to crop this image to be the same size as the coverage maps so we can easily combine them.
# Read in the I1 coverage map (the WCS and dimensions are the same as in I2) and get the WCS.
i1_cov_wcs = WCS('Data_Repository/Images/Bootes/SDWFS/I1_bootes.cov.fits')

# Get the footprint of the coverage map in array indices of the new mask
i1_cov_footprint = i1_cov_wcs.calc_footprint()
new_mask_cov_footprint_idx = mask_wcs.world_to_array_index(SkyCoord(i1_cov_footprint, unit='deg'))

# Crop the new mask to have the same shape as the coverage maps.
new_mask = new_mask[new_mask_cov_footprint_idx[0][0]:new_mask_cov_footprint_idx[0][1] + 1,
                    new_mask_cov_footprint_idx[1][0]:new_mask_cov_footprint_idx[1][2] + 1]

# Update the WCS of the mask
new_mask_wcs = mask_wcs[new_mask_cov_footprint_idx[0][0]:new_mask_cov_footprint_idx[0][1] + 1,
                        new_mask_cov_footprint_idx[1][0]:new_mask_cov_footprint_idx[1][2] + 1]

# Push all updates into the header
new_mask_hdr = orig_hdr.copy()
new_mask_hdr.update(new_mask_wcs.to_header(), **{'NAXIS1': new_mask_wcs.array_shape[0],
                                                 'NAXIS2': new_mask_wcs.array_shape[1]})
new_mask_hdr['HISTORY'] = 'Mask converted to simple binary mask'
new_mask_hdr['HISTORY'] = f'by Benjamin Floyd, {datetime.today()}'

# Write the new mask back to disk
fits.writeto('Data_Repository/Images/Bootes/SDWFS/SDWFS_full_field_object_mask.fits.gz',
             data=new_mask, header=new_mask_hdr, overwrite=True)
