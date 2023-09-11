"""
SDWFS_full_field_mask.py
Author: Benjamin Floyd

This script takes the Optical + IRAC mask that Mark Brodwin provided and converts it into the simpler binary mask that
we use in the IRAGN pipeline analyses.
"""

from astropy.io import fits

# Read in the original mask
orig_mask, hdr = fits.getdata('Data_Repository/Images/Bootes/SDWFS/SDWFS_opt+irac.fits.gz', header=True)

# The original mask has `0` as "good" pixel. Set these to `True` and all others to `False`
new_mask = (orig_mask == 0).astype(int)

# There are a couple of erroneous regions that are marked as good in the corners
new_mask[:790, :1130] = 0
new_mask[:790, 14730:] = 0

# Write the new mask back to disk
fits.writeto('Data_Repository/Images/Bootes/SDWFS/SDWFS_full_field_object_mask.fits.gz',
             data=new_mask, header=hdr)

