"""
Saturation_cloud.py
Author: Benjamin Floyd

Just exploring the strange cloud in the FWHM v Mag plot in the SPT channel 2 images.
"""

from __future__ import print_function, division
from astropy.io import ascii
import numpy as np
from ds9_region import ds9_coord

catalog = ascii.read('Data/Saturation/ch2_cloud_objects.cat')
catalog_grouped = catalog.group_by('SPT_ID')

for key, group in zip(catalog_grouped.groups.keys, catalog_grouped.groups):
    spt_id = key[0]
    ds9_coord(group['ALPHA_J2000', 'DELTA_J2000'],
              'Data/Saturation/objects_reg/{id}_cloud_objects.reg'.format(id=spt_id), 'green', 4., 'fk5', 'circle', 1,
              ra_col='ALPHA_J2000', dec_col='DELTA_J2000')
