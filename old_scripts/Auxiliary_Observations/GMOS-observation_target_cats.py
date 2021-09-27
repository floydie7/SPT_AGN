"""
GMOS-observation_target_cats.py
Author: Benjamin Floyd

A simple code to collate all the AGN catalogs for clusters being observed by Becky Canning on Gemini GMOS-MOS.
"""

from astropy.io import ascii
from astropy.table import Table, vstack
import astropy.units as u

# Read in all the catalogs.
spt_0000 = ascii.read('Data/Output/SPT-CLJ0000-5748_AGN.cat')
spt_0102 = ascii.read('Data/Output/SPT-CLJ0102-4603_AGN.cat')
spt_0142 = ascii.read('Data/Output/SPT-CLJ0142-5032_AGN.cat')
spt_0310 = ascii.read('Data/Output/SPT-CLJ0310-4647_AGN.cat')
spt_0324 = ascii.read('Data/Output/SPT-CLJ0324-6236_AGN.cat')
spt_0528 = ascii.read('Data/Output/SPT-CLJ0528-5300_AGN.cat')
spt_2258 = ascii.read('Data/Output/SPT-CLJ2258-4044_AGN.cat')
spt_2301 = ascii.read('Data/Output/SPT-CLJ2301-4023_AGN.cat')
spt_2337 = ascii.read('Data/Output/SPT-CLJ2337-5942_AGN.cat')
spt_2359 = ascii.read('Data/Output/SPT-CLJ2359-5009_AGN.cat')

# Join all the catalogs together
target_cat = vstack([spt_0000, spt_0102, spt_0142, spt_0310, spt_0324, spt_0528, spt_2258, spt_2301, spt_2337, spt_2359])

# Convert the radial distance column to arcmin (currently in arcsec 20170626)
target_cat['rad_dist'] = target_cat['rad_dist'] / 60.

# Add a [3.6] - [4.5] color column
target_cat['I1-I2_COLOR_APER4'] = target_cat['I1_MAG_APER4'] - target_cat['I2_MAG_APER4']

# Rename columns
target_cat.rename_column('ALPHA_J2000', 'RA')
target_cat.rename_column('DELTA_J2000', 'DEC')
target_cat.rename_column('rad_dist', 'RADIAL_DIST_ARCMIN')

# Sort the table
target_cat = target_cat.group_by('SPT_ID')

for group in target_cat.groups:
    group.sort('I1-I2_COLOR_APER4')
    group.reverse()

# Write the table to disk
ascii.write(target_cat['SPT_ID', 'RA', 'DEC', 'RADIAL_DIST_ARCMIN', 'I1_MAG_APER4', 'I1-I2_COLOR_APER4'],
            'Data/SPT_AGN_GMOS_target_list.cat')
