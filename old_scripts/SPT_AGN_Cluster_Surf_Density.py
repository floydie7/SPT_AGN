"""
SPT_AGN_Cluster_Surf_Density.py
Author: Benjamin Floyd

This script calculates the median AGN surface density of the SPT clusters.
"""

from __future__ import print_function, division
from astropy.io import ascii
import numpy as np
from os import listdir

# Get the file names of all the catalogs.
cat_files = ['Data/Output/' + f for f in listdir('Data/Output') if not f.startswith('.')]

# Read in all the catalogs.
SPT_cats = []
for cat in cat_files:
    SPT_cats.append(ascii.read(cat))

# Determine the mean number of objects in the catalogs. Then the mean number of completeness corrected objects.
num_obj = []
comp_corr_obj = []
area = []
for cat in SPT_cats:
    num_obj.append(len(cat))
    comp_corr_obj.append(np.sum(cat['completeness_correction']))
    area.append(cat['IMAGE_AREA'][0])

# Preform all calculations
mean_num_objs = np.mean(num_obj)
mean_comp_corr = np.median(comp_corr_obj)
surf_dens = [objs / img_area for objs, img_area in zip(comp_corr_obj, area)]
mean_surf_dens = np.median(surf_dens)
stdev_surf_dens = np.std(surf_dens)

print(
 '''Number of catalogs: {num_cat}
Mean number of objects selected in all SDWFS cutouts: {num_obj:.3f}
Mean number of completeness corrected objects: {num_corr:.3f}
Mean number of AGN per square arcminute: {surf_den:.3f}
Standard deviation of AGN per square arcminute: {surf_den_std:.4f}'''
 .format(num_cat=len(SPT_cats), num_obj=mean_num_objs, num_corr=mean_comp_corr,
         surf_den=mean_surf_dens, surf_den_std=stdev_surf_dens))
