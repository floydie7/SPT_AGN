"""
SPT_AGN_Plots.py
author: Benjamin Floyd
A script to generate any plots needed to visualize the data of the SPT_AGN data
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table
from os import listdir

catalogs = []

# Read in all the AGN catalogs
for cat in listdir('Data/Output'):
    catalogs.append(ascii.read('Data/Output/'+cat))

# Read in Bleem catalog
bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))

# print(bleem[np.where(bleem['REDSHIFT'] == 0.0)]['SPT_ID'])

# Redshift range for SPT clusters
fig, ax = plt.subplots()
ax.hist(bleem[np.where(bleem['REDSHIFT'] != 0.0)]['REDSHIFT'], bins=np.arange(0.1, 2.0, 0.1), rwidth=0.9, color='#3F5D7D', linewidth=0)
ax.set(title='SPT Clusters', xlabel='Redshift', ylabel='Number of Clusters')
plt.show()

# Redshift range for AGN
z_range = []
z_0 = 0
for i in range(len(catalogs)):
    z_range.append(catalogs[i]['REDSHIFT'][0])
    if catalogs[i]['REDSHIFT'][0] == 0.0:
        z_0 += 1

print(z_0)

fig, ax = plt.subplots()
ax.hist(z_range, bins=np.arange(0.0, 1.8, 0.1), rwidth=0.9, color='#3F5D7D', linewidth=0)
ax.set(title='IR-Selected AGN in SPT', xlabel='Redshift', ylabel='Incidence')
fig.savefig('SPT_AGN_redshift.pdf', format='pdf')
plt.show()
