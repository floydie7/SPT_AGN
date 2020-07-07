"""
Example_mock_cluster_image.py
Author: Benjamin Floyd

Generates an image of a mock cluster for presentations.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

# Read in the catalog
mock_catalog = Table.read('Data/MCMC/Mock_Catalog/Catalogs/Final_tests/Slope_tests/trial_5/'
                          'mock_AGN_catalog_t0.462_e4.00_z-1.00_b1.00_C0.371_rc0.100_maxr5.00_clseed890_objseed930_slope_test.cat',
                          format='ascii')

# Find the richest cluster
mock_catalog_grp = mock_catalog.group_by('SPT_ID')
richness = [np.count_nonzero(cluster['Cluster_AGN'].astype(bool)) for cluster in mock_catalog_grp.groups]
richest_cluster = mock_catalog_grp.groups[np.argmax(richness)]

# Split the cluster AGN from the background
cluster_agn = richest_cluster[richest_cluster['Cluster_AGN'].astype(bool)]
background = richest_cluster[~richest_cluster['Cluster_AGN'].astype(bool)]

# Read in the cluster's mask
mask_img, mask_hdr = fits.getdata(richest_cluster['MASK_NAME'][0], header=True)

#%% Make the plot
fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True, subplot_kw=dict(projection=WCS(mask_hdr)))
ax.imshow(mask_img, origin='lower', cmap='Greys')
ax.plot(richest_cluster['SZ_RA'][0], richest_cluster['SZ_DEC'][0], marker='+', color='magenta', markersize=10,
        linestyle='None', transform=ax.get_transform('world'), label='Cluster Center')
ax.scatter(cluster_agn['RA'], cluster_agn['DEC'], transform=ax.get_transform('world'), c='b', label='Cluster AGN')
ax.scatter(background['RA'], background['DEC'], transform=ax.get_transform('world'), c='r', label='Background Galaxies')
ax.set(xlabel='Right Ascension', ylabel='Declination')
ax.legend(frameon=False)
plt.show()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Final_tests/Slope_tests/trial_5/'
            f'Slope_test_trial_5_{richest_cluster["SPT_ID"][0]}_example.png')