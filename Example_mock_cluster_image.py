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
from matplotlib.lines import Line2D

# mock_catalog = Table.read('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Final_tests/'
#                           'fuzzy_selection/mock_AGN_catalog_t2.500_e1.20_z-1.00_b1.00_C0.371_rc0.100_maxr5.00'
#                           '_clseed890_objseed930_fuzzy_selection.fits')
mock_catalog = Table.read('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Final_tests/LF_tests/'
                          'variable_theta/flagged_versions/mock_AGN_catalog_t2.500_e4.00_z-1.00_b1.00_rc0.100_C0.333'
                          '_maxr5.00_clseed890_objseed930_photometry_weighted_kde_rejection_flag.fits')
mock_catalog = mock_catalog[mock_catalog['COMPLETENESS_REJECT'].astype(bool)]

# Find the richest cluster
mock_catalog_grp = mock_catalog.group_by('SPT_ID')
richness = [np.count_nonzero(cluster['CLUSTER_AGN'].astype(bool)) for cluster in mock_catalog_grp.groups]
# richest_cluster = mock_catalog_grp.groups[np.argmax(richness)]
idx = np.random.randint(0, len(richness)-1)  # Using idx = 50 for use
richest_cluster = mock_catalog_grp.groups[np.argpartition(richness, -1)[50]]  # This uses a nicer looking mask

# Split the cluster AGN from the background
cluster_agn = richest_cluster[richest_cluster['CLUSTER_AGN'].astype(bool)]
background = richest_cluster[~richest_cluster['CLUSTER_AGN'].astype(bool)]

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
ax.legend(frameon=True)
plt.show()
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/'
            f'LF_selection_{richest_cluster["SPT_ID"][0]}_example.png')

#%% Make a version of the plot showing degrees of membership
membership_degree_cluster = cluster_agn['SELECTION_MEMBERSHIP']
membership_degree_background = background['SELECTION_MEMBERSHIP']

fig, ax = plt.subplots(figsize=(9, 8), tight_layout=True, subplot_kw=dict(projection=WCS(mask_hdr)))
ax.imshow(mask_img, origin='lower', cmap='Greys')
ax.plot(richest_cluster['SZ_RA'][0], richest_cluster['SZ_DEC'][0], marker='+', color='magenta', markersize=10,
        linestyle='None', transform=ax.get_transform('world'), label='Cluster Center')
cm = ax.scatter(cluster_agn['RA'], cluster_agn['DEC'], transform=ax.get_transform('world'), c=membership_degree_cluster,
           cmap='Blues', vmin=0.3, vmax=1, label='Cluster AGN')
ax.scatter(background['RA'], background['DEC'], transform=ax.get_transform('world'), c=membership_degree_background,
           cmap='Reds', vmin=0.3, vmax=1, label='Background Galaxies')
ax.set(xlabel='Right Ascension', ylabel='Declination')
handles, labels = ax.get_legend_handles_labels()
handles[1:] = [Line2D([0], [0], marker='o', color='w', markerfacecolor='b'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor='r')]
ax.legend(handles, labels)
cbar = plt.colorbar(cm, shrink=0.9)
cbar.set_label('Degree of Membership')
plt.show()
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/'
            f'LF_selection_{richest_cluster["SPT_ID"][0]}_example_membership.png')
