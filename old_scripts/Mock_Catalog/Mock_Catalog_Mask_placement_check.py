"""
Mock_Catalog_Mask_placement_check.py
Author: Benjamin Floyd

Produces images for every cluster in the sample using the cluster's mask image and overplotting the objects in the
catalog.
"""
import re
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import matplotlib

matplotlib.rcParams['figure.max_open_warning'] = 0

# Read in the catalog
mock_catalog = Table.read('Data/MCMC/Mock_Catalog/Catalogs/pre-final_tests/'
                          'mock_AGN_catalog_t12.00_e1.20_z-1.00_b0.50_C0.371_maxr5.00_seed890_gpf_fixed_multicluster.cat', format='ascii')

for cluster in mock_catalog.group_by('SPT_ID').groups:
    mask_cluster_id = re.search('SPT-CLJ(.+?)_', cluster['MASK_NAME'][0]).group(0)[:-1]
    mask_image, mask_header = fits.getdata(cluster['MASK_NAME'][0], header=True)
    # Convert the SZ center coordinate into pixel coordinates
    w = WCS(mask_header)
    sz_center_pix = w.wcs_world2pix(cluster['SZ_RA'][0], cluster['SZ_DEC'][0], 0)

    # Separate the cluster objects from the background objects
    cluster_agn = cluster[np.where(cluster['Cluster_AGN'].astype(bool))]
    background_agn = cluster[np.where(~cluster['Cluster_AGN'].astype(bool))]

    # Create the plot
    fig, ax = plt.subplots(subplot_kw={'projection': w})
    ax.imshow(mask_image, origin='lower', cmap='gray_r')
    ax.plot(sz_center_pix[0], sz_center_pix[1], 'w+', markersize=10)
    ax.scatter(cluster_agn['x_pixel'], cluster_agn['y_pixel'],
               edgecolor='cyan', facecolor='none', alpha=1, label='Cluster_AGN')
    ax.scatter(background_agn['x_pixel'], background_agn['y_pixel'],
               edgecolor='red', facecolor='none', alpha=1, label='Background AGN')
    ax.coords[0].set_major_formatter('hh:mm:ss.s')
    ax.coords[1].set_major_formatter('dd:mm:ss')
    ax.set(title='AGN Placement for {mock_id}\nUsing Mask from {mask_id}'.format(mock_id=cluster['SPT_ID'][0],
                                                                                 mask_id=mask_cluster_id),
           xlabel='Right Ascension', ylabel='Declination')
    ax.legend(handletextpad=0.001)
    fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/mask_placement_tests/gpf_fixed_multicluster/'
                '{}.pdf'.format(cluster['SPT_ID'][0]), format='pdf')
