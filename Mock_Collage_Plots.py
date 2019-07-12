"""
Mock_Collage_Plots.py
Author: Benjamin Floyd

Checking all mock clusters to verify that the maximum integrating radius is contained on image for GPF testing purposes.
"""
import astropy.units as u
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.patches import Circle, Rectangle

cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

catalog_dir = 'Data/MCMC/Mock_Catalog/Catalogs/pre-final_tests/refresh/'
catalog_name = glob.glob(catalog_dir+'*.cat')

for catalog in catalog_name:
    catalog_file = catalog
    # Read in catalog as we would in the main script
    mock_catalog = Table.read(catalog_file, format='ascii')

    # Group the catalog by cluster ID
    mock_catalog_grp = mock_catalog.group_by('SPT_ID')

    # Read in the mask files for each cluster
    mask_dict = {cluster_id[0]: fits.getdata(mask_file, header=True) for cluster_id, mask_file in
                 zip(mock_catalog_grp.groups.keys.as_array(),
                     mock_catalog_grp['MASK_NAME'][mock_catalog_grp.groups.indices[:-1]])}

    # For our collage
    fig, axarr = plt.subplots(nrows=15, ncols=13, figsize=(50, 50))

    # Cycle through each cluster and create the catalog dictionary used in the probability functions
    catalog_dict = {}
    for cluster, ax in zip(mock_catalog_grp.groups, axarr.flatten()):
        cluster_id = cluster['SPT_ID'][0]
        cluster_z = cluster['REDSHIFT'][0]
        cluster_m500 = cluster['M500'][0] * u.Msun
        cluster_r500 = cluster['r500'][0] * u.Mpc
        cluster_sz_cent = cluster['SZ_RA', 'SZ_DEC'][0]
        cluster_sz_cent = cluster_sz_cent.as_void()
        cluster_radial_r500 = cluster['radial_r500']

        # Determine the maximum radius we can integrate to while remaining completely on image
        mask_image, mask_header = mask_dict[cluster_id]
        mask_wcs = WCS(mask_header)
        pix_scale = mask_wcs.pixel_scale_matrix[1, 1] * u.deg
        cluster_sz_cent_pix = mask_wcs.wcs_world2pix(cluster_sz_cent['SZ_RA'], cluster_sz_cent['SZ_DEC'], 0)

        max_radius_pix = np.min([cluster_sz_cent_pix[0],
                                 cluster_sz_cent_pix[1],
                                 np.abs(cluster_sz_cent_pix[0] - mask_wcs.pixel_shape[0]),
                                 np.abs(cluster_sz_cent_pix[1] - mask_wcs.pixel_shape[1])])
        max_radius_r500 = max_radius_pix * pix_scale * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc / u.deg) / cluster_r500

        # Generate a radial integration mesh
        rall = np.logspace(-2, np.log10(max_radius_r500.value), num=15)

        # Store all relevant information into the master catalog dictionary
        catalog_dict[cluster_id] = {'redshift': cluster_z,
                                    'm500': cluster_m500,
                                    'r500': cluster_r500,
                                    'radial_r500': cluster_radial_r500,
                                    'rall': rall}

        cluster_agn = cluster[np.where(cluster['Cluster_AGN'].astype(bool))]
        background_agn = cluster[np.where(~cluster['Cluster_AGN'].astype(bool))]

        # The following lines are not contained within the standard script but are for the diagnostic
        if 'no_mask' not in catalog:
            ax.imshow(mask_image, origin='lower', cmap='gray' if 'no_masks' in cluster['MASK_NAME'][0] else 'gray_r')
        else:
            ax.set_aspect('equal')
            ax.add_artist(Rectangle((0., 0.,), width=mask_wcs.pixel_shape[0], height=mask_wcs.pixel_shape[1],
                                    edgecolor='k', facecolor='none'))
        ax.scatter(cluster_sz_cent_pix[0], cluster_sz_cent_pix[1], marker='+', color='r', label='SZ Center')
        ax.scatter(cluster_agn['x_pixel'], cluster_agn['y_pixel'], marker='o', edgecolor='cyan', facecolor='none', label='AGN')
        ax.scatter(background_agn['x_pixel'], background_agn['y_pixel'], marker='o', edgecolor='red', facecolor='none', label='Background')
        ax.add_artist(Circle(cluster_sz_cent_pix, radius=max_radius_pix, edgecolor='r', facecolor='none'))
        ax.set(title='{}'.format(cluster_id))
    plt.tight_layout()
    fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/gpf-masking_check_collages/'
                '{}_mask_collage_refresh.pdf'.format(re.search('seed890_(.+?)_mask', catalog_file).group(1)),
                format='pdf')
    # plt.show()
