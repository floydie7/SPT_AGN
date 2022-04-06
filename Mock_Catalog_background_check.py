"""
Mock_Catalog_background_check.py
Author: Benjamin Floyd

Does a "by-hand" check of the distribution of background galaxies in the mock catalog.
"""
import glob
import re

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.stats import describe
import matplotlib.pyplot as plt

# Read in the mock catalog
# mock_catalog = Table.read('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Final_tests/LF_tests/'
#                           'variable_theta/flagged_versions/mock_AGN_catalog_t2.600_e4.00_z-1.00_b1.00_rc0.100_C0.333'
#                           '_maxr5.00_clseed890_objseed930_semiempirical.fits')
# mock_catalog = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')
catalog_list = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Final_tests/LF_tests/'
                         'variable_theta/flagged_versions/*t2.600*semiempirical*')

for catalog in catalog_list:
    clseed, objseed = re.search(r'clseed(\d+)_objseed(\d+)', catalog).group(1, 2)
    print(f'cluster seed: {clseed}\tobject seed: {objseed}')
    mock_catalog = Table.read(catalog)
    surface_den = []
    for cluster in mock_catalog.group_by('SPT_ID').groups:
        # Get the mask file
        mask_img, mask_hdr = fits.getdata(cluster['MASK_NAME'][0], header=True)

        # Get the WCS and from that, the pixel scale
        mask_wcs = WCS(mask_hdr)
        pix_scale = mask_wcs.proj_plane_pixel_scales()[0]

        # Isolate background galaxies
        background = cluster[~cluster['CLUSTER_AGN'].astype(bool)]

        # Compute the area in the image (in arcmin^2)
        area = (mask_img.sum() * pix_scale * pix_scale).to_value(u.arcmin**2)

        # Get the weighted number of objects
        num_of_gals = np.sum(background['COMPLETENESS_CORRECTION'] * background['SELECTION_MEMBERSHIP'])
        # num_of_gals = np.sum(cluster['COMPLETENESS_CORRECTION'] * cluster['SELECTION_MEMBERSHIP'])

        # Store the surface density
        surface_den.append(num_of_gals / area)

    # Find the mean and standard deviation of the surface density
    mean_surf_den = np.mean(surface_den)
    std_surf_den = np.std(surface_den)
    print(f'{mean_surf_den = :.3f} +- {std_surf_den = :.4f}')
    print(describe(surface_den))

    q = np.quantile(surface_den, [0.16, 0.5, 0.84])
    print(q)

    bins = np.arange(0.0, 0.8, 0.05)
    fig, ax = plt.subplots()
    ax.hist(surface_den, bins=bins, histtype='step', color='k')
    ax.axvline(x=mean_surf_den, color='k', ls='--', label=r'Mean $C$')
    # ax.axvline(x=q[1], color='k', ls='--', label=r'Median $C$')
    ax.axvline(x=q[0], color='k', ls='--', alpha=0.4)
    ax.axvline(x=q[2], color='k', ls='--', alpha=0.4)
    ax.axvline(x=0.333, color='tab:blue', label=r'Input $C$')
    # ax.axvline(x=0.296, color='tab:red', label='MCMC fit')
    ax.legend()
    ax.set(xlabel=r'$C$ [arcmin$^{-2}$]', ylabel='N')
    # fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/mock_rework/'
    #             'background_check_t2.600_e4.00_z-1.00_b1.00_rc0.100_C0.333_semiempirical.pdf')
    plt.show()
