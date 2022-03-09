"""
Mock_Catalog_background_check.py
Author: Benjamin Floyd

Does a "by-hand" check of the distribution of background galaxies in the mock catalog.
"""

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.stats import describe
import matplotlib.pyplot as plt

# Read in the mock catalog
mock_catalog = Table.read('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Final_tests/LF_tests/'
                          'variable_theta/flagged_versions/mock_AGN_catalog_t2.600_e4.00_z-1.00_b1.00_rc0.100_C0.333'
                          '_maxr5.00_clseed890_objseed930_semiempirical.fits')

surface_den = []
for cluster in mock_catalog.group_by('SPT_ID').groups:
    # Get the mask file
    mask_img, mask_hdr = fits.getdata(cluster['MASK_NAME'][0], header=True)

    # Get the WCS and from that, the pixel scale
    mask_wcs = WCS(mask_hdr)
    try:
        assert mask_wcs.pixel_scale_matrix[0, 1] == 0.
        pix_scale = mask_wcs.pixel_scale_matrix[1, 1] * mask_wcs.wcs.cunit[1]
    except AssertionError:
        # The pixel scale matrix is not diagonal. We need to diagonalize first
        cd = mask_wcs.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        pix_scale = cd_diag[1, 1] * mask_wcs.wcs.cunit[1]

    # Isolate background galaxies
    background = cluster[~cluster['CLUSTER_AGN'].astype(bool)]

    # Compute the area in the image (in arcmin^2)
    area = (mask_img.sum() * pix_scale * pix_scale).to_value(u.arcmin**2)

    # Get the weighted number of objects
    num_of_gals = np.sum(background['COMPLETENESS_CORRECTION'] * background['SELECTION_MEMBERSHIP'])

    # Store the surface density
    surface_den.append(num_of_gals / area)

# Find the mean and standard deviation of the surface density
mean_surf_den = np.mean(surface_den)
std_surf_den = np.std(surface_den)
print(f'{mean_surf_den = :.3f} +- {std_surf_den = :.4f}')
print(describe(surface_den))

plt.hist(surface_den, bins='auto')
plt.show()
