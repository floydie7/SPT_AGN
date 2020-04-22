"""
Mass-Redshift_richness_plot.py
Author: Benjamin Floyd

Generates a Cluster Mass-Redshift-AGN Richness plot for the SPTcl-IRAGN sample
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial.distance import cdist


def image_area(radius, redshift, center, mask_name):
    # Read in the mask
    image, header = fits.getdata(mask_name, header=True)
    image_wcs = WCS(header)

    # From the WCS get the pixel scale
    try:
        assert image_wcs.pixel_scale_matrix[0, 1] == 0.
        pix_scale = image_wcs.pixel_scale_matrix[1, 1] * image_wcs.wcs.cunit[1]
    except AssertionError:
        # The pixel scale matrix is not diagonal. We need to diagonalize first
        cd = image_wcs.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        pix_scale = cd_diag[1, 1] * image_wcs.wcs.cunit[1]

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

    # Convert our radius to pixels
    r_pix = (radius * cosmo.arcsec_per_kpc_proper(redshift).to(u.arcmin / u.Mpc) / pix_scale).value

    # Because for low redshift the r500 radius may be larger than the image we will pad the image to be uniform in r500
    # Find the width needed to pad the image to include the largest radius inside the image.
    width = ((int(round(r_pix - center_pix[1])),
              int(round(r_pix - (image.shape[0] - center_pix[1])))),
             (int(round(r_pix - center_pix[0])),
              int(round(r_pix - (image.shape[1] - center_pix[0])))))

    # Insure that we are adding a non-negative padding width.
    width = tuple(tuple([i if i >= 0 else 0 for i in axis]) for axis in width)

    large_image = np.pad(image, pad_width=width, mode='constant', constant_values=0)

    # Generate a list of all pixel coordinates in the padded image
    image_coords = np.dstack(np.mgrid[0:large_image.shape[0], 0:large_image.shape[1]]).reshape(-1, 2)

    # The center pixel's coordinate needs to be transformed into the large image system
    center_coord = np.array(center_pix) + np.array([width[1][0], width[0][0]])
    center_coord = center_coord.reshape((1, 2))

    # Compute the distance matrix. The entries are a_ij = sqrt((x_j - cent_x)^2 + (y_i - cent_y)^2)
    image_dists = cdist(image_coords, np.flip(center_coord)).reshape(large_image.shape)

    # Select all pixels within our radius
    large_image = large_image[image_dists <= r_pix]

    # Find the total area
    total_area = np.sum(large_image) * pix_scale**2

    return total_area.to(u.arcmin**2)


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

SDWFS_AGN_surface_density = 0.371 * u.arcmin**-2

SPTcl_AGN = Table.read('Data/Output/SPTcl_IRAGN.fits')
SPTcl_AGN_grp = SPTcl_AGN.group_by('SPT_ID')

# Compute the angular area available
cluster_redshift = []
cluster_mass = []
AGN_richness = []
for cluster in SPTcl_AGN_grp.groups:
    r500 = cluster['R500'][0] * u.Mpc
    z = cluster['REDSHIFT'][0]
    m500 = cluster['M500'][0]
    sz_center = cluster['SZ_RA', 'SZ_DEC'][0]
    mask_filename = cluster['MASK_NAME'][0]

    # Find the usable area of the cluster
    angular_area = image_area(r500, z, sz_center, mask_filename)

    # Find the expected field level within r500
    n_field = angular_area * SDWFS_AGN_surface_density

    # Select AGN within r500
    agn_in_r500 = cluster[cluster['RADIAL_SEP_R500'] <= 1]

    # Field correct the AGN richness within r500
    cluster_richness = len(agn_in_r500) - n_field.value

    # Collect the redshift, mass, and richness
    cluster_redshift.append(z)
    cluster_mass.append(m500)
    AGN_richness.append(cluster_richness)

#%%
fig, ax = plt.subplots()
cm = ax.scatter(cluster_redshift, cluster_mass, c=AGN_richness, cmap='gist_heat_r')
plt.colorbar(cm)
plt.show()
