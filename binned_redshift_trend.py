"""
binned_redshift_trend.py
Author: Benjamin Floyd

Attempts to isolate and qualitatively analyze the redshift trend present in the SPTcl-IRAGN sample.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial.distance import cdist

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def total_area(mask_name, center, radius, redshift):
    # Read in the mask and get the WCS
    image, header = fits.getdata(mask_name, header=True)
    wcs = WCS(header)

    # From the WCS get the pixel scale
    try:
        assert wcs.pixel_scale_matrix[0, 1] == 0.
        pix_scale = wcs.pixel_scale_matrix[1, 1] * wcs.wcs.cunit[1]
    except AssertionError:
        # The pixel scale matrix is not diagonal. We need to diagonalize first
        cd = wcs.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        pix_scale = cd_diag[1, 1] * wcs.wcs.cunit[1]

    # Convert the center coordinate to pixel units
    center_pix = wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

    # Convert the radius to pixel units and get the maximum radius
    radius_pix = (radius * cosmo.arcsec_per_kpc_proper(redshift).to(pix_scale.unit / u.Mpc) / pix_scale).value
    max_radius = np.max(radius_pix)

    # Find the width needed to pad the image to the maximum radius
    width = ((np.round(max_radius - center_pix[1]).astype(int),
              np.round(max_radius - (image.shape[0] - center_pix[1])).astype(int)),
             (np.round(max_radius - center_pix[0]).astype(int),
              np.round(max_radius - (image.shape[1] - center_pix[0])).astype(int)))

    # Insure that we are adding a non-negative padding width
    width = tuple(tuple(i if i >= 0 else 0 for i in axis) for axis in width)

    # Pad the image to the maximum radius
    large_image = np.pad(image, pad_width=width, mode='constant', constant_values=0)

    # Generate a list of all pixel coordinates in the padded image
    image_coords = np.dstack(np.mgrid[0:large_image.shape[0], 0:large_image.shape[1]]).reshape(-1, 2)

    # Transform the center pixel coordinate to the large image coordinate system
    center_coord = np.reshape(np.array(center_pix) + np.array([width[1][0], width[0][0]]), (1, 2))

    # Compute the distance matrix
    image_dists = cdist(image_coords, np.flip(center_coord)).reshape(large_image.shape)

    # Add the number of pixels inside the radius
    pixel_area = np.count_nonzero(large_image[image_dists <= max_radius])

    # Convert the pixel area to physical area
    angular_area = pixel_area * pix_scale**2
    physical_area = angular_area * cosmo.kpc_proper_per_arcmin(redshift).to(u.Mpc / pix_scale.unit)**2

    return angular_area


r500_radius_factor = 1
bin_width = 0.1

# Read in the catalog and group by cluster
sptcl_iragn = Table.read('Data/Output/SPTcl_IRAGN.fits').group_by('SPT_ID')

stacked_catalogs = Table(names=['REDSHIFT', 'M500', 'SURFACE_DENSITY'])
for cluster in sptcl_iragn.groups:
    mask_filename = cluster['MASK_NAME'][0]
    sz_center = cluster['SZ_RA', 'SZ_DEC'][0]
    cluster_z = cluster['REDSHIFT'][0]
    cluster_r500 = cluster['R500'][0] * u.Mpc
    cluster_m500 = cluster['M500'][0]

    # We will find surface densities within this radius
    r500_radius = r500_radius_factor * cluster_r500

    # Compute the cluster area and convert to r500 units
    cluster_area = total_area(mask_filename, sz_center, r500_radius, cluster_z).to(u.arcmin**2)

    # Select AGN within our radius
    cluster = cluster[cluster['RADIAL_SEP_R500'] <= r500_radius_factor]

    cluster_surface_density = cluster['COMPLETENESS_CORRECTION'].sum() / cluster_area - (0.371 / u.arcmin**2)
    cluster_surface_density /= cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc / u.arcmin)**2

    # Append to our list
    stacked_catalogs.add_row([cluster_z, cluster_m500, cluster_surface_density])

# Set the histogram bins
bins = np.arange(stacked_catalogs['REDSHIFT'].min(), stacked_catalogs['REDSHIFT'].max() + bin_width, bin_width)
bin_centers = bins[:-1] + np.diff(bins) / 2

hist_all, _ = np.histogram(stacked_catalogs['REDSHIFT'], bins=bins, weights=stacked_catalogs['SURFACE_DENSITY'])
num_clusters_per_bin_all = np.bincount(np.digitize(stacked_catalogs['REDSHIFT'], bins=bins))[1:]

hist_all /= num_clusters_per_bin_all

# Create the plot
fig, ax = plt.subplots()
ax.bar(bin_centers, hist_all, width=np.diff(bins))
ax.set(title='All SPTcl-IRAGN', xlabel='Redshift',
       ylabel=rf'$\Sigma_{{\rm AGN}}(\leq {r500_radius_factor} R_{{500}}) [R_{{500}}^{{-2}}$ per cluster]')
plt.show()
fig.savefig(f'Data/Plots/SPTcl-IRAGN_surfden_redshift_{r500_radius_factor}r500_bkg_subtracted.pdf')

# Narrow the cluster selection to only clusters with masses in 3-4e14 Msun
stacked_catalogs_narrow_mass = stacked_catalogs[np.abs(stacked_catalogs['M500'] - 3.5e14) <= 0.5e14]

# Set the histogram bins
# bins = np.arange(stacked_catalogs_narrow_mass['REDSHIFT'].min(),
#                  stacked_catalogs_narrow_mass['REDSHIFT'].max() + bin_width, bin_width)
# bin_centers = bins[:-1] + np.diff(bins) / 2

hist_narrow_mass, _ = np.histogram(stacked_catalogs_narrow_mass['REDSHIFT'], bins=bins,
                                   weights=stacked_catalogs_narrow_mass['SURFACE_DENSITY'])
num_clusters_per_bin_narrow_mass = np.bincount(np.digitize(stacked_catalogs_narrow_mass['REDSHIFT'], bins=bins))[1:]

hist_narrow_mass /= num_clusters_per_bin_narrow_mass

# Create the plot
fig, ax = plt.subplots()
ax.bar(bin_centers, hist_narrow_mass, width=np.diff(bins))
ax.set(title=r'(3-4)$\times 10^{14} M_\odot$ SPTcl-IRAGN', xlabel='Redshift',
       ylabel=rf'$\Sigma_{{\rm AGN}}(\leq {r500_radius_factor} R_{{500}}) [R_{{500}}^{{-2}}$ per cluster]')
plt.show()
fig.savefig(f'Data/Plots/SPTcl-IRAGN_surfden_redshift_{r500_radius_factor}r500_M500_3-4e14_bkg_subtracted.pdf')
