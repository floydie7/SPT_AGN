"""
SPTcl_mass_radial_freq_analysis.py
Author: Benjamin Floyd

Recreates the old frequentist analysis using the updated dataset SPTcl (SPT-SZ + SPTpol 100d) for both the mass and
radial trends binned by redshift.
"""

import glob
import re

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import astropy.units as u
from scipy import stats
from astro_compendium.utils.small_poisson import small_poisson
from astropy.cosmology import FlatLambdaCDM

SDWFS_SURF_DEN = 0.333 / u.arcmin ** 2
SDWFS_SURF_DEN_ERR = 0.024 / u.arcmin ** 2

cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

# Read in the catalogs
sptcl = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

clusters = []
for cluster in sptcl.group_by('SPT_ID').groups:
    # Read in the pixel mask and WCS
    mask_name = cluster['MASK_NAME'][0]
    mask_img, mask_hdr = fits.getdata(mask_name, header=True)
    mask_wcs = WCS(mask_hdr)

    # Using the WCS, compute the pixel scale
    try:
        assert mask_wcs.pixel_scale_matrix[0, 1] == 0.
        mask_pixel_scale = mask_wcs.pixel_scale_matrix[1, 1] * mask_wcs.wcs.cunit[1]
    except AssertionError:
        cd = mask_wcs.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        mask_pixel_scale = cd_diag[1, 1] * mask_wcs.wcs.cunit[1]

    # Compute the area of the image
    mask_area = mask_img.sum() * mask_pixel_scale.to(u.arcmin) ** 2

    # Add the area calculation to the catalog
    cluster['IMAGE_AREA'] = mask_area
    clusters.append(cluster)

# Re-merge the catalog
sptcl = vstack(clusters)

# Add a log-mass column and isolate the cluster masses
sptcl['logM500'] = np.log10(sptcl['M500'])
cluster_log_mass = np.array([cluster['logM500'][0] for cluster in sptcl.group_by('SPT_ID').groups])

# Set the three binnings
z_bin_edges = np.array([np.median(sptcl['REDSHIFT']), 1.4, sptcl['REDSHIFT'].max()])
logM_bin_edges = stats.mstats.mquantiles(cluster_log_mass, [0., 0.25, 0.5, 0.75, 1.])
radial_bin_edges = stats.mstats.mquantiles(sptcl['RADIAL_SEP_R500'], [0., 0.25, 0.5, 0.75, 1.])

# Bin the clusters by redshift
z_binned_clusters = sptcl.group_by(np.digitize(sptcl['REDSHIFT'], bins=z_bin_edges))

# Iterate through the redshift bins
for cluster_z_grp in z_binned_clusters.groups:
    # Group the catalog according to the mass bins
    cluster_z_m_clusters = [cluster_z_grp[(logM_bin_edges[i] <= cluster_z_grp['logM500']) &
                                          (cluster_z_grp['logM500'] <= logM_bin_edges[i + 1])]
                            for i in range(len(logM_bin_edges) - 1)]

    # Iterate through the mass bins
    m_bin_surf_den, m_bin_surf_den_err = [], []
    for cluster_z_m_grp in cluster_z_m_clusters:
        surf_den, surf_den_err = [], []
        if len(cluster_z_m_grp) == 0:
            m_bin_surf_den.append(np.nan)
            m_bin_surf_den_err.append(np.nan)
        else:
            for cluster in cluster_z_m_grp.group_by('SPT_ID').groups:
                # Compute the (corrected) total number of AGN candidates along the line-of-sight to each cluster
                los_counts = np.sum(cluster['COMPLETENESS_CORRECTION'] * cluster['SELECTION_MEMBERSHIP'])

                # Compute the field expectation for this line-of-sight
                field_counts = SDWFS_SURF_DEN * cluster['IMAGE_AREA'].quantity[0]

                # Subtract the field counts from the line-of-sight
                cluster_counts = los_counts - field_counts

                # Compute the surface density of the cluster excess objects
                cluster_surf_den = cluster_counts / cluster['IMAGE_AREA'].quantity[0]

                # For the errors on our measurement, compute the Poisson errors of the line-of-sight and field counts
                los_counts_uerr, los_counts_lerr = small_poisson(los_counts)
                field_counts_uerr, field_counts_lerr = small_poisson(field_counts)

                # Propagate the errors
                cluster_counts_uerr = np.sqrt(los_counts_uerr ** 2 + field_counts_uerr ** 2)
                cluster_counts_lerr = np.sqrt(los_counts_lerr ** 2 + field_counts_lerr ** 2)

                # Compute the surface density errors
                cluster_surf_den_uerr = cluster_counts_uerr / cluster['IMAGE_AREA'].quantity[0]
                cluster_surf_den_lerr = cluster_counts_lerr / cluster['IMAGE_AREA'].quantity[0]

                # Convert our cluster counts (and errors) from angular to physical units
                cluster_surf_den_mpc2 = cluster_surf_den / cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0]) \
                    .to(u.Mpc / u.arcmin) ** 2
                cluster_surf_den_uerr_mpc2 = cluster_surf_den_uerr / cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0]) \
                    .to(u.Mpc / u.arcmin) ** 2
                cluster_surf_den_lerr_mpc2 = cluster_surf_den_lerr / cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0]) \
                    .to(u.Mpc / u.arcmin) ** 2

                # Accumulate the surface densities and errors (lower, upper)
                surf_den.append(cluster_surf_den_mpc2)
                surf_den_err.append((cluster_surf_den_lerr_mpc2.value, cluster_surf_den_uerr_mpc2.value))

            # Compute the mean surface density for the mass bin
            bin_mean_surf_den = np.nanmean(u.Quantity(surf_den))

            # Combine the surface density errors in quadrature divided by the number of clusters contributing to the bin
            bin_surf_den_err = np.sqrt(np.nansum(np.power(surf_den_err, 2), axis=0)) / len(surf_den_err) / u.Mpc ** 2

            # Accumulate the bin surface density and errors
            m_bin_surf_den.append(bin_mean_surf_den)
            m_bin_surf_den_err.append(bin_surf_den_err)


