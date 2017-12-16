"""
SPT_AGN_Prelim_Sci_Mass_Plots.py
Author: Benjamin Floyd

This script generates the preliminary Mass trend science plots for the SPT AGN study.
"""

from __future__ import print_function, division

from os import listdir

import astropy.units as u
import matplotlib
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from itertools import product
from scipy.spatial.distance import cdist

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Set our cosmology
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

# The field AGN surface density from SDWFS was found to be 0.371 AGN / arcmin^2.
field_surf_den = 0.371 / u.arcmin**2
field_surf_den_err = 0.157 / u.arcmin**2

# Read in the Bleem catalog. We'll need the cluster center coordinates to anchor the radial annuli.
Bleem = Table.read('Data/2500d_cluster_sample_fiducial_cosmology.fits')


def spt_bootstrap(data, bootnum=100, samples=None, bootfunc=None):
    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)

    # create empty boot array
    boot = np.empty(resultdims, dtype=data.dtype)

    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])

    return boot


def small_poisson(n, S=1):
    """
    Calculates the upper and lower Poisson confidence limits for extremely low counts i.e., n << 50. These equations are
    outlined in [Gehrels1986]_.

    .._[Gehrels1986] http://adsabs.harvard.edu/abs/1986ApJ...303..336G

    :param n: The number of Poisson counts.
    :type n: float, array-like
    :param S: The S-sigma Gaussian levels. Defaults to `S=1` sigma.
    :type S: int
    :return: The upper and lower errors corresponding to the confidence levels.
    :rtype: tuple
    """

    # Parameters for the lower limit equation. These are for the 1, 2, and 3-sigma levels.
    beta = [0.0, 0.06, 0.222]
    gamma = [0.0, -2.19, -1.88]

    # Upper confidence level using equation 9 in Gehrels 1986.
    lambda_u = (n + 1.) * (1. - 1. / (9. * (n + 1.)) + S / (3. * np.sqrt(n + 1.)))**3

    # Lower confidence level using equation 14 in Gehrels 1986.
    lambda_l = n * (1. - 1. / (9. * n) - S / (3. * np.sqrt(n)) + beta[S - 1] * n**gamma[S - 1])**3

    # To clear the lower limit array of any possible NaNs from n = 0 incidences.
    np.nan_to_num(lambda_l, copy=False)

    # Calculate the upper and lower errors from the confidence values.
    upper_err = lambda_u - n
    lower_err = n - lambda_l

    return upper_err, lower_err


def make_z_mass_bin_histogram(z_bin, mass_bins, radius):  # TODO change from 2r500 area to the IMAGE_AREA value.

    def annulus_pixel_area(spt_id, _rad_bins):
        # Using the Bleem SPT ID read in the correct mask image.
        mask_filename = 'Data/Masks/{spt_id}_cov_mask4_4.fits'.format(spt_id=spt_id)

        # Read in the mask image.
        mask_image = fits.getdata(mask_filename, ignore_missing_end=True, memmap=False)

        # Read in the WCS from the coverage mask we made earlier.
        w = WCS(mask_filename)

        # Get the pixel scale as well for single value conversions.
        try:
            pix_scale = fits.getval(mask_filename, 'PXSCAL2') * u.arcsec
        except KeyError:  # Just in case the file doesn't have 'PXSCAL2'
            try:
                pix_scale = fits.getval(mask_filename, 'CDELT2') * u.deg
            # If both cases fail report the cluster and the problem
            except KeyError("Header is missing both 'PXSCAL2' and 'CDELT2'. Please check the header of: {file}"
                                    .format(file=mask_filename)):
                raise

        # Convert the cluster center to pixel coordinates.
        cluster_x, cluster_y = w.wcs_world2pix(Bleem['RA'][np.where(Bleem['SPT_ID'] == spt_id)],
                                               Bleem['DEC'][np.where(Bleem['SPT_ID'] == spt_id)], 0)

        # Convert the radial bins from arcmin to pixels.
        rad_bins_pix = _rad_bins / pix_scale.to(u.arcmin)

        # Generate the list of coordinates
        image_coordinates = np.array(list(product(range(fits.getval(mask_filename, 'NAXIS1')),
                                                  range(fits.getval(mask_filename, 'NAXIS2')))))

        # Calculate the distances from the cluster center to all other pixels
        image_distances = cdist(image_coordinates, np.array([[cluster_x[0], cluster_y[0]]])).flatten()

        # Select the coordinates in the annuli
        annuli_coords = [image_coordinates[np.where(image_distances <= rad_bins_pix.value)]]

        # For each annuli query the values of the pixels matching the coordinates found above and count the number of
        # good pixels (those with a value of `1`).
        area_pixels = [np.count_nonzero(mask_image[annulus.T[1], annulus.T[0]]) for annulus in annuli_coords]

        # Convert the pixel areas into arcmin^2 areas.
        area_arcmin2 = area_pixels * pix_scale.to(u.arcmin) * pix_scale.to(u.arcmin)

        return area_arcmin2

    def mass_surf_den(_z_bin, _mass_bins, _radius):
        # Group the catalog by cluster
        z_bin_grouped = _z_bin.group_by('SPT_ID')

        total_mass_surf_den = []
        cluster_field_mass_surf_den_err = []
        for cluster in z_bin_grouped.groups:
            print('Cluster: {}'.format(cluster['SPT_ID'][0]))

            # Convert the fractional radius location into a physical distance
            cluster_radius_mpc = _radius * cluster['r500'][0] * u.Mpc

            # Convert the physical radius to an on-sky radius.
            cluster_radius_arcmin = (cluster_radius_mpc
                                     / cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0]).to(u.Mpc / u.arcmin))

            # Calculate the area inclosed by the radius in arcmin2.
            cluster_area = annulus_pixel_area(cluster['SPT_ID'][0], cluster_radius_arcmin)

            # Select only the AGN within the radius.
            cluster_agn = cluster[np.where(cluster['RADIAL_DIST'] <= cluster_radius_arcmin.value)]

            # Also, using the SDWFS surface density, calculate the expected field AGN counts in the selected area.
            field_agn = field_surf_den * cluster_area

            # Create the histogram, binned by log-mass, weighted by the completeness value.
            cluster_mass_hist, _ = np.histogram(cluster_agn['logM500'],
                                                bins=_mass_bins,
                                                weights=cluster_agn['completeness_correction'])

            # Calculate the Poisson errors for each mass bin. Also calculate the Poisson error for the field expectation
            cluster_poisson_err = small_poisson(cluster_mass_hist, S=1)
            field_poisson_err = small_poisson(field_agn, S=1)

            # Subtract the field expecation from the cluster counts.
            cluster_field_counts = cluster_mass_hist - field_agn

            # Propagate the cluster and field errors together.
            cluster_field_counts_upper_err = np.sqrt(cluster_poisson_err[0]**2 + field_poisson_err[0]**2)
            cluster_field_counts_lower_err = np.sqrt(cluster_poisson_err[0]**2 + field_poisson_err[0]**2)

            # Calculate the surface density for each mass bin
            cluster_field_surf_den = cluster_field_counts / cluster_area

            # Convert the errors to surface densities
            cluster_field_surf_den_upper_err = cluster_field_counts_upper_err / cluster_area
            cluster_field_surf_den_lower_err = cluster_field_counts_lower_err / cluster_area

            # Using the cluster's redshift, convert the surface densities from sky units to physical units.
            cluster_field_surf_den_mpc = cluster_field_surf_den / (cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0])
                                                                   .to(u.Mpc / u.arcmin))**2

            # Also convert the errors to physical units.
            cluster_field_surf_den_upper_err_mpc = (cluster_field_surf_den_upper_err
                                                    / (cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0])
                                                       .to(u.Mpc / u.arcmin))**2)
            cluster_field_surf_den_lower_err_mpc = (cluster_field_surf_den_lower_err
                                                    / (cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0])
                                                       .to(u.Mpc / u.arcmin)) ** 2)
            cluster_field_surf_den_err = (cluster_field_surf_den_upper_err_mpc, cluster_field_surf_den_lower_err_mpc)

            total_mass_surf_den.append(cluster_field_surf_den_mpc)
            cluster_field_mass_surf_den_err.append(cluster_field_surf_den_err)

        return total_mass_surf_den, cluster_field_mass_surf_den_err

    # Calculate the surface densities.
    cluster_mass_surf_den, cluster_mass_surf_den_err = mass_surf_den(z_bin, mass_bins, radius)

    # Compute the average AGN surface density per cluster. As in the radial analysis script we will use the
    # `nanmean` function to avoid issues with any NaN values in the sample.
    z_mass_surf_den = np.nanmean(cluster_mass_surf_den, axis=0)

    # Extract the upper and lower Poisson errors for each cluster.
    cluster_surf_den_upper_err = np.array([error[0] for error in cluster_mass_surf_den_err])
    cluster_surf_den_lower_err = np.array([error[1] for error in cluster_mass_surf_den_err])

    # The errors may have non-finite (inf, neginf, NaN) values due to divisions by zero. However, we never want to
    # include these values in our calculations. Therefore, let us identify all non-finite values and send them to a
    # single value e.g., NaN which we can process with the numpy nan* functions.
    poisson_upper_err_filtered = np.where(np.isfinite(cluster_mass_surf_den), cluster_surf_den_upper_err, np.nan)
    poisson_lower_err_filtered = np.where(np.isfinite(cluster_mass_surf_den), cluster_surf_den_lower_err, np.nan)

    # Combine all errors in quadrature within each mass bin and divide by the number of clusters contributing to the
    # mass bin.
    z_mass_surf_den_upper_err = (np.sqrt(np.nansum(poisson_upper_err_filtered**2, axis=0))
                                 / np.count_nonzero(np.isfinite(cluster_mass_surf_den), axis=0))
    z_mass_surf_den_lower_err = (np.sqrt(np.nansum(poisson_lower_err_filtered ** 2, axis=0))
                                 / np.count_nonzero(np.isfinite(cluster_mass_surf_den), axis=0))

    z_mass_surf_den_err = [z_mass_surf_den_upper_err, z_mass_surf_den_lower_err]

    return z_mass_surf_den, z_mass_surf_den_err


# Read in all the catalogs
AGN_cats = [Table.read('Data/Output/'+f, format='ascii') for f in listdir('Data/Output/') if not f.startswith('.')]

# Convert the radial distance column in the catalogs from arcmin to Mpc.
for cat in AGN_cats:
    cat['RADIAL_DIST'].unit = u.arcmin
    cat['RADIAL_DIST_Mpc'] = (cat['RADIAL_DIST'] * cosmo.kpc_proper_per_arcmin(cat['REDSHIFT'])).to(u.Mpc)

    # Calculate the r500
    cat['M500'].unit = u.Msun
    cat['r500'] = (3 * cat['M500'] /
                   (4 * np.pi * 500 * cosmo.critical_density(cat['REDSHIFT']).to(u.Msun / u.Mpc ** 3)))**(1/3)

    # Calculate the log(M500)
    cat['logM500'] = np.log10(cat['M500'])


# Combine all the catalogs into a single table.
full_AGN_cat = vstack(AGN_cats)

# Set up mass bins
mass_bins = np.array([14.3, 14.4, 14.5, 14.75, 15.1])

# Set our redshift bins
low_z_bin = full_AGN_cat[np.where(full_AGN_cat['REDSHIFT'] <= 0.5)]
mid_low_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.5) & (full_AGN_cat['REDSHIFT'] <= 0.65))]
mid_mid_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.65) & (full_AGN_cat['REDSHIFT'] <= 0.75))]
mid_high_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.75) & (full_AGN_cat['REDSHIFT'] <= 1.0))]
high_z_bin = full_AGN_cat[np.where(full_AGN_cat['REDSHIFT'] > 1.0)]

# To explore how different radius choices affect the mass-surface density relation we will calculate the histogram
# at multiple radii.
radial_mass_bins = np.array([0.5, 1.0, 1.5, 2.0])

for radius in radial_mass_bins:
    # Generate the histograms and errors for the AGN surface density per cluster binned by halo mass.
    mid_low_z_mass_surf_den, mid_low_z_mass_surf_den_err = make_z_mass_bin_histogram(mid_low_z_bin, mass_bins, radius)
    print('mid low')
    mid_mid_z_mass_surf_den, mid_mid_z_mass_surf_den_err = make_z_mass_bin_histogram(mid_mid_z_bin, mass_bins, radius)
    print('mid mid')
    mid_high_z_mass_surf_den, mid_high_z_mass_surf_den_err = make_z_mass_bin_histogram(mid_high_z_bin, mass_bins, radius)
    print('mid high')
    high_z_mass_surf_den, high_z_mass_surf_den_err = make_z_mass_bin_histogram(high_z_bin, mass_bins, radius)
    print('high')

    # Center the bins
    mass_bin_cent = mass_bins[:-1] + np.diff(mass_bins) / 2.

    np.save('Data/Mass_{rad}r500_bin_data'.format(rad=radius),
            {'mass_bin_cent': mass_bin_cent,
             'mid_low_z_mass_surf_den': mid_low_z_mass_surf_den,
             'mid_low_z_mass_surf_den_err': mid_low_z_mass_surf_den_err,
             'mid_mid_z_mass_surf_den': mid_mid_z_mass_surf_den,
             'mid_mid_z_mass_surf_den_err': mid_mid_z_mass_surf_den_err,
             'mid_high_z_mass_surf_den': mid_high_z_mass_surf_den,
             'mid_high_z_mass_surf_den_err': mid_high_z_mass_surf_den_err,
             'high_z_mass_surf_den': high_z_mass_surf_den,
             'high_z_mass_surf_den_err': high_z_mass_surf_den_err})


# # Generate the histogram heights for the AGN surface density per cluster binned by mass.
# low_z_mass_surf_den, low_z_mass_err = make_z_mass_bin_histogram(low_z_bin, mass_bins)
# high_z_mass_surf_den, high_z_mass_err = make_z_mass_bin_histogram(high_z_bin, mass_bins)
#
# # Center the bins
# mass_bin_cent = mass_bins[:-1] + np.diff(mass_bins) / 2.
#
# # Make the mass plot
# fig, ax = plt.subplots()
# # ax.xaxis.set_minor_locator(AutoMinorLocator(5))
# ax.errorbar(mass_bin_cent, low_z_mass_surf_den, yerr=low_z_mass_err, fmt='o', c='C0', label='$z \leq 0.8$')
# ax.errorbar(mass_bin_cent, high_z_mass_surf_den, yerr=high_z_mass_err, fmt='o', c='C1', label='$z > 0.8$')
# ax.axhline(y=field_surf_den.value, c='k', linestyle='--')
# ax.axhspan(ymax=field_surf_den.value + field_surf_den_err.value, ymin=field_surf_den.value - field_surf_den_err.value,
#            color='0.5', alpha=0.2)
# ax.set(title='239 SPT Clusters', xlabel='$M_{500} [M_\odot]$',
#        ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [arcmin$^{-2}$]',
#        xscale='log')
# ax.legend()
# fig.savefig('Data/Plots/SPT_AGN_Mass_Sci_Plot.pdf', format='pdf')