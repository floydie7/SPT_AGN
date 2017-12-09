"""
SPT_AGN_Prelim_Sci_Radial_Plots.py
Author: Benjamin Floyd

This script generates the preliminary Radial trend science plots for the SPT AGN study.
"""

from __future__ import print_function, division

from os import listdir

import astropy.units as u
import matplotlib
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.utils import NumpyRNGContext
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


def make_z_rad_bin_histogram(z_bin, rad_bins):

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
        annuli_coords = [image_coordinates[np.where((image_distances > rad_bins_pix.value[i]) &
                                                    (image_distances <= rad_bins_pix.value[i+1]))]
                         for i in range(len(rad_bins_pix)-1)]

        # For each annuli query the values of the pixels matching the coordinates found above and count the number of
        # good pixels (those with a value of `1`).
        area_pixels = [np.count_nonzero(mask_image[annulus.T[1], annulus.T[0]]) for annulus in annuli_coords]

        # Convert the pixel areas into arcmin^2 areas.
        area_arcmin2 = area_pixels * pix_scale.to(u.arcmin) * pix_scale.to(u.arcmin)

        return area_arcmin2

    def radial_surf_den(_z_bin, _rad_bins):
        # Group the catalog by cluster
        z_bin_grouped = _z_bin.group_by('SPT_ID')

        total_rad_surf_den = []
        cluster_field_rad_surf_den_err = []
        for cluster in z_bin_grouped.groups:
            print('Cluster: {}'.format(cluster['SPT_ID'][0]))
            # Convert the fractional radial bin locations to physical distances.
            cluster_rad_bin = _rad_bins * cluster['r500'][0] * u.Mpc

            # Convert the radial bins from Mpc to arcmin.
            cluster_rad_bin_arcmin = (cluster_rad_bin /
                                      cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0]).to(u.Mpc / u.arcmin))

            # Make the histogram of AGN in each bin, weighted by the completeness corrected number.
            cluster_rad_hist, _ = np.histogram(cluster['RADIAL_DIST_Mpc'],
                                               weights=cluster['completeness_correction'],
                                               bins=cluster_rad_bin)
            print('Observed AGN counts: {}'.format(cluster_rad_hist))

            # Calculate the area of each annulus in arcmin^2 using only the good pixels from the cluster pixel map.
            cluster_bin_area_arcmin2 = annulus_pixel_area(cluster['SPT_ID'][0], cluster_rad_bin_arcmin)
            print('Annuli area (arcmin2): {}'.format(cluster_bin_area_arcmin2))

            # Using the SDWFS field surface density and the area of the annuli, calculate the expected AGN counts in the
            # radial bins.
            field_rad_counts = cluster_bin_area_arcmin2 * field_surf_den
            print('SDWFS expected counts: {}'.format(field_rad_counts))

            # Calculate the Poisson errors for both the observed cluster counts and the field expected counts.
            cluster_poisson_err = small_poisson(cluster_rad_hist, S=1)
            field_poisson_err = small_poisson(field_rad_counts, S=1)
            print('cluster errors: {cl}\nfield errors: {fl}'.format(cl=cluster_poisson_err, fl=field_poisson_err))

            # Subtract the field expectation from the observed cluster counts.
            cluster_field_counts = cluster_rad_hist - field_rad_counts
            print('Field-subtracted AGN counts: {}'.format(cluster_field_counts))

            # Propagate the Poisson errors of both the cluster and field counts to the excess counts.
            cluster_field_counts_upper_err = np.sqrt(cluster_poisson_err[0]**2 + field_poisson_err[0]**2)
            cluster_field_counts_lower_err = np.sqrt(cluster_poisson_err[1] ** 2 + field_poisson_err[1] ** 2)
            print('Field subtracted errors\nupper: {up}\nlower: {lo}'.format(up=cluster_field_counts_upper_err, lo=cluster_field_counts_lower_err))

            # Calculate the AGN excess surface density for each radial bin.
            cluster_field_rad_surf_den = cluster_field_counts / cluster_bin_area_arcmin2
            print('field subtracted surface densities: {}'.format(cluster_field_rad_surf_den))

            # Convert the errors to surface densities.
            cluster_field_surf_den_upper_err = cluster_field_counts_upper_err / cluster_bin_area_arcmin2
            cluster_field_surf_den_lower_err = cluster_field_counts_lower_err / cluster_bin_area_arcmin2
            print('surface density errors\nupper: {up}\nlower: {lo}'.format(up=cluster_field_surf_den_upper_err, lo=cluster_field_surf_den_lower_err))

            # Using the cluster's redshift convert the angular surface density into physical units.
            cluster_rad_surf_den = cluster_field_rad_surf_den / (cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0])
                                                                 .to(u.Mpc / u.arcmin))**2

            # Also convert the errors to physical units
            cluster_rad_surf_den_err = (cluster_field_surf_den_upper_err /
                                        (cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0]).to(u.Mpc / u.arcmin))**2,
                                        cluster_field_surf_den_lower_err /
                                        (cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT'][0]).to(u.Mpc / u.arcmin))**2)

            total_rad_surf_den.append(cluster_rad_surf_den)
            cluster_field_rad_surf_den_err.append(cluster_rad_surf_den_err)

        return total_rad_surf_den, cluster_field_rad_surf_den_err

    # Make the data point
    total_rad_surf_den, cluster_field_poisson_err = radial_surf_den(z_bin, rad_bins)
    # print(total_rad_surf_den)

    # Compute the average AGN surface density per cluster. We use `nanmean` to avoid issues with any NaN values in the
    # sample. The NaNs can occur due to a cluster having radial bins are completely off the image resulting in 0 objects
    # divided by 0 area which gives a NaN for that cluster's surface density bin.
    z_rad_surf_den = np.nanmean(total_rad_surf_den, axis=0)

    # For the errors, we will preform a bootstrap resampling at the AGN level for the surface densities.
    z_rad_boot_array = np.array(
        [z_bin['RADIAL_DIST_Mpc'], z_bin['completeness_correction'], z_bin['r500'], z_bin['REDSHIFT'],
         z_bin['SPT_ID']]).T
    with NumpyRNGContext(1):
        z_rad_bootresult = spt_bootstrap(z_rad_boot_array, 100)

    # Make the histograms for each resampling.
    z_rad_boot_surf_den = []
    for z_rad_boot_samp in z_rad_bootresult:
        # Set up a table for the bootstrapped sample.
        z_rad_boot_table = Table(z_rad_boot_samp,
                                 names=['RADIAL_DIST_Mpc', 'completeness_correction', 'r500', 'REDSHIFT', 'SPT_ID'],
                                 dtype=('f8', 'f8', 'f8', 'f8', 'S32'))

        # Compute the radial surface densities for the sample.
        z_rad_boot_hist, _ = radial_surf_den(z_rad_boot_table, rad_bins)

        # Find the average surface density for the sample using `nanmean` to ignore NaN values.
        z_rad_boot_hist_mean = np.nanmean(z_rad_boot_hist, axis=0)
        z_rad_boot_surf_den.append(z_rad_boot_hist_mean)

    # Compute the standard deviation of the bootstrapped surface densities in the radial bins.
    z_rad_boot_err = np.std(z_rad_boot_surf_den, axis=0)

    # Extract the upper and lower Poisson errors for each cluster.
    cluster_field_poisson_upper_err = np.array([error[0] for error in cluster_field_poisson_err])
    cluster_field_poisson_lower_err = np.array([error[1] for error in cluster_field_poisson_err])

    # We want the errors to be conservative so we will calculate the Poisson error in each bin and average over the
    # number of clusters. If the Poisson error is larger than the bootstrapped error for that bin, we replace with the
    # Poisson error.
    z_rad_poisson_upper_err = (np.sqrt(np.nansum(np.array(cluster_field_poisson_upper_err)**2, axis=0))
                               / np.count_nonzero(~np.isnan(total_rad_surf_den), axis=0))
    z_rad_poisson_lower_err = (np.sqrt(np.nansum(np.array(cluster_field_poisson_lower_err)**2, axis=0))
                               / np.count_nonzero(~np.isnan(total_rad_surf_den), axis=0))
    print('Combined radial bin errors\nupper: {up}\nlower: {lo}'.format(up=z_rad_poisson_upper_err, lo=z_rad_poisson_lower_err))

    # Return the errors, using the larger of the two errors for each bin.
    # z_rad_surf_den_err = np.fmax(z_rad_boot_err, z_rad_poisson_err)
    z_rad_surf_den_err = [z_rad_poisson_upper_err, z_rad_poisson_lower_err]

    return z_rad_surf_den, z_rad_surf_den_err, total_rad_surf_den


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


# Combine all the catalogs into a single table.
full_AGN_cat = vstack(AGN_cats)

# Set up radial bins
rad_bin_r_r500 = np.arange(0., 2.5, 0.5)

# Set up mass bins
mass_bins = np.arange(2e14, 1.3e15, 0.5e14)

# Set our redshift bins
low_z_bin = full_AGN_cat[np.where(full_AGN_cat['REDSHIFT'] <= 0.5)]
mid_low_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.5) & (full_AGN_cat['REDSHIFT'] <= 0.65))]
mid_mid_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.65) & (full_AGN_cat['REDSHIFT'] <= 0.75))]
mid_high_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.75) & (full_AGN_cat['REDSHIFT'] <= 1.0))]
high_z_bin = full_AGN_cat[np.where(full_AGN_cat['REDSHIFT'] > 1.0)]

# Generate the histogram heights for the AGN surface density per cluster binned by radius.
# low_z_rad_surf_den, low_z_rad_err, low_z_total = make_z_rad_bin_histogram(low_z_bin, rad_bin_r_r500)
mid_low_z_rad_surf_den, mid_low_z_rad_err, mid_low_z_total = make_z_rad_bin_histogram(mid_low_z_bin, rad_bin_r_r500)
# mid_mid_z_rad_surf_den, mid_mid_z_rad_err, mid_mid_z_total = make_z_rad_bin_histogram(mid_mid_z_bin, rad_bin_r_r500)
# mid_high_z_rad_surf_den, mid_high_z_rad_err, mid_high_z_total = make_z_rad_bin_histogram(mid_high_z_bin, rad_bin_r_r500)
# high_z_rad_surf_den, high_z_rad_err, high_z_total = make_z_rad_bin_histogram(high_z_bin, rad_bin_r_r500)

# Print the counts from each z bin.
print("low z bin: {low} \nmid low z bin: {mid_l} \nmid mid z bin: {mid_m} \nmid high z bin: {mid_h} \nhigh z bin: {high}"
      .format(low='', mid_l=mid_low_z_rad_surf_den, mid_m=mid_mid_z_rad_surf_den, mid_h=mid_high_z_rad_surf_den,
              high=high_z_rad_surf_den))
print("low z error: {low} \nmid low z error: {mid_l} \nmid mid z error: {mid_m} \nmid high z error: {mid_h} \nhigh z error: {high}"
      .format(low='', mid_l=mid_low_z_rad_err, mid_m=mid_mid_z_rad_err, mid_h=mid_high_z_rad_err, high=high_z_rad_err))
raise SystemExit
# Center the bins
rad_bin_cent = rad_bin_r_r500[:-1] + np.diff(rad_bin_r_r500) / 2.

# Make the radial distance plot
fig, ax = plt.subplots()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
# ax.errorbar(rad_bin_cent, low_z_rad_surf_den, yerr=low_z_rad_err, fmt='x', c='C0', label='$z \leq 0.5$')
ax.errorbar(rad_bin_cent, mid_low_z_rad_surf_den, yerr=mid_low_z_rad_err[::-1], fmt='o', c='C0', label='$0.5 < z \leq 0.65$')
ax.errorbar(rad_bin_cent, mid_mid_z_rad_surf_den, yerr=mid_mid_z_rad_err[::-1], fmt='o', c='C1', label='$0.65 < z \leq 0.75$')
ax.errorbar(rad_bin_cent, mid_high_z_rad_surf_den, yerr=mid_high_z_rad_err[::-1], fmt='o', c='C2', label='$0.75 < z \leq 1.0$')
ax.errorbar(rad_bin_cent, high_z_rad_surf_den, yerr=high_z_rad_err, fmt='o', c='C3', label='$z > 1.0$')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.axhspan(ymax=field_surf_den.value + field_surf_den_err.value, ymin=field_surf_den.value - field_surf_den_err.value,
#            color='0.5', alpha=0.2)
ax.set(title='239 SPT Clusters', xlabel='r/r$_{500}$', ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]',
       xlim=[0, 2.5], ylim=[-3, 2])
ax.legend()
fig.savefig('Data/Plots/SPT_AGN_Radial_Distance_Sci_Plot.pdf', format='pdf')

# Make individual radial plots for each redshift bin.
# Calculate the mean redshifts for each bin.
mid_low_mean_z = mid_low_z_bin['REDSHIFT'].mean()
mid_mid_mean_z = mid_mid_z_bin['REDSHIFT'].mean()
mid_high_mean_z = mid_high_z_bin['REDSHIFT'].mean()
high_mean_z = mid_low_z_bin['REDSHIFT'].mean()

# Scale the field error to the mean redshifts
mid_low_field_err = field_surf_den_err.value * (cosmo.kpc_proper_per_arcmin(mid_low_mean_z).to(u.Mpc / u.arcmin))**2
mid_mid_field_err = field_surf_den_err.value * (cosmo.kpc_proper_per_arcmin(mid_mid_mean_z).to(u.Mpc / u.arcmin))**2
mid_high_field_err = field_surf_den_err.value * (cosmo.kpc_proper_per_arcmin(mid_high_mean_z).to(u.Mpc / u.arcmin))**2
high_field_err = field_surf_den_err.value * (cosmo.kpc_proper_per_arcmin(high_mean_z).to(u.Mpc / u.arcmin))**2


fig = plt.figure(figsize=(11, 8.5), tight_layout=False)
fig.suptitle('Overdensity of SPT AGN', va='center')
fig.text(0.5, 0.04, 'r/r$_{500}$', ha='center', va='top')
ax = fig.add_subplot(221)
ax.tick_params(bottom='off', labelbottom='off')


ax = fig.add_subplot(221, sharey=ax, sharex=ax)
ax.errorbar(rad_bin_cent, mid_low_z_rad_surf_den, yerr=mid_low_z_rad_err[::-1], fmt='o', c='C0', label='$0.5 < z \leq 0.65$')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
ax.axhspan(ymax=mid_low_field_err.value, ymin=0.0-mid_low_field_err.value, color='0.5', alpha=0.2)
ax.set(ylim=[-3, 2], ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]')
ax.legend()

ax = fig.add_subplot(222, sharey=ax, sharex=ax)
ax.errorbar(rad_bin_cent, mid_mid_z_rad_surf_den, yerr=mid_mid_z_rad_err[::-1], fmt='o', c='C1', label='$0.65 < z \leq 0.75$')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
ax.axhspan(ymax=mid_mid_field_err.value, ymin=0.0-mid_mid_field_err.value, color='0.5', alpha=0.2)
ax.set(ylim=[-3, 2])
ax.legend()

ax = fig.add_subplot(223, sharey=ax, sharex=ax)
ax.errorbar(rad_bin_cent, mid_high_z_rad_surf_den, yerr=mid_high_z_rad_err[::-1], fmt='o', c='C2', label='$0.75 < z \leq 1.0$')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
ax.axhspan(ymax=mid_high_field_err.value, ymin=0.0-mid_high_field_err.value, color='0.5', alpha=0.2)
ax.set(ylim=[-3, 2], ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]')
ax.legend()

ax = fig.add_subplot(224, sharey=ax, sharex=ax)
ax.errorbar(rad_bin_cent, high_z_rad_surf_den, yerr=high_z_rad_err[::-1], fmt='o', c='C3', label='$z > 1.0$')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
ax.axhspan(ymax=high_field_err.value, ymin=0.0-high_field_err.value, color='0.5', alpha=0.2)
ax.set(ylim=[-3, 2])
ax.legend()

fig.savefig('Data/Plots/SPT_AGN_Radial_Distance_Sci_Plot_Field_Errors.pdf', format='pdf')
