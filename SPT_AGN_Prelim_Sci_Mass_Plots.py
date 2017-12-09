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
from astropy.stats import bootstrap
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


def make_z_mass_bin_histogram(z_bin, mass_bins):  # TODO change from 2r500 area to the IMAGE_AREA value.

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
        annuli_coords = [image_coordinates[np.where(image_distances <= rad_bins_pix.value[i])]
                         for i in range(len(rad_bins_pix))]

        # For each annuli query the values of the pixels matching the coordinates found above and count the number of
        # good pixels (those with a value of `1`).
        area_pixels = [np.count_nonzero(mask_image[annulus.T[1], annulus.T[0]]) for annulus in annuli_coords]

        # Convert the pixel areas into arcmin^2 areas.
        area_arcmin2 = area_pixels * pix_scale.to(u.arcmin) * pix_scale.to(u.arcmin)

        return area_arcmin2

    radial_mass_bins = np.array([0.5, 1.0, 1.5, 2.0])

    # Make the histogram binned on mass.
    z_mass_surf_den, _ = np.histogram(z_bin['M500'],
                                      weights=z_bin['completeness_correction'],
                                      bins=mass_bins)

    # For the errors we will preform a bootstrap resampling at the individual AGN level before the histogram is made.
    z_mass_boot_array = np.array([z_bin['M500'], z_bin['completeness_correction'], cluster_area]).T
    with NumpyRNGContext(1):
        z_mass_bootresult = bootstrap(z_mass_boot_array, 1000)

    # Make the histograms for each resampling
    z_mass_boot_surf_den = []
    for z_mass_boot_samp in z_mass_bootresult:
        z_mass_boot_table = Table(z_mass_boot_samp, names=['M500', 'completeness_correction', 'cluster_area'])

        z_mass_boot_hist, _ = np.histogram(z_mass_boot_table['M500'],
                                           weights=(z_mass_boot_table['completeness_correction'] /
                                                    z_mass_boot_table['cluster_area']),
                                           bins=mass_bins)
        z_mass_boot_surf_den.append(z_mass_boot_hist)

    # Compute the standard deviation of surface densities for the bootstrapped histograms
    z_mass_surf_den_err = np.std(z_mass_boot_surf_den, axis=0)

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


# Combine all the catalogs into a single table.
full_AGN_cat = vstack(AGN_cats)

# Set up mass bins
mass_bins = np.arange(2e14, 1.3e15, 0.5e14)

# Set our redshift bins
low_z_bin = full_AGN_cat[np.where(full_AGN_cat['REDSHIFT'] <= 0.5)]
mid_low_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.5) & (full_AGN_cat['REDSHIFT'] <= 0.65))]
mid_mid_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.65) & (full_AGN_cat['REDSHIFT'] <= 0.75))]
mid_high_z_bin = full_AGN_cat[np.where((full_AGN_cat['REDSHIFT'] > 0.75) & (full_AGN_cat['REDSHIFT'] <= 1.0))]
high_z_bin = full_AGN_cat[np.where(full_AGN_cat['REDSHIFT'] > 1.0)]



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