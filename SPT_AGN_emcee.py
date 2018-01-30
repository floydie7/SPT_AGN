"""
SPT_AGN_emcee.py
Author: Benjamin Floyd

This script will preform the Bayesian analysis on the SPT-AGN data to produce the posterior probability distributions
for all fitting parameters.
"""

from __future__ import print_function, division

from itertools import product
from os import listdir

import astropy.units as u
import corner
import emcee
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack, QTable
from astropy.wcs import WCS
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import cdist
from small_poisson import small_poisson
from time import time

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Set our cosmology
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

# Read in the Bleem catalog. We'll need the cluster center coordinates to anchor the radial annuli.
Bleem = Table.read('Data/2500d_cluster_sample_fiducial_cosmology.fits')


def aperture_pixel_area(spt_id, radius):
    """
    Calculates the area of all 'good' pixels in the image using the previously generated mask images.

    :param spt_id: The SPT ID for the cluster as defined in Bleem et al. 2015.
    :type spt_id: str

    :param radius: The outer radius of the aperture in units of arcmin.
    :type radius: float

    :return: The area of all good pixels within the aperture in arcmin^2.
    :rtype: float
    """

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
    radius_pix = radius / pix_scale.to(u.arcmin)

    # Generate the list of coordinates
    image_coordinates = np.array(list(product(range(fits.getval(mask_filename, 'NAXIS1')),
                                              range(fits.getval(mask_filename, 'NAXIS2')))))

    # Calculate the distances from the cluster center to all other pixels
    image_distances = cdist(image_coordinates, np.array([[cluster_x[0], cluster_y[0]]])).flatten()

    # Select the coordinates in the annuli
    annuli_coords = [image_coordinates[np.where(image_distances <= radius_pix.value)]]

    # For each annuli query the values of the pixels matching the coordinates found above and count the number of
    # good pixels (those with a value of `1`).
    area_pixels = [np.count_nonzero(mask_image[annulus.T[1], annulus.T[0]]) for annulus in annuli_coords]

    # Convert the pixel areas into arcmin^2 areas.
    area_arcmin2 = area_pixels * pix_scale.to(u.arcmin) * pix_scale.to(u.arcmin)

    return area_arcmin2


def observed_surf_den(catalog):
    # Group the catalog by cluster
    cat_grped = catalog.group_by('SPT_ID')

    total_surf_den = []
    total_surf_den_err = []
    for cluster in cat_grped.groups:
        # Create a dictionary to store the relevant data.
        cluster_dict = {'spt_id': [cluster['SPT_ID'][0]]}

        # Sum over all the completeness corrections for all AGN within our selected aperture.
        cluster_counts = np.sum(cluster['completeness_correction'])

        # Calculate the 1-sigma Poisson errors for the AGN observed.
        cluster_err = small_poisson(cluster_counts, s=1)

        # Using the area enclosed by our aperture calculate the surface density.
        cluster_surf_den = cluster_counts / cluster['aper_area'][0]

        # Also convert our errors to surface densities.
        cluster_surf_den_err_upper = cluster_err[0] / cluster['aper_area'][0]
        cluster_surf_den_err_lower = cluster_err[1] / cluster['aper_area'][0]

        cluster_dict.update({'surf_den': [cluster_surf_den]})
        total_surf_den.append(Table(cluster_dict))
        total_surf_den_err.append((cluster_surf_den_err_upper, cluster_surf_den_err_lower))

    surf_den_table = vstack(total_surf_den)

    # Find the mean surface density over the entire sample accounting for the 2 dropped clusters.
    surf_den_mean = np.sum(np.array(surf_den_table['surf_den'])) / (len(surf_den_table) + 2.) / u.arcmin**2

    # Separate the upper and lower errors for each cluster.
    upper_err = np.array([error[0] for error in total_surf_den_err])
    lower_err = np.array([error[1] for error in total_surf_den_err])

    # Combine all the errors in quadrature and divide by the total number of clusters in the sample.
    upper_surf_den_err = np.sqrt(np.sum(upper_err**2, axis=0)) / (len(surf_den_table) + 2.) / u.arcmin**2
    lower_surf_den_err = np.sqrt(np.sum(lower_err**2, axis=0)) / (len(surf_den_table) + 2.) / u.arcmin**2

    # Combine the upper and lower errors to give a variance.
    surf_den_var = upper_surf_den_err * lower_surf_den_err

    return surf_den_mean, surf_den_var


# Set our log-likelihood
def lnlike(param, catalog, surf_den_mean, surf_den_var):
    # Extract our parameters
    eta, beta, zeta, C = param

    # Get the total number of clusters in the sample and add 2 to account for the dropped clusters.
    N_cl = len(catalog.group_by('SPT_ID').groups.keys) + 2.

    # Convert our catalog into a QTable so units are handled correctly.
    qcatalog = QTable(catalog)

    # Set model
    model = 1./N_cl * np.sum((1. / qcatalog['aper_area'])
                             * (1 + qcatalog['REDSHIFT'])**eta
                             * (qcatalog['RADIAL_DIST_Mpc'] / qcatalog['r500'])**beta
                             * (qcatalog['M500'] / (1e15 * u.Msun))**zeta) + (C / u.arcmin**2)

    # Return the log-likelihood function
    return -0.5 * np.sum((surf_den_mean - model)**2 / surf_den_var)


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(param):
    # Extract our parameters
    eta, beta, zeta, C = param

    # Set our hyperparameters
    h_eta = 0.
    h_eta_err = np.inf
    h_beta = 0.
    h_beta_err = np.inf
    h_zeta = 0.
    h_zeta_err = np.inf
    h_C = 0.371
    h_C_err = 0.157

    # Define all priors to be gaussian
    if -10. <= eta <= 10. and -10. <= beta <= 10. and -10. <= zeta <= 10.:
        eta_lnprior = -0.5 * np.sum((eta - h_eta)**2 / h_eta_err**2)
        beta_lnprior = -0.5 * np.sum((beta - h_beta)**2 / h_beta_err**2)
        zeta_lnprior = -0.5 * np.sum((zeta - h_zeta)**2 / h_zeta_err**2)
    else:
        eta_lnprior = -np.inf
        beta_lnprior = -np.inf
        zeta_lnprior = -np.inf

    C_lnprior = -0.5 * np.sum((C - h_C)**2 / h_C_err**2)

    # Assuming all parameters are independent the joint log-prior is
    total_lnprior = eta_lnprior + beta_lnprior + zeta_lnprior + C_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(param, catalog, surf_den, surf_den_err):
    lp = lnprior(param)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(param, catalog, surf_den, surf_den_err)


# Read in all the catalogs
AGN_cats = [Table.read('Data/Output/'+f, format='ascii') for f in listdir('Data/Output/') if not f.startswith('.')]

final_AGN_cats = []
for cat in AGN_cats:
    # Convert the radial distance column in the catalogs from arcmin to Mpc.
    cat['RADIAL_DIST'].unit = u.arcmin
    cat['RADIAL_DIST_Mpc'] = (cat['RADIAL_DIST'] * cosmo.kpc_proper_per_arcmin(cat['REDSHIFT'])).to(u.Mpc)

    # Calculate the r500
    cat['M500'].unit = u.Msun
    cat['r500'] = (3 * cat['M500'] /
                   (4 * np.pi * 500 * cosmo.critical_density(cat['REDSHIFT']).to(u.Msun / u.Mpc ** 3)))**(1/3)

    # Convert the fractional aperture radius into a physical distance.
    cluster_radius_mpc = 1.5 * cat['r500'][0] * u.Mpc

    # Convert the physical radius to an angle.
    cluster_radius_arcmin = cluster_radius_mpc.to(u.kpc) / cosmo.kpc_proper_per_arcmin(cat['REDSHIFT'][0])

    # Calculate the area enclosed by our aperture in units of arcmin2.
    cat['aper_area'] = aperture_pixel_area(cat['SPT_ID'][0], cluster_radius_arcmin)

    # Select only the AGN within the aperture.
    final_cat = cat[np.where(cat['RADIAL_DIST'] <= cluster_radius_arcmin.value)]

    final_AGN_cats.append(final_cat)

# Combine all the catalogs into a single table.
full_AGN_cat = vstack(final_AGN_cats)

# Find the total observed AGN surface density and variance.
surf_den_obs, surf_den_obs_var = observed_surf_den(full_AGN_cat)
print(surf_den_obs, surf_den_obs_var)

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 4
nwalkers = 64

# Also, set the number of steps to run the sampler for.
nsteps = 1000

# We will initialize our walkers in a tight ball near the initial parameter values.
pos0 = emcee.utils.sample_ball(p0=[0., 0., 0., 0.371], std=[1e-2, 1e-2, 1e-2, 0.157], size=nwalkers)

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(full_AGN_cat, surf_den_obs, surf_den_obs_var), threads=4)

# Run the sampler.
start_sampler_time = time()
sampler.run_mcmc(pos0, nsteps)
print('Sampler runtime: {:.2f} s'.format(time() - start_sampler_time))

# Plot the chains
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
ax1.plot(sampler.chain[:, :, 0].T, color='k', alpha=0.4)
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.set(ylabel=r'$\eta$', title='MCMC Chains')

ax2.plot(sampler.chain[:, :, 1].T, color='k', alpha=0.4)
ax2.yaxis.set_major_locator(MaxNLocator(5))
ax2.set(ylabel=r'$\beta$')

ax3.plot(sampler.chain[:, :, 2].T, color='k', alpha=0.4)
ax3.yaxis.set_major_locator(MaxNLocator(5))
ax3.set(ylabel=r'$\zeta$')

ax4.plot(sampler.chain[:, :, 3].T, color='k', alpha=0.4)
ax4.yaxis.set_major_locator(MaxNLocator(5))
ax4.set(ylabel=r'$C$', xlabel='Steps')

fig.savefig('Data/MCMC/Param_chains.pdf', format='pdf')

# Remove the burnin, typically 1/3 number of steps
burnin = nsteps//3
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Produce the corner plot
fig = corner.corner(samples, labels=[r'$\eta$', r'$\beta$', r'$\zeta$', r'$C$'], truths=[0.0, 0.0, 0.0, 0.371],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig('Data/MCMC/Corner_plot.pdf', format='pdf')

eta_mcmc, beta_mcmc, zeta_mcmc, C_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print("""MCMC Results:
eta = {eta[0]:.2f} +{eta[1]:.3f} -{eta[2]:.3f}
beta = {beta[0]:.2f} +{beta[1]:.3f} -{beta[2]:.3f}
zeta = {zeta[0]:.2f} +{zeta[1]:.3f} -{zeta[2]:.3f}
C = {C[0]:.2f} +{C[1]:.3f} -{C[2]:.3f}""".format(eta=eta_mcmc, beta=beta_mcmc, zeta=zeta_mcmc, C=C_mcmc))

print('Mean acceptance fraction: {}'.format(np.mean(sampler.acceptance_fraction)))
# print('Integrates autocorrelation time: {}'.format(sampler.get_autocorr_time()))
