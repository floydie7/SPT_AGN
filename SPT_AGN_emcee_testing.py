"""
SPT_AGN_emcee.py
Author: Benjamin Floyd

This script will preform the Bayesian analysis on the SPT-AGN data to produce the posterior probability distributions
for all fitting parameters.
"""

from __future__ import print_function, division

from itertools import product
from time import time

import astropy.units as u
import corner
import emcee
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from custom_math import trap_weight  # Custom trapezoidal integration
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import cdist

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def good_pixel_fraction(r, z, r500, image_name, center):
    # Read in the mask file and the mask file's WCS
    image, header = fits.getdata(image_name, header=True)
    image_wcs = WCS(header)

    # From the WCS get the pixel scale
    pix_scale = (image_wcs.pixel_scale_matrix[1, 1] * u.deg).to(u.arcsec)

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

    # Convert our radius to pixels
    r_pix = r * r500 * cosmo.arcsec_per_kpc_proper(z).to(u.arcsec / u.Mpc) / pix_scale

    # Because we potentially integrate to larger radii than can be fit on the image we will need to increase the size of
    # our mask. To do this, we will pad the mask with a zeros out to the radius we need.
    # Find the width needed to pad the image to include the largest radius inside the image.
    width = (int(np.max(r_pix) - image.shape[0] // 2), int(np.max(r_pix) - image.shape[1] // 2))

    # Insure that we are adding a non-negative padding width.
    if (width[0] <= 0) or (width[1] <= 0):
        width = (0, 0)

    large_image = np.pad(image, pad_width=width, mode='constant', constant_values=0)

    # find the distances from center pixel to all other pixels
    image_coords = np.array(list(product(range(large_image.shape[0]), range(large_image.shape[1]))))

    center_coord = np.array(center_pix) + np.array(width) + 1
    center_coord = center_coord.reshape((1, 2))

    image_dists = cdist(image_coords, center_coord).reshape(large_image.shape)

    # select all pixels that are within the annulus
    good_pix_frac = []
    pix_area = []
    for i in np.arange(len(r_pix) - 1):
        pix_ring = large_image[np.where((image_dists >= r_pix[i]) & (image_dists < r_pix[i + 1]))]

        # Calculate the fraction
        good_pix_frac.append(np.sum(pix_ring) / len(pix_ring))

    return good_pix_frac


def model_rate(z, m, r500, r_r500, maxr, params):
    """
    Our generating model.

    :param z: Redshift of the cluster
    :param m: M_500 mass of the cluster
    :param r500: r500 radius of the cluster
    :param maxr: maximum radius in units of r500 to consider
    :param r_r500: A vector of radii of objects within the cluster normalized by the cluster's r500
    :param params: Tuple of (theta, eta, zeta, beta, background)
    :return model: A surface density profile of objects as a function of radius
    """

    # Unpack our parameters
    theta, eta, zeta, beta = params

    # theta = theta / u.Mpc**2 * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc/u.arcmin)**2

    # Convert our background surface density from angular units into units of r500^-2
    background = 0.371 / u.arcmin ** 2 * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc) ** 2 * r500 ** 2

    # r_r500 = r * u.arcmin * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc/u.arcmin) / r500

    # The cluster's core radius in units of r500
    rc_r500 = 0.1 * u.Mpc / r500

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc_r500) ** 2) ** (-1.5 * beta + 0.5) + background

    # We impose a cut off of all objects with a radius greater than 1.1r500
    model[r_r500 > maxr] = 0.

    return model.value


# Set our log-likelihood
def lnlike(param, maxr, catalog):

    catalog_grp = catalog.group_by('SPT_ID')

    lnlike_list = []
    for cluster in catalog_grp.groups:
        # For convenience get the cluster information from the catalog
        cluster_z = cluster['REDSHIFT'][0]
        cluster_m500 = cluster['M500'][0] * u.Msun
        cluster_r500 = cluster['r500'][0] * u.Mpc
        cluster_mask = cluster['MASK_NAME'][0]
        cluster_sz_cent = cluster['SZ_RA', 'SZ_DEC'][0]

        ni = model_rate(cluster_z, cluster_m500, cluster_r500, cluster['radial_r500'], maxr, param)

        rall = np.linspace(0, maxr, num=100)  # radius in r500^-1 units
        nall = model_rate(cluster_z, cluster_m500, cluster_r500, rall, maxr, param)

        # Calculate the good pixel fraction of the annuli in rall
        gpf_all = good_pixel_fraction(rall, cluster_z, cluster_r500, cluster_mask, cluster_sz_cent)

        # Use a spatial possion point-process log-likelihood
        cluster_lnlike = (np.sum(np.log(ni * cluster['radial_r500']))
                          - trap_weight(nall * 2*np.pi * rall, rall, weight=gpf_all))
        lnlike_list.append(cluster_lnlike)

    total_lnlike = np.sum(lnlike_list)

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(param):
    # Extract our parameters
    theta, eta, zeta, beta = param

    # # Set our hyperparameters
    # h_C = 0.371
    # h_C_err = 0.157

    # Define all priors to be gaussian
    if 0. <= theta <= 24. and -3. <= eta <= 3. and -3. <= zeta <= 3. and -3. <= beta <= 3.:
        theta_lnprior = 0.0
        eta_lnprior = 0.0
        beta_lnprior = 0.0
        zeta_lnprior = 0.0
    else:
        theta_lnprior = -np.inf
        eta_lnprior = -np.inf
        beta_lnprior = -np.inf
        zeta_lnprior = -np.inf

    # C_lnprior = -0.5 * np.sum((C - h_C)**2 / h_C_err**2)

    # Assuming all parameters are independent the joint log-prior is
    total_lnprior = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior  # + C_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(param, maxr, catalog):
    lp = lnprior(param)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(param, maxr, catalog)


# Read in the mock catalog
mock_catalog = Table.read('Data/MCMC/Mock_Catalog/Catalogs/pre-final_tests/'
                          'mock_AGN_catalog_t12.00_e1.20_z-1.00_b0.50_maxr1.00_seed890.cat', format='ascii')


# Set parameter values
theta_true = 12    # Amplitude.
eta_true = 1.2       # Redshift slope
beta_true = 0.5      # Radial slope
zeta_true = -1.0     # Mass slope
C_true = 0.371       # Background AGN surface density

max_radius = 1.0

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 4
nwalkers = 16

# Also, set the number of steps to run the sampler for.
nsteps = 500

# We will initialize our walkers in a tight ball near the initial parameter values.
pos0 = emcee.utils.sample_ball(p0=[theta_true, eta_true, zeta_true, beta_true],
                               std=[1e-2, 1e-2, 1e-2, 1e-2], size=nwalkers)

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=[max_radius, mock_catalog], threads=4)

# Run the sampler.
start_sampler_time = time()
sampler.run_mcmc(pos0, nsteps)
print('Sampler runtime: {:.2f} s'.format(time() - start_sampler_time))
np.save('Data/MCMC/Mock_Catalog/Chains/pre-final_tests/'
        'emcee_run_w{nwalkers}_s{nsteps}_mock_t{theta}_e{eta}_z{zeta}_b{beta}_maxr{maxr}'
        .format(nwalkers=nwalkers, nsteps=nsteps,
                theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, maxr=max_radius),
        sampler.chain)

# Plot the chains
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, sharex='col')

ax0.plot(sampler.chain[:, :, 0].T, color='k', alpha=0.4)
ax0.axhline(theta_true, color='b')
ax0.yaxis.set_major_locator(MaxNLocator(5))
ax0.set(ylabel=r'$\theta$', title='MCMC Chains')

ax1.plot(sampler.chain[:, :, 1].T, color='k', alpha=0.4)
ax1.axhline(eta_true, color='b')
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.set(ylabel=r'$\eta$')

ax2.plot(sampler.chain[:, :, 2].T, color='k', alpha=0.4)
ax2.axhline(zeta_true, color='b')
ax2.yaxis.set_major_locator(MaxNLocator(5))
ax2.set(ylabel=r'$\zeta$')

ax3.plot(sampler.chain[:, :, 3].T, color='k', alpha=0.4)
ax3.axhline(beta_true, color='b')
ax3.yaxis.set_major_locator(MaxNLocator(5))
ax3.set(ylabel=r'$\beta$', xlabel='Steps')

# ax4.plot(sampler.chain[:, :, 4].T, color='k', alpha=0.4)
# ax4.axhline(beta_true, color='b')
# ax4.yaxis.set_major_locator(MaxNLocator(5))
# ax4.set(ylabel=r'$\C$', xlabel='Steps')

fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/'
            'Param_chains_mock_t{theta}_e{eta}_z{zeta}_b{beta}_maxr{maxr}_new.pdf'
            .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, maxr=max_radius),
            format='pdf')

# Remove the burnin, typically 1/3 number of steps
burnin = nsteps//3
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Produce the corner plot
fig = corner.corner(samples, labels=[r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$'],
                    truths=[theta_true, eta_true, zeta_true, beta_true],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/'
            'Corner_plot_mock_t{theta}_e{eta}_z{zeta}_b{beta}_maxr{maxr}_new.pdf'
            .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, maxr=max_radius),
            format='pdf')

theta_mcmc, eta_mcmc, zeta_mcmc, beta_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                 zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print("""MCMC Results:
theta = {theta[0]:.2f} +{theta[1]:.3f} -{theta[2]:.3f} (truth: {theta_true})
eta = {eta[0]:.2f} +{eta[1]:.3f} -{eta[2]:.3f} (truth: {eta_true})
zeta = {zeta[0]:.2f} +{zeta[1]:.3f} -{zeta[2]:.3f} (truth: {zeta_true})
beta = {beta[0]:.2f} +{beta[1]:.3f} -{beta[2]:.3f} (truth: {beta_true})"""
      .format(theta=theta_mcmc, eta=eta_mcmc,  zeta=zeta_mcmc, beta=beta_mcmc,
              theta_true=theta_true, eta_true=eta_true,  zeta_true=zeta_true, beta_true=beta_true))

print('Mean acceptance fraction: {}'.format(np.mean(sampler.acceptance_fraction)))
# print('Integrates autocorrelation time: {}'.format(sampler.get_autocorr_time()))
