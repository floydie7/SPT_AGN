"""
SPT_AGN_emcee.py
Author: Benjamin Floyd

This script will preform the Bayesian analysis on the SPT-AGN data to produce the posterior probability distributions
for all fitting parameters.
"""

from __future__ import print_function, division

from time import time

import astropy.units as u
import corner
import emcee
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.ticker import MaxNLocator

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)


def model_rate(z, m, r_r500, params):
    """
    Our generating model.

    :param z: Redshift
    :param m: M_500
    :param r_r500: r / r_500
    :param params: Tuple of (theta, eta, zeta, beta)
    :return:
    """

    theta, eta, zeta, beta = params

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z)**eta * (m / (1e15 * u.Msun))**zeta

    model = a * r_r500**beta

    model[r_r500 > 1.5] = 0.

    return model


# Set our log-likelihood
def lnlike(param, catalog, obs_surf_den_table=None):
    # Extract our parameters
    theta, eta, zeta, beta = param

    ni = model_rate(catalog['REDSHIFT'][0], catalog['M500'][0]*u.Msun, catalog['r_r500'], param)

    rall = np.linspace(0.1, 1.5, 100)
    nall = model_rate(catalog['REDSHIFT'][0], catalog['M500'][0]*u.Msun, rall, param)

    # Use a spatial possion point-process log-likelihood
    total_lnlike = np.sum(np.log(ni * catalog['r_r500'])) - np.trapz(nall * 2*np.pi * rall, rall)

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
    if 0. <= theta <= 1. and -3. <= eta <= 3. and -3. <= zeta <= 3. and -3. <= beta <= 3.:
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
    total_lnprior = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior  #+ C_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(param, catalog, x):
    lp = lnprior(param)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(param, catalog, x)


# Read in the mock catalog
mock_catalog = Table.read('Data/MCMC/Mock_Catalog/Catalogs/new_mock_test.cat', format='ascii')
mock_catalog['M500'].unit = u.Msun


# For diagnostic purposes, set the values of the parameters.
theta_true = 2e-4
eta_true = 1.2
beta_true = -1.5
zeta_true = -1.0

# nlnlike = lambda *args: -lnlike(*args)
# mle = op.minimize(nlnlike, x0=np.array([0,0,0]), args=(mock_catalog, None))

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 4
nwalkers = 100

# Also, set the number of steps to run the sampler for.
nsteps = 500

# We will initialize our walkers in a tight ball near the initial parameter values.
pos0 = emcee.utils.sample_ball(p0=[theta_true, eta_true, zeta_true, beta_true],
                               std=[1e-6, 1e-2, 1e-2, 1e-2], size=nwalkers)

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(mock_catalog, None), threads=1)

# Run the sampler.
start_sampler_time = time()
sampler.run_mcmc(pos0, nsteps)
print('Sampler runtime: {:.2f} s'.format(time() - start_sampler_time))
np.save('Data/MCMC/Mock_Catalog/Chains/emcee_run_w{nwalkers}_s{nsteps}_new_mock_test'
        .format(nwalkers=nwalkers, nsteps=nsteps),
        sampler.chain)

# Plot the chains
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, sharex=True)

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

fig.savefig('Data/MCMC/Mock_Catalog/Plots/Param_chains_new_mock_test.pdf', format='pdf')

# Remove the burnin, typically 1/3 number of steps
burnin = nsteps//3
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Produce the corner plot
fig = corner.corner(samples, labels=[r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$'],
                    truths=[theta_true, eta_true, zeta_true, beta_true],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Corner_plot_new_mock_test.pdf', format='pdf')

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
