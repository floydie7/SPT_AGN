"""
SPT_AGN_emcee.py
Author: Benjamin Floyd

This script will preform the Bayesian analysis on the SPT-AGN data to produce the posterior probability distributions
for all fitting parameters.
"""

from __future__ import print_function, division

import astropy.units as u
import corner
import emcee
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack, QTable, join
from matplotlib.ticker import MaxNLocator
from small_poisson import small_poisson
from time import time

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)


def observed_surf_den(catalog):
    # Group the catalog by cluster
    cat_grped = catalog.group_by('SPT_ID')

    total_surf_den = []
    for cluster in cat_grped.groups:
        # Create a dictionary to store the relevant data.
        cluster_dict = {'SPT_ID': [cluster['SPT_ID'][0]]}

        # Sum over all the completeness corrections for all AGN within our selected aperture.
        cluster_counts = len(cluster)

        # Calculate the 1-sigma Poisson errors for the AGN observed.
        cluster_err = small_poisson(cluster_counts, s=1)

        # Using the area enclosed by our aperture calculate the surface density.
        cluster_surf_den = cluster_counts / 1.

        # Also convert our errors to surface densities.
        cluster_surf_den_err_upper = cluster_err[0] / 1.
        cluster_surf_den_err_lower = cluster_err[1] / 1.

        cluster_dict.update({'obs_surf_den': [cluster_surf_den],
                             'obs_upper_surf_den_err': [cluster_surf_den_err_upper],
                             'obs_lower_surf_den_err': [cluster_surf_den_err_lower]})
        total_surf_den.append(Table(cluster_dict))

    # Combine all cluster tables into a single table to return
    surf_den_table = vstack(total_surf_den)

    # Calculate the variance of the errors.
    surf_den_table['obs_surf_den_var'] = (surf_den_table['obs_upper_surf_den_err']
                                          * surf_den_table['obs_lower_surf_den_err'])

    return surf_den_table


# Set our log-likelihood
def lnlike(param, catalog, obs_surf_den_table):
    # Extract our parameters
    eta, beta, zeta = param

    # Convert our catalog into a QTable so units are handled correctly.
    qcatalog = QTable(catalog)

    catalog_grp = catalog.group_by('SPT_ID')

    # For each cluster determine the model value and assign it to the correct observational value
    model_tables = []
    for cluster in catalog_grp.groups:
        # Calculate the model value for the cluster
        model_value = ((1 + cluster['REDSHIFT'][0])**eta
                       * ((cluster['M500'][0] * cluster['M500'].unit) / (1e15 * u.Msun))**zeta
                       * np.sum((cluster['r_r500_radial'])**beta))

        # Store the model values in a table.
        cluster_dict = {'SPT_ID': [cluster['SPT_ID'][0]], 'model_values': [model_value]}
        model_tables.append(Table(cluster_dict))

    # Combine all the model tables into a single table.
    model_table = vstack(model_tables)

    # Join the observed and model tables together based on the SPT_ID keys
    joint_table = join(obs_surf_den_table, model_table, keys='SPT_ID')

    # Our likelihood is then the chi-squared likelihood.
    total_lnlike = -0.5 * np.sum((joint_table['obs_surf_den'] - joint_table['model_values'])**2
                                 / joint_table['obs_surf_den_var'])

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(param):
    # Extract our parameters
    eta, beta, zeta = param

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
    if -3. <= eta <= 3. and -3. <= beta <= 3. and -3. <= zeta <= 3.:
        eta_lnprior = 0.0
        beta_lnprior = 0.0
        zeta_lnprior = 0.0
    else:
        eta_lnprior = -np.inf
        beta_lnprior = -np.inf
        zeta_lnprior = -np.inf

    # C_lnprior = -0.5 * np.sum((C - h_C)**2 / h_C_err**2)

    # Assuming all parameters are independent the joint log-prior is
    total_lnprior = eta_lnprior + beta_lnprior + zeta_lnprior #+ C_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(param, catalog, x):
    lp = lnprior(param)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(param, catalog, x)


# Read in the mock catalog
mock_catalog = Table.read('Data/MCMC/mock_AGN_catalog.cat', format='ascii')
mock_catalog['M500'].unit = u.Msun

# Calculate the "observed" surface density and variance
obs_cluster_surf_den = observed_surf_den(mock_catalog)

# For diagnostic purposes, set the values of the parameters.
eta_true = 1.2
beta_true = -1.5
zeta_true = -1.0

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 3
nwalkers = 28

# Also, set the number of steps to run the sampler for.
nsteps = 100

# We will initialize our walkers in a tight ball near the initial parameter values.
pos0 = emcee.utils.sample_ball(p0=[eta_true, beta_true, zeta_true], std=[1e-2, 1e-2, 1e-2], size=nwalkers)

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(mock_catalog, obs_cluster_surf_den))

# Run the sampler.
start_sampler_time = time()
sampler.run_mcmc(pos0, nsteps)
print('Sampler runtime: {:.2f} s'.format(time() - start_sampler_time))

# Plot the chains
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax1.plot(sampler.chain[:, :, 0].T, color='k', alpha=0.4)
ax1.axhline(eta_true, color='b')
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.set(ylabel=r'$\eta$', title='MCMC Chains')

ax2.plot(sampler.chain[:, :, 1].T, color='k', alpha=0.4)
ax2.axhline(beta_true, color='b')
ax2.yaxis.set_major_locator(MaxNLocator(5))
ax2.set(ylabel=r'$\beta$')

ax3.plot(sampler.chain[:, :, 2].T, color='k', alpha=0.4)
ax3.axhline(zeta_true, color='b')
ax3.yaxis.set_major_locator(MaxNLocator(5))
ax3.set(ylabel=r'$\zeta$')

# ax4.plot(sampler.chain[:, :, 3].T, color='k', alpha=0.4)
# ax4.yaxis.set_major_locator(MaxNLocator(5))
# ax4.set(ylabel=r'$C$', xlabel='Steps')

fig.savefig('Data/MCMC/Param_chains_mock_catalog.pdf', format='pdf')

# Remove the burnin, typically 1/3 number of steps
burnin = nsteps//3
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Produce the corner plot
fig = corner.corner(samples, labels=[r'$\eta$', r'$\beta$', r'$\zeta$'], truths=[eta_true, beta_true, zeta_true],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig('Data/MCMC/Corner_plot_mock_catalog.pdf', format='pdf')

eta_mcmc, beta_mcmc, zeta_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                     zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print("""MCMC Results:
eta = {eta[0]:.2f} +{eta[1]:.3f} -{eta[2]:.3f} (truth: {eta_true})
beta = {beta[0]:.2f} +{beta[1]:.3f} -{beta[2]:.3f} (truth: {beta_true})
zeta = {zeta[0]:.2f} +{zeta[1]:.3f} -{zeta[2]:.3f} (truth: {zeta_true})"""
      .format(eta=eta_mcmc, beta=beta_mcmc, zeta=zeta_mcmc,
              eta_true=eta_true, beta_true=beta_true, zeta_true=zeta_true))

print('Mean acceptance fraction: {}'.format(np.mean(sampler.acceptance_fraction)))
# print('Integrates autocorrelation time: {}'.format(sampler.get_autocorr_time()))
