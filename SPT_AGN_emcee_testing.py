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
import scipy.optimize as op
from astropy.table import Table, vstack
from matplotlib.ticker import MaxNLocator
from small_poisson import small_poisson

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)


def data_matrix_gen(catalog):
    # Group the catalog by cluster
    catalog_grp = catalog.group_by('SPT_ID')

    # Initialize the matrix filled with zeros
    data_mat = np.zeros((len(catalog_grp.groups.keys), len(catalog)))

    # Populate the matrix
    for i in range(len(catalog_grp.groups.keys)):
        for j in range(len(catalog_grp.groups[i])):
            agn_radius = catalog_grp.groups[i]['r_r500_radial'][j]
            # Place the AGN's radial position in the appropriate row in the matrix
            data_mat[i,j] = agn_radius

    return data_mat


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
def lnlike(param, inp_catalog, obs_surf_den_table=None):
    # Extract our parameters
    eta, zeta, beta = param

    # Make a copy of the catalog
    catalog = inp_catalog.copy()

    # Calculate the maximum possible model value for normalization
    max_model = lambda data: -(1+data[0])**eta * (data[1]/1e15)**zeta * (data[2])**beta

    res = op.minimize(max_model, x0=np.array([0., 0., 0.]), bounds=[(0.5, 1.7), (0.2e15, 1.8e15), (0.1, 1.5)])

    # Our normalization value
    max_model_val = -res.fun

    # Compute the model probabilities
    model = 1. / max_model_val * ((1 + catalog['REDSHIFT'])**eta
                                  * (catalog['M500'] / 1e15)**zeta
                                  * catalog['r_r500_radial']**beta)

    # Add the model probabilities to the catalog
    catalog.add_column(model, name='model_prob')

    # Use a spatial possion point-process log-likelihood
    total_lnlike = (np.sum(catalog['model_prob'] * catalog['r_r500_radial'])
                    - np.trapz(catalog['model_prob'] * 2*np.pi * catalog['r_r500_radial']))

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(param):
    # Extract our parameters
    eta, zeta, beta = param

    # # Set our hyperparameters
    # h_C = 0.371
    # h_C_err = 0.157

    # Define all priors to be gaussian
    if -3. <= eta <= 3. and -3. <= zeta <= 3. and -3 <= beta <= 3:
        eta_lnprior = 0.0
        beta_lnprior = 0.0
        zeta_lnprior = 0.0
    else:
        eta_lnprior = -np.inf
        beta_lnprior = -np.inf
        zeta_lnprior = -np.inf

    # C_lnprior = -0.5 * np.sum((C - h_C)**2 / h_C_err**2)

    # Assuming all parameters are independent the joint log-prior is
    total_lnprior = eta_lnprior + zeta_lnprior + beta_lnprior  #+ C_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(param, catalog, x):
    lp = lnprior(param)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(param, catalog, x)


# Read in the mock catalog
mock_catalog = Table.read('Data/MCMC/Mock_Catalog/Catalogs/mock_AGN_subcatalog00.cat', format='ascii')
mock_catalog['M500'].unit = u.Msun


# For diagnostic purposes, set the values of the parameters.
eta_true = 1.2
beta_true = -1.5
zeta_true = -1.0

# nlnlike = lambda *args: -lnlike(*args)
# mle = op.minimize(nlnlike, x0=np.array([0,0,0]), args=(mock_catalog, None))

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 3
nwalkers = 100

# Also, set the number of steps to run the sampler for.
nsteps = 500

# We will initialize our walkers in a tight ball near the initial parameter values.
pos0 = emcee.utils.sample_ball(p0=[eta_true, zeta_true, beta_true],
                               std=[1e-2, 1e-2, 1e-2], size=nwalkers)

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(mock_catalog, None), threads=4)

# Run the sampler.
start_sampler_time = time()
sampler.run_mcmc(pos0, nsteps)
print('Sampler runtime: {:.2f} s'.format(time() - start_sampler_time))
np.save('Data/MCMC/Mock_Catalog/Chains/emcee_run_w{nwalkers}_s{nsteps}_sc00'.format(nwalkers=nwalkers, nsteps=nsteps),
        sampler.chain)

# Plot the chains
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

ax1.plot(sampler.chain[:, :, 0].T, color='k', alpha=0.4)
ax1.axhline(eta_true, color='b')
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.set(ylabel=r'$\eta$', title='MCMC Chains')

ax2.plot(sampler.chain[:, :, 1].T, color='k', alpha=0.4)
ax2.axhline(zeta_true, color='b')
ax2.yaxis.set_major_locator(MaxNLocator(5))
ax2.set(ylabel=r'$\zeta$')

ax3.plot(sampler.chain[:, :, 2].T, color='k', alpha=0.4)
ax3.axhline(beta_true, color='b')
ax3.yaxis.set_major_locator(MaxNLocator(5))
ax3.set(ylabel=r'$\beta$', xlabel='Steps')

fig.savefig('Data/MCMC/Mock_Catalog/Plots/Param_chains_mock_catalog_sc00.pdf', format='pdf')

# Remove the burnin, typically 1/3 number of steps
burnin = nsteps//3
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Produce the corner plot
fig = corner.corner(samples, labels=[r'$\eta$', r'$\zeta$', r'$\beta$'],
                    truths=[eta_true, zeta_true, beta_true],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Corner_plot_mock_catalog_sc00.pdf', format='pdf')

eta_mcmc, zeta_mcmc, beta_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                 zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print("""MCMC Results:
eta = {eta[0]:.2f} +{eta[1]:.3f} -{eta[2]:.3f} (truth: {eta_true})
zeta = {zeta[0]:.2f} +{zeta[1]:.3f} -{zeta[2]:.3f} (truth: {zeta_true})
beta = {beta[0]:.2f} +{beta[1]:.3f} -{beta[2]:.3f} (truth: {beta_true})"""
      .format(eta=eta_mcmc,  zeta=zeta_mcmc, beta=beta_mcmc,
              eta_true=eta_true,  zeta_true=zeta_true, beta_true=beta_true))

print('Mean acceptance fraction: {}'.format(np.mean(sampler.acceptance_fraction)))
# print('Integrates autocorrelation time: {}'.format(sampler.get_autocorr_time()))
