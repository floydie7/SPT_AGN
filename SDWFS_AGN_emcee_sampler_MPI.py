"""
SPT_AGN_emcee_sampler_MPI.py
Author: Benjamin Floyd

This script will preform the Bayesian analysis on the SPT-AGN data to produce the posterior probability distributions
for all fitting parameters.
"""
import json
import os
from argparse import ArgumentParser
from time import time

import emcee
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from custom_math import trap_weight  # Custom trapezoidal integration
from schwimmbad import MPIPool

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def model_rate_opted(params, r):
    """
    Our generating model

    Parameters
    ----------
    params : tuple of float
        Model parameters.

    Returns
    -------
    model : float
        A surface density profile of objects in the field of view
    """

    # Unpack our parameters
    a, C, sigma_C = params

    # Convert our background surface density from angular units into units of r500^-2
    background = a * np.exp(-0.5 * ((r - C) / sigma_C)**2)

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = background

    return model


# Set our log-likelihood
def lnlike(param):
    lnlike_list = []
    for cutout_id in catalog_dict:
        # Get the good pixel fraction for this cluster
        gpf_all = catalog_dict[cutout_id]['gpf_rall']

        # Get the radial positions of the AGN
        radial_arcmin_maxr = catalog_dict[cutout_id]['radial_arcmin_maxr']

        # Get the completeness weights for the AGN
        completeness_weight_maxr = catalog_dict[cutout_id]['completeness_weight_maxr']

        # Get the radial mesh for integration
        rall = catalog_dict[cutout_id]['rall']

        # Compute the completeness ratio for this cluster
        completeness_ratio = len(completeness_weight_maxr) / np.sum(completeness_weight_maxr)

        # Compute the model rate at the locations of the AGN.
        ni = model_rate_opted(param, radial_arcmin_maxr)

        # Compute the full model along the radial direction.
        # The completeness weight is set to `1` as the model in the integration is assumed to be complete.
        nall = model_rate_opted(param, rall)

        # Use a spatial poisson point-process log-likelihood
        cluster_lnlike = np.sum(np.log(ni * radial_arcmin_maxr)) - completeness_ratio * trap_weight(
            nall * 2 * np.pi * rall, rall, weight=gpf_all)
        lnlike_list.append(cluster_lnlike)

    total_lnlike = np.sum(lnlike_list)

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(params):
    # Extract our parameters
    a, C, sigma_C = params

    # Define all priors
    if 0.0 <= C < 5.0 and 0.0 <= a <= 100. and 0. <= sigma_C <= 5.:
        C_lnprior = 0.0
        sigma_C_lnprior = 0.0
        a_lnprior = 0.0
    else:
        C_lnprior = -np.inf
        sigma_C_lnprior = -np.inf
        a_lnprior = -np.inf

    # Assuming all parameters are independent the joint log-prior is
    total_lnprior = C_lnprior + sigma_C_lnprior + a_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(params):
    lp = lnprior(params)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(params)


hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
# hcc_prefix = ''

parser = ArgumentParser(description='Runs MCMC sampler')
parser.add_argument('--restart', help='Allows restarting the chain in place rather than resetting the chain.',
                    action='store_true')
args = parser.parse_args()

# Load in the prepossessing file
preprocess_file = os.path.abspath('SDWFS_IRAGN_preprocessing.json')
with open(preprocess_file, 'r') as f:
    catalog_dict = json.load(f)

# Go through the catalog dictionary and recasting the cluster's mass and r500 to quantities and recast the radial
# position and completeness lists to arrays.
for cutout_id, cluster_info in catalog_dict.items():
    catalog_dict[cutout_id]['gpf_rall'] = np.array(cluster_info['gpf_rall'])
    catalog_dict[cutout_id]['rall'] = np.array(cluster_info['rall'])
    catalog_dict[cutout_id]['radial_arcmin_maxr'] = np.array(cluster_info['radial_arcmin_maxr'])

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 3
nwalkers = 15

# Also, set the number of steps to run the sampler for.
nsteps = int(1e6)

# We will initialize our walkers in a tight ball near the initial parameter values.
pos0 = np.vstack([[np.random.uniform(0., 1.)  # C
                   ]
                  for i in range(nwalkers)])

# Set up the autocorrelation and convergence variables
autocorr = np.empty(nsteps)
old_tau = np.inf  # For convergence

with MPIPool() as pool:
    # if not pool.is_master():
    #     pool.wait()
    #     sys.exit(0)

    # Filename for hd5 backend
    chain_file = 'emcee_chains_SDWFS_IRAGN.h5'
    backend = emcee.backends.HDFBackend(chain_file, name='SDWFS_background_uniform_prior_variance')
    if not args.restart:
        backend.reset(nwalkers, ndim)

    # Stretch move proposal. Manually specified to tune the `a` parameter.
    moves = emcee.moves.StretchMove(a=2.75)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend=backend, moves=moves, pool=pool)

    # Run the sampler.
    print('Starting sampler.')
    start_sampler_time = time()

    # Sample up to nsteps.
    for index, sample in enumerate(sampler.sample(pos0, iterations=nsteps)):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far.
        # Using tol = 0 means we will always get an estimate even if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print('Chains have converged. Ending sampler early.\nIteration stopped at: {}'.format(sampler.iteration))
            break
        old_tau = tau

print('Sampler runtime: {:.2f} s'.format(time() - start_sampler_time))

# Get the chain from the sampler
labels = [r'$a$', r'$C$', r'$\sigma_C$']

try:
    # Calculate the autocorrelation time
    tau_est = sampler.get_autocorr_time()

    tau = np.mean(tau_est)

    # Remove the burn-in. We'll use ~3x the autocorrelation time
    burnin = int(3 * tau)

    # We will also thin by roughly half our autocorrelation time
    thinning = int(tau // 2)

except emcee.autocorr.AutocorrError:
    tau_est = sampler.get_autocorr_time(quiet=True)
    tau = np.mean(tau_est)

    burnin = int(sampler.iteration // 3)
    thinning = 1

flat_samples = sampler.get_chain(discard=burnin, thin=thinning, flat=True)

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f}'
          .format(labels=labels[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0]))

print('Mean acceptance fraction: {:.2f}'.format(np.mean(sampler.acceptance_fraction)))

# Get estimate of autocorrelation time
print('Autocorrelation time: {:.1f}'.format(tau))
