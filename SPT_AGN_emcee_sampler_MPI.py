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

import astropy.units as u
import emcee
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from custom_math import trap_weight  # Custom trapezoidal integration
from schwimmbad import MPIPool

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def model_rate_opted(params, cluster_id, r_r500):
    """
    Our generating model.

    :param params: Tuple of (theta, eta, zeta, beta, background)
    :param cluster_id: SPT ID of our cluster in the catalog dictionary
    :param r_r500: A vector of radii of objects within the cluster normalized by the cluster's r500
    :return model: A surface density profile of objects as a function of radius
    """

    # Unpack our parameters
    theta, eta, zeta, beta, rc, C = params
    # theta, eta, zeta, beta, rc = params
    # C = 0.371  # Fixed background

    # Extract our data from the catalog dictionary
    z = catalog_dict[cluster_id]['redshift']
    m = catalog_dict[cluster_id]['m500']
    r500 = catalog_dict[cluster_id]['r500']

    # Convert our background surface density from angular units into units of r500^-2
    background = C / u.arcmin ** 2 * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc) ** 2 * r500 ** 2

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc) ** 2) ** (-1.5 * beta + 0.5) + background

    return model.value


# Set our log-likelihood
def lnlike(param):
    lnlike_list = []
    for cluster_id in catalog_dict:
        # Get the good pixel fraction for this cluster
        gpf_all = catalog_dict[cluster_id]['gpf_rall']

        # Get the radial positions of the AGN
        radial_r500_maxr = catalog_dict[cluster_id]['radial_r500_maxr']

        # Get the completeness weights for the AGN
        completeness_weight_maxr = catalog_dict[cluster_id]['completeness_weight_maxr']

        # Get the AGN sample degrees of membership
        agn_membership = catalog_dict[cluster_id]['agn_membership_maxr']

        # Get the radial mesh for integration
        rall = catalog_dict[cluster_id]['rall']

        # Compute the completeness ratio for this cluster
        completeness_ratio = len(completeness_weight_maxr) / np.sum(completeness_weight_maxr)

        # Compute the joint probability of AGN sample membership
        membership_degree = np.prod(agn_membership)

        # Compute the model rate at the locations of the AGN.
        ni = model_rate_opted(param, cluster_id, radial_r500_maxr)

        # Compute the full model along the radial direction.
        # The completeness weight is set to `1` as the model in the integration is assumed to be complete.
        nall = model_rate_opted(param, cluster_id, rall)

        # Use a spatial poisson point-process log-likelihood
        cluster_lnlike = np.sum(np.log(ni * radial_r500_maxr * agn_membership)) \
                         - completeness_ratio * membership_degree * trap_weight(nall * 2 * np.pi * rall,
                                                                                rall, weight=gpf_all)
        lnlike_list.append(cluster_lnlike)

    total_lnlike = np.sum(lnlike_list)

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(params):
    # Extract our parameters
    theta, eta, zeta, beta, rc, C = params
    # theta, eta, zeta, beta, rc = params

    # Set our hyperparameters
    # h_rc = 0.25
    # h_rc_err = 0.1
    h_C = 0.371
    h_C_err = 0.157 * args.prior_scale_factor  # Artificially scaling the background prior

    # Define all priors
    if (0.0 <= theta <= np.inf and
            -6. <= eta <= 6. and
            -3. <= zeta <= 3. and
            -3. <= beta <= 3. and
            0.0 <= C < np.inf and
            # np.isclose(C, h_C) and
            0.05 <= rc <= 0.5):
        theta_lnprior = 0.0
        eta_lnprior = 0.0
        beta_lnprior = 0.0
        zeta_lnprior = 0.0
        # rc_lnprior = -0.5 * np.sum((rc - h_rc) ** 2 / h_rc_err ** 2)
        rc_lnprior = 0.0
        C_lnprior = -0.5 * np.sum((C - h_C) ** 2 / h_C_err ** 2)
        # C_lnprior = 0.0
    else:
        theta_lnprior = -np.inf
        eta_lnprior = -np.inf
        beta_lnprior = -np.inf
        zeta_lnprior = -np.inf
        rc_lnprior = -np.inf
        C_lnprior = -np.inf

    # Assuming all parameters are independent the joint log-prior is
    total_lnprior = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior + rc_lnprior + C_lnprior
    # total_lnprior = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior + rc_lnprior

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
parser.add_argument('--prior_scale_factor', help='Scale factor to the standard deviation of the background prior.',
                    default=1.0, type=float)
args = parser.parse_args()

# # Set parameter values
# theta_true = id_params[0]  # Amplitude.
# eta_true = id_params[1]  # Redshift slope
# zeta_true = id_params[2]  # Mass slope
# beta_true = 1.0  # Radial slope
# rc_true = 0.1  # Core radius (in r500)
# C_true = 0.371  # Background AGN surface density

# Load in the prepossessing file
preprocess_file = os.path.abspath('SPTcl_IRAGN_preprocessing.json')
with open(preprocess_file, 'r') as f:
    catalog_dict = json.load(f)

# Go through the catalog dictionary and recasting the cluster's mass and r500 to quantities and recast the radial
# position and completeness lists to arrays.
for cluster_id, cluster_info in catalog_dict.items():
    catalog_dict[cluster_id]['m500'] = cluster_info['m500'] * u.Msun
    catalog_dict[cluster_id]['r500'] = cluster_info['r500'] * u.Mpc
    catalog_dict[cluster_id]['gpf_rall'] = cluster_info['gpf_rall']
    catalog_dict[cluster_id]['radial_r500_maxr'] = np.array(cluster_info['radial_r500_maxr'])
    catalog_dict[cluster_id]['completeness_weight_maxr'] = cluster_info['completeness_weight_maxr']
    catalog_dict[cluster_id]['agn_membership_maxr'] = cluster_info['agn_membership_maxr']

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 6  # Fixed background
nwalkers = 36

# Also, set the number of steps to run the sampler for.
nsteps = int(1e6)

# We will initialize our walkers in a tight ball near the initial parameter values.
pos0 = np.vstack([[np.random.uniform(0., 12.),  # theta
                   np.random.uniform(-1., 6.),  # eta
                   np.random.uniform(-3., 3.),  # zeta
                   np.random.uniform(-3., 3.),  # beta
                   np.random.normal(loc=0.1, scale=6e-3),  # rc
                   np.random.normal(loc=0.371, scale=0.157)  # C
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
    chain_file = 'emcee_chains_Mock_fuzzy_selection.h5'
    backend = emcee.backends.HDFBackend(chain_file, name=f'fuzzy_selection_mod_like_sum_term')
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
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']
# labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$']
# truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, C_true]

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
