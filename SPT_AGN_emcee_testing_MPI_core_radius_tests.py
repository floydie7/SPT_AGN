"""
SPT_AGN_emcee.py
Author: Benjamin Floyd

This script will preform the Bayesian analysis on the SPT-AGN data to produce the posterior probability distributions
for all fitting parameters.
"""
import json
import sys
from time import time

import astropy.units as u
import emcee
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from schwimmbad import MPIPool

from custom_math import trap_weight  # Custom trapezoidal integration

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
    # theta, eta, zeta, beta, rc, C = params
    theta, eta, zeta, beta, C = params

    # Extract our data from the catalog dictionary
    z = catalog_dict[cluster_id]['redshift']
    m = catalog_dict[cluster_id]['m500']
    r500 = catalog_dict[cluster_id]['r500']

    # Convert our background surface density from angular units into units of r500^-2
    background = C_true / u.arcmin ** 2 * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc) ** 2 * r500 ** 2

    # The cluster's core radius in units of r500
    rc_r500 = rc_true * u.Mpc / r500

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc_r500) ** 2) ** (-1.5 * beta + 0.5) + background

    return model.value


# Set our log-likelihood
def lnlike(param):
    lnlike_list = []
    for cluster_id in catalog_dict:
        # Get the good pixel fraction for this cluster
        gpf_all = catalog_dict[cluster_id]['gpf_rall']

        # Get the radial positions of the AGN
        radial_r500 = catalog_dict[cluster_id]['radial_r500']

        # Get the radial mesh for integration
        rall = catalog_dict[cluster_id]['rall']

        # Select only the objects within the same radial limit we are using for integration.
        radial_r500_maxr = radial_r500[radial_r500 <= rall[-1]]

        # Compute the model rate at the locations of the AGN.
        ni = model_rate_opted(param, cluster_id, radial_r500_maxr)

        # Compute the full model along the radial direction
        nall = model_rate_opted(param, cluster_id, rall)

        # Use a spatial poisson point-process log-likelihood
        cluster_lnlike = np.sum(np.log(ni * radial_r500_maxr)) - trap_weight(nall * 2 * np.pi * rall, rall,
                                                                             weight=gpf_all)
        lnlike_list.append(cluster_lnlike)

    total_lnlike = np.sum(lnlike_list)

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(param):
    # Extract our parameters
    # theta, eta, zeta, beta, rc, C = param
    theta, eta, zeta, beta, C = param

    # Set our hyperparameters
    h_rc = 0.25
    h_rc_err = 0.1
    h_C = 0.371
    h_C_err = 0.157
    h_theta = theta_true
    h_theta_err = 0.06

    # Narrow the prior on theta to bracket +- 50% of theta_true
    # theta_lower = theta_true - theta_true * 0.5
    # theta_upper = theta_true + theta_true * 0.5

    # Define all priors to be gaussian
    # if 0.0 <= theta <= 1.0 and -3. <= eta <= 3. and -3. <= zeta <= 3. and -3. <= beta <= 3. and 0.0 <= C < np.inf \
    #         and 0.1 <= rc <= 0.5:
    if 0.0 <= theta <= 1.0 and -3. <= eta <= 3. and -3. <= zeta <= 3. and -3. <= beta <= 3. and 0.0 <= C < np.inf:
        theta_lnprior = 0.0
        # theta_lnprior = -0.5 * (np.log(theta) - np.log(h_theta))**2 / h_theta_err**2
        eta_lnprior = 0.0
        beta_lnprior = 0.0
        zeta_lnprior = 0.0
        # rc_lnprior = -0.5 * np.sum((rc - h_rc) ** 2 / h_rc_err ** 2)
        # rc_lnprior = 0.0
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
    # total_lnprior = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior + rc_lnprior + C_lnprior
    total_lnprior = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior + C_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(param):
    lp = lnprior(param)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(param)


hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
# hcc_prefix = ''
# Read in the mock catalog
mock_catalog = Table.read(hcc_prefix + 'Data/MCMC/Mock_Catalog/Catalogs/Final_tests/core_radius_tests/trial_7/'
                                       'mock_AGN_catalog_t0.098_e1.20_z-1.00_b0.50_C0.371_rc0.250'
                                       '_maxr5.00_clseed890_objseed930_core_radius.cat',
                          format='ascii')

# Read in the mask files for each cluster
mock_catalog_grp = mock_catalog.group_by('SPT_ID')

# Set parameter values
theta_true = 0.098  # Amplitude.
eta_true = 1.2  # Redshift slope
beta_true = 0.5  # Radial slope
zeta_true = -1.0  # Mass slope
rc_true = 0.2  # Core radius (in Mpc)
C_true = 0.371  # Background AGN surface density

# Load in the prepossessing file
preprocess_file = hcc_prefix + 'Data/MCMC/Mock_Catalog/Catalogs/Final_tests/core_radius_tests/trial_7/core_radius_preprocessing.json'
with open(preprocess_file, 'r') as f:
    catalog_dict = json.load(f)

# Go through the catalog dictionary and add the radial positions of all the AGN and add units to the mass and r500 radius
for cluster_id, cluster_info in catalog_dict.items():
    catalog_dict[cluster_id]['m500'] = cluster_info['m500'] * u.Msun
    catalog_dict[cluster_id]['r500'] = cluster_info['r500'] * u.Mpc
    group_mask = mock_catalog_grp.groups.keys['SPT_ID'] == cluster_id
    catalog_dict[cluster_id]['radial_r500'] = mock_catalog_grp.groups[group_mask]['radial_r500']

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 5
nwalkers = 36

# Also, set the number of steps to run the sampler for.
nsteps = int(1e6)

# We will initialize our walkers in a tight ball near the initial parameter values.
# pos0 = emcee.utils.sample_ball(p0=[theta_true, eta_true, zeta_true, beta_true, C_true],
#                                std=[1e-2, 1e-2, 1e-2, 1e-2, 0.157], size=nwalkers)
pos0 = np.vstack([[np.random.uniform(0., 2.),  # theta
                   np.random.uniform(-3., 3.),  # eta
                   np.random.uniform(-3., 3.),  # zeta
                   np.random.uniform(-3., 3.),  # beta
                   # np.random.lognormal(mean=np.log(0.35), sigma=0.06),  # rc
                   np.random.uniform(0., 1.),  # rc
                   np.random.normal(loc=0.371, scale=0.157)]  # C
                  for i in range(nwalkers)])

# Set up the autocorrelation and convergence variables
index = 0
autocorr = np.empty(nsteps)
old_tau = np.inf  # For convergence

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Filename for hd5 backend
    chain_file = 'emcee_run_w{nwalkers}_s{nsteps}_mock_t{theta}_e{eta}_z{zeta}_b{beta}_rc_{rc}_C{C}_core_radius_tests.h5' \
        .format(nwalkers=nwalkers, nsteps=nsteps,
                theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, rc=rc_true, C=C_true)
    backend = emcee.backends.HDFBackend(chain_file,
                                        name='core_radius_new_rc_gen_trial7.3_fixed_rc{rc}'.format(rc=rc_true))
    backend.reset(nwalkers, ndim)

    # Stretch move proposal. Manually specified to tune the `a` parameter.
    moves = emcee.moves.StretchMove(a=2.75)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend=backend, moves=moves, pool=pool)

    # Run the sampler.
    print('Starting sampler.')
    start_sampler_time = time()

    # Sample up to nsteps.
    for sample in sampler.sample(pos0, iterations=nsteps):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far.
        # Using tol = 0 means we will always get an estimate even if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print('Chains have converged. Ending sampler early.\nIteration stopped at: {}'.format(sampler.iteration))
            break
        old_tau = tau

print('Sampler runtime: {:.2f} s'.format(time() - start_sampler_time))

# Get the chain from the sampler
samples = sampler.get_chain()
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']
truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, C_true]

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
    print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f} (truth: {true})'
          .format(labels=labels[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0], true=truths[i]))

print('Mean acceptance fraction: {:.2f}'.format(np.mean(sampler.acceptance_fraction)))

# Get estimate of autocorrelation time
print('Autocorrelation time: {:.1f}'.format(tau))
