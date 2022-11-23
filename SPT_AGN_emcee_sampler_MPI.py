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
from schwimmbad import MPIPool
from scipy.interpolate import lagrange, interp1d

from custom_math import trap_weight

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

seed = 3775
rng = np.random.default_rng(seed)

# Set up the luminosity and density evolution using the fits from Assef+11 Table 2
z_i = [0.25, 0.5, 1., 2., 4.]
m_star_z_i = [-23.51, -24.64, -26.10, -27.08]
phi_star_z_i = [-3.41, -3.73, -4.17, -4.65, -5.77]
m_star = lagrange(z_i[1:], m_star_z_i)
log_phi_star = lagrange(z_i, phi_star_z_i)


def luminosity_function(abs_mag, redshift):
    """
    Assef+11 QLF using luminosity and density evolution.

    Parameters
    ----------
    abs_mag : astropy.table.Table | array-like
        Rest-frame J-band absolute magnitude.
    redshift : astropy.table.Table | array-like
        Cluster redshift

    Returns
    -------
    Phi : np.ndarray
        Luminosity density

    """

    # L/L_*(z) = 10**(0.4 * (M_*(z) - M))
    L_L_star = 10 ** (0.4 * (m_star(redshift) - abs_mag))

    # Phi*(z) = 10**(log(Phi*(z))
    phi_star = 10 ** log_phi_star(redshift) * (cosmo.h / u.Mpc) ** 3

    # QLF slopes
    alpha1 = -3.35  # alpha in Table 2
    alpha2 = -0.37  # beta in Table 2

    Phi = 0.4 * np.log(10) * L_L_star * phi_star * (L_L_star ** -alpha1 + L_L_star ** -alpha2) ** -1

    return Phi


def model_rate_opted(params, cluster_id, r_r500, j_mag, integral=False):
    """
    Our generating model.

    Parameters
    ----------
    params : tuple
        Tuple of (theta, eta, zeta, beta, rc, C)
    cluster_id : str
        SPT ID of our cluster in the catalog dictionary
    r_r500 : array-like
        A vector of radii of objects within the cluster normalized by the cluster's r500
    j_mag : array-like
        A vector of J-band absolute magnitudes to be used in the luminosity function
    integral : bool, optional
        Flag indicating if the luminosity function factor of the model should be integrated. Defaults to `False`.

    Returns
    -------
    model: np.ndarray
        A surface density profile of objects as a function of radius and luminosity.
    """

    if args.cluster_only:
        # Unpack our parameters
        ln_theta, eta, zeta, beta, ln_rc = params
        theta, rc = np.exp([ln_theta, ln_rc])
        # Set background parameter to 0
        c0 = 0
    elif args.background_only:
        # Unpack our parameters
        c0, = params
        # Set all other parameters to neutral values
        theta, eta, zeta = [0.]*3
        beta, rc = 1/3., 1.
    else:
        # Unpack our parameters
        ln_theta, eta, zeta, beta, ln_rc, ln_c0 = params
    # eta, zeta, beta, ln_rc, ln_c0 = params

    # Exponentiate all the log-sampled parameters
        theta, rc, c0 = np.exp([ln_theta, ln_rc, ln_c0])
    # rc, c0 = np.exp([ln_rc, ln_c0])
    # theta = 50.

    # Extract our data from the catalog dictionary
    z = catalog_dict[cluster_id]['redshift']
    m = catalog_dict[cluster_id]['m500']
    r500 = catalog_dict[cluster_id]['r500']

    # Luminosity function number
    if integral:
        lum_funct_value = np.trapz(luminosity_function(j_mag, z), j_mag)
    else:
        lum_funct_value = luminosity_function(j_mag, z)

    # if args.no_luminosity or args.poisson_only:
    #     LF = 1
    # else:
    LF = cosmo.angular_diameter_distance(z) ** 2 * r500 * lum_funct_value

    # Convert our background surface density from angular units into units of r500^-2
    if args.cluster_only:
        cz = 0.
    else:
        cz = (((c0 + delta_c(z)) / u.arcmin ** 2)
              * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc) ** 2 * r500 ** 2)

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta * LF

    model = a * (1 + (r_r500 / rc) ** 2) ** (-1.5 * beta + 0.5) + cz

    return model.value


# Set our log-likelihood
def lnlike(param: tuple[float, ...]):
    lnlike_list = []
    for cluster_id in catalog_dict:
        # Get the good pixel fraction for this cluster
        gpf_all = catalog_dict[cluster_id]['gpf_rall']

        # Get the radial positions of the AGN
        ri = catalog_dict[cluster_id]['radial_r500_maxr']

        # Get the completeness weights for the AGN
        completeness_weight_maxr = catalog_dict[cluster_id]['completeness_weight_maxr']

        # Get the AGN sample degrees of membership
        # if args.no_selection_membership or args.poisson_only:
        #     agn_membership = 1
        # else:
        #     agn_membership = catalog_dict[cluster_id]['agn_membership_maxr']

        # Get the J-band absolute magnitudes
        # j_band_abs_mag = catalog_dict[cluster_id]['j_abs_mag']

        # Get the radial mesh for integration
        rall = catalog_dict[cluster_id]['rall']

        # Get the luminosity mesh for integration
        jall = catalog_dict[cluster_id]['jall']

        # Compute the completeness ratio for this cluster
        if args.no_completeness or args.poisson_only:
            completeness_ratio = 1.
        else:
            completeness_ratio = len(completeness_weight_maxr) / np.sum(completeness_weight_maxr)

        # Compute the model rate at the locations of the AGN.
        ni = model_rate_opted(param, cluster_id, ri, jall, integral=True)

        # Compute the full model along the radial direction.
        # The completeness weight is set to `1` as the model in the integration is assumed to be complete.
        n_mesh = model_rate_opted(param, cluster_id, rall, jall, integral=True)

        # Use a spatial poisson point-process log-likelihood
        cluster_lnlike = (np.sum(np.log(ni * ri)) - completeness_ratio * trap_weight(n_mesh * 2 * np.pi * rall, rall, weight=gpf_all))

        lnlike_list.append(cluster_lnlike)

    total_lnlike = np.sum(lnlike_list)

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(params: tuple[float, ...]):
    # Extract our parameters
    if args.cluster_only:
        ln_theta, eta, zeta, beta, ln_rc = params
        theta, rc = np.exp([ln_theta, ln_rc])
        c0 = c0_true
    elif args.background_only:
        ln_c0, = params
        c0 = np.exp(ln_c0)
        # Set other parameters to values that will pass the priors.
        theta, eta, zeta, beta = [0.]*4
        rc = 0.1  # Cannot be 0., min rc = 0.05
    else:
        ln_theta, eta, zeta, beta, ln_rc, ln_c0 = params
    # eta, zeta, beta, ln_rc, ln_c0 = params

    # Exponentiate all the log-sampled parameters
        theta, rc, c0 = np.exp([ln_theta, ln_rc, ln_c0])
    # rc, c0 = np.exp([ln_rc, ln_c0])
    # theta = 50.

    cluster_lnprior = []
    for cluster_id in catalog_dict:
        # Get the cluster redshift to set the background hyperparameters
        z = catalog_dict[cluster_id]['redshift']
        h_c = agn_prior_surf_den(z)
        h_c_err = agn_prior_surf_den_err(z)

        # Shift background parameter to redshift-dependent value.
        cz = c0 + delta_c(z)

        # Define all priors
        if (0.0 <= theta <= np.inf and
                -6. <= eta <= 6. and
                -3. <= zeta <= 3. and
                -3. <= beta <= 3. and
                0.05 <= rc <= 0.5 and
                0.0 <= cz < np.inf):
            theta_lnprior = 0.0
            eta_lnprior = 0.0
            zeta_lnprior = 0.0
            beta_lnprior = 0.0
            rc_lnprior = 0.0
            if args.cluster_only:
                c_lnprior = 0.
            else:
                c_lnprior = -0.5 * np.sum((cz - h_c) ** 2 / h_c_err ** 2)
            # C_lnprior = 0.0
        else:
            theta_lnprior = -np.inf
            eta_lnprior = -np.inf
            zeta_lnprior = -np.inf
            beta_lnprior = -np.inf
            rc_lnprior = -np.inf
            c_lnprior = -np.inf

        ln_prior_prob = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior + rc_lnprior + c_lnprior
        cluster_lnprior.append(ln_prior_prob)

    # Assuming all parameters are independent the joint log-prior is
    total_lnprior = np.sum(cluster_lnprior)

    return total_lnprior


# Define the log-posterior probability
def lnpost(params: tuple[float, ...]):
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
parser.add_argument('name', help='Chain name', type=str)
parser.add_argument('--preprocessing', help='Preprocessing file name', default='SPTcl_IRAGN_preprocessing.json', type=str)
parser.add_argument('--no-luminosity', action='store_true', help='Deactivate luminosity dependence in model.')
parser.add_argument('--no-selection-membership', action='store_true',
                    help='Deactivate fuzzy degree of membership for AGN selection in likelihood function.')
parser.add_argument('--no-completeness', action='store_true',
                    help='Deactivate photometric completeness correction in likelihood function.')
parser.add_argument('--poisson-only', action='store_true',
                    help='Use a pure Poisson likelihood function with a model that has no luminosity dependence.')
parser_grp = parser.add_mutually_exclusive_group()
parser_grp.add_argument('--cluster-only', action='store_true',
                        help='Sample only on cluster objects.')
parser_grp.add_argument('--background-only', action='store_true',
                        help='Sample only on background objects.')
args = parser.parse_args()

# Load in the prepossessing file
# local_dir = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Port_Rebuild_Tests/pure_poisson/'
local_dir = ''
# preprocess_file = os.path.abspath(f'{local_dir}SPTcl_IRAGN_preprocessing_fullMasks_withGPF_withLF_2kdenseJall_100cl.json')
preprocess_file = os.path.abspath(f'{local_dir}{args.preprocessing}')
with open(preprocess_file, 'r') as f:
    catalog_dict = json.load(f)

# Read in the purity and surface density files
with (open(f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
           f'SDWFS_purity_color_4.5_17.48.json', 'r') as f,
      open(f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
           f'SDWFS_background_prior_distributions_mu_cut_updated_cuts.json', 'r') as g):
    sdwfs_purity_data = json.load(f)
    sdwfs_prior_data = json.load(g)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
threshold_bins = sdwfs_prior_data['color_thresholds'][:-1]

# Set up interpolators
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')
agn_surf_den = interp1d(threshold_bins, sdwfs_prior_data['agn_surf_den'], kind='previous')
agn_surf_den_err = interp1d(threshold_bins, sdwfs_prior_data['agn_surf_den_err'], kind='previous')


# For convenience, set up the function compositions
def agn_prior_surf_den(redshift: float) -> float:
    return agn_surf_den(agn_purity_color(redshift))


def agn_prior_surf_den_err(redshift: float) -> float:
    return agn_surf_den_err(agn_purity_color(redshift))


# Set up an interpolation for the AGN surface density relative to the reference surface density at z = 0
delta_c = interp1d(z_bins, agn_prior_surf_den(z_bins) - agn_prior_surf_den(0.), kind='previous')

# Go through the catalog dictionary and recasting the cluster's mass and r500 to quantities and recast all the list-type
# data to numpy arrays
for cluster_id, cluster_info in catalog_dict.items():
    catalog_dict[cluster_id]['m500'] = cluster_info['m500'] * u.Msun
    catalog_dict[cluster_id]['r500'] = cluster_info['r500'] * u.Mpc
    for data_name, data in filter(lambda x: isinstance(x[1], list), cluster_info.items()):
        catalog_dict[cluster_id][data_name] = np.array(data)

theta_true = 5.0
eta_true = 4.0
zeta_true = -1.0
beta_true = 1.0
rc_true = 0.1
c0_true = agn_prior_surf_den(0.)

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 5 if args.cluster_only else (1 if args.background_only else 6)
# nwalkers = 6 * ndim
# ndim = 6
nwalkers = 50

# Also, set the number of steps to run the sampler for.
nsteps = int(1e6)

# We will initialize our walkers in a tight ball near the initial parameter values.
if args.cluster_only:
    pos0 = np.array([
        rng.uniform(low=0.,  high=15., size=nwalkers),
        rng.uniform(low=-6., high=6., size=nwalkers),
        rng.uniform(low=-3., high=3., size=nwalkers),
        rng.uniform(low=-1., high=1., size=nwalkers),
        rng.uniform(low=0.05, high=0.5, size=nwalkers)]).T
elif args.background_only:
    pos0 = np.array([rng.normal(np.log(c0_true), 1e-4, size=nwalkers)]).T
else:
    pos0 = np.array([
        rng.uniform(low=0.,  high=15., size=nwalkers),
        rng.uniform(low=-6., high=6., size=nwalkers),
        rng.uniform(low=-3., high=3., size=nwalkers),
        rng.uniform(low=-1., high=1., size=nwalkers),
        rng.uniform(low=0.05,  high=0.5, size=nwalkers),
        rng.normal(np.log(c0_true), 1e-4, size=nwalkers)]).T

# Set up the autocorrelation and convergence variables
autocorr = np.empty(nsteps)
old_tau = np.inf  # For convergence

with MPIPool() as pool:
    # Filename for hd5 backend
    # chain_file = f'{local_dir}emcee_mock_eta-zeta_grid.h5'
    chain_file = f'{local_dir}emcee_mock_eta-zeta_grid_308cl_snr13.h5'
    # chain_file = f'{local_dir}emcee_SPTcl-IRAGN_empirical.h5'
    backend = emcee.backends.HDFBackend(chain_file, name=f'{args.name}'
                                                         f'{"_no-LF" if args.no_luminosity else ""}'
                                                         f'{"_no-mu" if args.no_selection_membership else ""}'
                                                         f'{"_no-comp_corr" if args.no_completeness else ""}'
                                                         f'{"_poisson-only" if args.poisson_only else ""}')
    if not args.restart:
        backend.reset(nwalkers, ndim)

    # Stretch move proposal. Manually specified to tune the `a` parameter.
    # moves = emcee.moves.StretchMove(a=2.75)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend=backend, pool=pool)

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
            print(f'Chains have converged. Ending sampler early.\nIteration stopped at: {sampler.iteration}')
            break
        old_tau = tau

print(f'Sampler runtime: {time() - start_sampler_time:.2f} s')
