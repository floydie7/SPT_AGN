"""
SPT_AGN_emcee_restart.py
Author: Benjamin Floyd

Restarts the MCMC chains.
"""

import os
from itertools import product
from multiprocessing import Pool, cpu_count
from time import time

import astropy.units as u
import emcee
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from custom_math import trap_weight  # Custom trapezoidal integration
from scipy.spatial.distance import cdist

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def good_pixel_fraction(r, z, r500, center, cluster_id):
    # Read in the mask file and the mask file's WCS
    image, header = mask_dict[cluster_id]  # This is provided by the global variable mask_dict
    image_wcs = WCS(header)

    # From the WCS get the pixel scale
    pix_scale = (image_wcs.pixel_scale_matrix[1, 1] * u.deg).to(u.arcsec)

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

    # Convert our radius to pixels
    r_pix = r * r500 * cosmo.arcsec_per_kpc_proper(z).to(u.arcsec / u.Mpc) / pix_scale
    r_pix = r_pix.value

    # Because we potentially integrate to larger radii than can be fit on the image we will need to increase the size of
    # our mask. To do this, we will pad the mask with a zeros out to the radius we need.
    # Find the width needed to pad the image to include the largest radius inside the image.
    width = ((int(round(np.max(r_pix) - center_pix[1])),
              int(round(np.max(r_pix) - (image.shape[0] - center_pix[1])))),
             (int(round(np.max(r_pix) - center_pix[0])),
              int(round(np.max(r_pix) - (image.shape[1] - center_pix[0])))))

    # Insure that we are adding a non-negative padding width.
    width = tuple(tuple([i if i >= 0 else 0 for i in axis]) for axis in width)

    large_image = np.pad(image, pad_width=width, mode='constant', constant_values=0)

    # find the distances from center pixel to all other pixels
    image_coords = np.array(list(product(range(large_image.shape[0]), range(large_image.shape[1]))))

    # The center pixel's coordinate needs to be transformed into the large image system
    center_coord = np.array(center_pix) + np.array([width[1][0], width[0][0]])
    center_coord = center_coord.reshape((1, 2))

    # Compute the distance matrix. The entries are a_ij = sqrt((x_j - cent_x)^2 + (y_i - cent_y)^2)
    image_dists = cdist(image_coords, np.flip(center_coord)).reshape(large_image.shape)

    # select all pixels that are within the annulus
    good_pix_frac = []
    for j in np.arange(len(r_pix) - 1):
        pix_ring = large_image[np.where((r_pix[j] <= image_dists) & (image_dists < r_pix[j + 1]))]

        # Calculate the fraction
        good_pix_frac.append(np.sum(pix_ring) / len(pix_ring))

    return good_pix_frac


def model_rate_opted(params, cluster_id, r_r500):
    """
    Our generating model.

    :param params: Tuple of (theta, eta, zeta, beta, background)
    :param cluster_id: SPT ID of our cluster in the catalog dictionary
    :param r_r500: A vector of radii of objects within the cluster normalized by the cluster's r500
    :return model: A surface density profile of objects as a function of radius
    """

    # Unpack our parameters
    theta, eta, zeta, beta, C = params

    # Extract our data from the catalog dictionary
    z = catalog_dict[cluster_id]['redshift']
    m = catalog_dict[cluster_id]['m500']
    r500 = catalog_dict[cluster_id]['r500']

    # theta = theta / u.Mpc**2 * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc/u.arcmin)**2

    # Convert our background surface density from angular units into units of r500^-2
    background = C / u.arcmin ** 2 * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc) ** 2 * r500 ** 2

    # The cluster's core radius in units of r500
    rc_r500 = 0.1 * u.Mpc / r500

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc_r500) ** 2) ** (-1.5 * beta + 0.5) + background

    return model.value


# Set our log-likelihood
def lnlike(param):

    lnlike_list = []
    for cluster_id in catalog_dict.keys():
        # Get the good pixel fraction for this cluster
        gpf_all = catalog_dict[cluster_id]['gpf_rall']

        # Get the radial positions of the AGN
        radial_r500 = catalog_dict[cluster_id]['radial_r500']

        # Get the radial mesh for integration
        rall = catalog_dict[cluster_id]['rall']

        # Compute the model rate at the locations of the AGN.
        ni = model_rate_opted(param, cluster_id, radial_r500)

        # Compute the full model along the radial direction
        nall = model_rate_opted(param, cluster_id, rall)

        # Use a spatial poisson point-process log-likelihood
        cluster_lnlike = (np.sum(np.log(ni * radial_r500)) - trap_weight(nall * 2*np.pi * rall, rall, weight=gpf_all))
        lnlike_list.append(cluster_lnlike)

    total_lnlike = np.sum(lnlike_list)

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(param):
    # Extract our parameters
    theta, eta, zeta, beta, C = param

    # Set our hyperparameters
    h_C = 0.371
    h_C_err = 0.157

    # Define all priors to be gaussian
    if 0. <= theta <= 24000. and -3. <= eta <= 3. and -3. <= zeta <= 3. and -3. <= beta <= 3. and 0 <= C <= np.inf:
        theta_lnprior = 0.0
        eta_lnprior = 0.0
        beta_lnprior = 0.0
        zeta_lnprior = 0.0
        C_lnprior = -0.5 * np.sum((C - h_C)**2 / h_C_err**2)
        # C_lnprior = 0.0
    else:
        theta_lnprior = -np.inf
        eta_lnprior = -np.inf
        beta_lnprior = -np.inf
        zeta_lnprior = -np.inf
        C_lnprior = -np.inf

    # Assuming all parameters are independent the joint log-prior is
    total_lnprior = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior + C_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(param):
    lp = lnprior(param)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(param)


tusker_prefix = '/work/mei/bfloyd/SPT_AGN/'
# hcc_prefix = ''
# Read in the mock catalog
mock_catalog = Table.read(tusker_prefix+'Data/MCMC/Mock_Catalog/Catalogs/pre-final_tests/'
                                        'mock_AGN_catalog_t12000.00_e1.20_z-1.00_b0.50_C0.371_maxr5.00_seed890'
                                        '_gpf_fixed_single_cluster.cat',
                          format='ascii')

# Read in the mask files for each cluster
mock_catalog_grp = mock_catalog.group_by('SPT_ID')
mask_dict = {cluster_id[0]: fits.getdata(tusker_prefix+mask_file, header=True) for cluster_id, mask_file
             in zip(mock_catalog_grp.groups.keys.as_array(),
                    mock_catalog_grp['MASK_NAME'][mock_catalog_grp.groups.indices[:-1]])}

# Set parameter values
theta_true = 12000    # Amplitude.
eta_true = 1.2       # Redshift slope
beta_true = 0.5      # Radial slope
zeta_true = -1.0     # Mass slope
C_true = 0.371       # Background AGN surface density

max_radius = 5.0  # Maximum integration radius in r500 units

# Compute the good pixel fractions for each cluster and store the array in the catalog.
print('Generating Good Pixel Fractions.')
start_gpf_time = time()
catalog_dict = {}
for cluster in mock_catalog_grp.groups:
    cluster_id = cluster['SPT_ID'][0]
    cluster_z = cluster['REDSHIFT'][0]
    cluster_m500 = cluster['M500'][0] * u.Msun
    cluster_r500 = cluster['r500'][0] * u.Mpc
    cluster_sz_cent = cluster['SZ_RA', 'SZ_DEC'][0]
    cluster_sz_cent = cluster_sz_cent.as_void()
    cluster_radial_r500 = cluster['radial_r500']

    # Generate a radial integration mesh
    rall = np.histogram_bin_edges(cluster_radial_r500, bins='auto')

    cluster_gpf_all = good_pixel_fraction(rall, cluster_z, cluster_r500, cluster_sz_cent, cluster_id)

    cluster_dict = {'redshift': cluster_z, 'm500': cluster_m500, 'r500': cluster_r500,
                    'radial_r500': cluster_radial_r500, 'gpf_rall': cluster_gpf_all, 'rall': rall}

    # Store the cluster dictionary in the master catalog dictionary
    catalog_dict[cluster_id] = cluster_dict
print('Time spent calculating GPFs: {:.2f}s'.format(time() - start_gpf_time))

# Set up our MCMC sampler.
# Set the number of dimensions for the parameter space and the number of walkers to use to explore the space.
ndim = 5
nwalkers = 30

# Also, set the number of steps to run the sampler for.
nsteps = 100000

# Set up the autocorrelation and convergence variables
index = 0
autocorr = np.empty(nsteps)
old_tau = np.inf  # For convergence

# Set up multiprocessing pool
# get number of cpus available to job
try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split('(')[0])
except KeyError:
    ncpus = cpu_count()

with Pool(processes=ncpus) as pool:
    # Filename for hd5 backend
    chain_file = tusker_prefix+'Data/MCMC/Mock_Catalog/Chains/pre-final_tests/' \
                 'emcee_run_w30_s100000_mock_t12000_e1.2_z-1.0_b0.5_C0.371_maxr5.0_gpf_fixed_single_cluster.h5'
    backend = emcee.backends.HDFBackend(chain_file)

    # Stretch move proposal. Manually specified to tune the `a` parameter.
    moves = emcee.moves.StretchMove(a=2.75)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend=backend, moves=moves, pool=pool)

    # Run the sampler.
    print('Starting sampler.')
    start_sampler_time = time()

    # Get the previous state so we can restart the chains
    restart_state = sampler._previous_state

    # Sample up to nsteps.
    for sample in sampler.sample(restart_state, iterations=nsteps):
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
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$C$']
truths = [theta_true, eta_true, zeta_true, beta_true, C_true]

# Calculate the autocorrelation time
tau = np.mean(sampler.get_autocorr_time())

# Remove the burn-in. We'll use ~3x the autocorrelation time
if not np.isnan(tau):
    burnin = int(3 * tau)

    # We will also thin by roughly half our autocorrelation time
    thinning = int(tau // 2)
else:
    burnin = int(nsteps // 3)
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
