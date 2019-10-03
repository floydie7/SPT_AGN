"""
SPT_AGN_emcee.py
Author: Benjamin Floyd

This script will preform the Bayesian analysis on the SPT-AGN data to produce the posterior probability distributions
for all fitting parameters.
"""

import sys
from itertools import product
from time import time

import astropy.units as u
import emcee
import matplotlib
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from custom_math import trap_weight  # Custom trapezoidal integration
from schwimmbad import MPIPool
from scipy.spatial.distance import cdist

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def rebin(a, rebin_factor, wcs=None):
    """
    Rebin an image to the new shape and adjust the WCS
    :param a: Original image
    :param rebin_factor: rebinning scale factor
    :param wcs: Optional, original WCS object
    :return:
    """
    newshape = tuple(rebin_factor * x for x in a.shape)

    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')  # choose the biggest smaller integer index
    new_image = a[tuple(indices)]

    if wcs is not None:
        new_wcs = wcs.deepcopy()
        new_wcs.pixel_shape = new_image.shape  # Update the NAXIS1/2 values
        new_wcs.wcs.cd /= rebin_factor  # Update the pixel scale

        # Transform the reference pixel coordinate
        old_crpix = wcs.wcs.crpix
        new_crpix = np.floor(old_crpix) / a.shape * new_image.shape + old_crpix - np.floor(old_crpix)
        new_wcs.wcs.crpix = new_crpix

        return new_image, new_wcs

    return new_image


def good_pixel_fraction(r, z, r500, center, cluster_id, rescale_factor=None):
    # Read in the mask file and the mask file's WCS
    image, header = mask_dict[cluster_id]  # This is provided by the global variable mask_dict
    image_wcs = WCS(header)

    if rescale_factor is not None:
        image, image_wcs = rebin(image, rescale_factor, wcs=image_wcs)

    # From the WCS get the pixel scale
    pix_scale = image_wcs.pixel_scale_matrix[1, 1] * u.deg

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

    # Convert our radius to pixels
    r_pix = r * r500 * cosmo.arcsec_per_kpc_proper(z).to(u.deg / u.Mpc) / pix_scale
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
    # eta, zeta, beta, C = params

    # Extract our data from the catalog dictionary
    z = catalog_dict[cluster_id]['redshift']
    m = catalog_dict[cluster_id]['m500']
    r500 = catalog_dict[cluster_id]['r500']

    # theta = theta / u.Mpc**2 * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc/u.arcmin)**2

    # Convert our background surface density from angular units into units of r500^-2
    background = C_true / u.arcmin**2 * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc)**2 * r500**2

    # The cluster's core radius in units of r500
    rc_r500 = 0.1 * u.Mpc / r500

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z)**eta * (m / (1e15 * u.Msun))**zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc_r500)**2)**(-1.5 * beta + 0.5) + background

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

        # Select only the objects within the same radial limit we are using for integration.
        radial_r500_maxr = radial_r500[radial_r500 <= rall[-1]]

        # Compute the model rate at the locations of the AGN.
        ni = model_rate_opted(param, cluster_id, radial_r500_maxr)

        # Compute the full model along the radial direction
        nall = model_rate_opted(param, cluster_id, rall)

        # Use a spatial poisson point-process log-likelihood
        cluster_lnlike = np.sum(np.log(ni * radial_r500_maxr)) - trap_weight(nall * 2*np.pi * rall, rall, weight=gpf_all)
        lnlike_list.append(cluster_lnlike)

    total_lnlike = np.sum(lnlike_list)

    return total_lnlike


# For our prior, we will choose uninformative priors for all our parameters and for the constant field value we will use
# a gaussian distribution set by the values obtained from the SDWFS data set.
def lnprior(param):
    # Extract our parameters
    theta, eta, zeta, beta, C = param
    # eta, zeta, beta, C = param

    # Set our hyperparameters
    h_C = 0.371
    h_C_err = 0.157

    # Narrow the prior on theta to bracket +- 50% of theta_true
    # theta_lower = theta_true - theta_true * 0.5
    # theta_upper = theta_true + theta_true * 0.5

    # Define all priors to be gaussian
    if 0.0 <= theta <= 1.0 and -3. <= eta <= 3. and -3. <= zeta <= 3. and -3. <= beta <= 3. \
            and 0.0 <= C < np.inf:
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
    # total_lnprior = eta_lnprior + zeta_lnprior + beta_lnprior + C_lnprior

    return total_lnprior


# Define the log-posterior probability
def lnpost(param):
    lp = lnprior(param)

    # Check the finiteness of the prior.
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(param)


tusker_prefix = '/work/mei/bfloyd/SPT_AGN/'
# tusker_prefix = ''
theta_input = sys.argv[1]
# Read in the mock catalog
mock_catalog = Table.read(tusker_prefix+'Data/MCMC/Mock_Catalog/Catalogs/Signal-Noise_tests/full_spt/'
                                        'mock_AGN_catalog_t{theta}_e1.20_z-1.00_b0.50_C0.371'
                                        '_maxr5.00_seed890_full_spt.cat'.format(theta=theta_input),
                          format='ascii')

# Read in the mask files for each cluster
mock_catalog_grp = mock_catalog.group_by('SPT_ID')
mask_dict = {cluster_id[0]: fits.getdata(tusker_prefix+mask_file, header=True) for cluster_id, mask_file
             in zip(mock_catalog_grp.groups.keys.as_array(),
                    mock_catalog_grp['MASK_NAME'][mock_catalog_grp.groups.indices[:-1]])}

# Set parameter values
theta_true = float(theta_input)    # Amplitude.
eta_true = 1.2       # Redshift slope
beta_true = 0.5      # Radial slope
zeta_true = -1.0     # Mass slope
C_true = 0.371       # Background AGN surface density

# max_radius = 5.0  # Maximum integration radius in r500 units

rescale_fact = 6  # Factor by which we will rescale the mask images to gain higher resolution

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

    # Determine the maximum radius we can integrate to while remaining completely on image.
    mask_image, mask_header = mask_dict[cluster_id]
    mask_wcs = WCS(mask_header)
    pix_scale = mask_wcs.pixel_scale_matrix[1, 1] * u.deg
    cluster_sz_cent_pix = mask_wcs.wcs_world2pix(cluster_sz_cent['SZ_RA'], cluster_sz_cent['SZ_DEC'], 0)
    max_radius_pix = np.min([cluster_sz_cent_pix[0], cluster_sz_cent_pix[1],
                             np.abs(cluster_sz_cent_pix[0] - mask_wcs.pixel_shape[0]),
                             np.abs(cluster_sz_cent_pix[1] - mask_wcs.pixel_shape[1])])
    max_radius_r500 = max_radius_pix * pix_scale * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc/u.deg) / cluster_r500

    # Find the appropriate mesh step size. Since we work in r500 units we convert the pixel scale from angle/pix to
    # r500/pix.
    pix_scale_r500 = pix_scale * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc / u.deg) / cluster_r500

    # Generate a radial integration mesh.
    rall = np.arange(0., max_radius_r500, pix_scale_r500/rescale_fact)
    # rall = np.logspace(-2, np.log10(max_radius), num=200)

    cluster_gpf_all = good_pixel_fraction(rall, cluster_z, cluster_r500, cluster_sz_cent, cluster_id, rescale_factor=rescale_fact)
    # cluster_gpf_all = None

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
nsteps = int(1e6)

# We will initialize our walkers in a tight ball near the initial parameter values.
pos0 = emcee.utils.sample_ball(p0=[theta_true, eta_true, zeta_true, beta_true, C_true],
                               std=[1e-2, 1e-2, 1e-2, 1e-2, 0.157], size=nwalkers)
# pos0 = emcee.utils.sample_ball(p0=[eta_true, zeta_true, beta_true, C_true],
#                                std=[1e-2, 1e-2, 1e-2, 0.157], size=nwalkers)

# Set up the autocorrelation and convergence variables
index = 0
autocorr = np.empty(nsteps)
old_tau = np.inf  # For convergence

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Filename for hd5 backend
    chain_file = 'emcee_run_w{nwalkers}_s{nsteps}_mock_t{theta}_e{eta}_z{zeta}_b{beta}_C{C}_full_spt_snr_tests.h5'\
        .format(nwalkers=nwalkers, nsteps=nsteps,
                theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true)
    backend = emcee.backends.HDFBackend(chain_file, name='snr_test_{theta:.3f}_bkg_free_rc_fixed'.format(theta=theta_true))
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
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$C$']
truths = [theta_true, eta_true, zeta_true, beta_true, C_true]
# labels = [r'$\eta$', r'$\zeta$', r'$\beta$', r'$C$']
# truths = [eta_true, zeta_true, beta_true, C_true]

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
