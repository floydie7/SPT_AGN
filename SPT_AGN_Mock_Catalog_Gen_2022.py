"""
SPT_AGN_Mock_Catalog_Gen_2022.py
Author: Benjamin Floyd

This is a script version of the Notebook `Mock_generation_2022_Floyd*.ipynb` That is a rewrite of the original mock
catalog generation script.
"""

import glob
# %%
import json
import re

import astropy.units as u
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack, join, unique
from astropy.wcs import WCS
from schwimmbad import MultiPool
from scipy import stats
from scipy.interpolate import interp1d

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Set up rng
seed = 123
rng = np.random.default_rng(seed)
print(f'Using RNG seed: {seed}')


# %% md
### Mock generation functions
# %%
def poisson_point_process(rate, dx, dy=None, lower_dx=0, lower_dy=0):
    """
    Uses a spatial Poisson point process to generate AGN candidate coordinates.

    Parameters
    ----------
    rate : float
        The model rate used in the Poisson distribution to determine the number of points being placed.
    dx, dy : int, Optional
        Upper bound on x- and y-axes respectively. If only `dx` is provided then `dy` = `dx`.
    lower_dx, lower_dy : int, Optional
        Lower bound on x- and y-axes respectively. If not provided, a default of 0 will be used

    Returns
    -------
    coord : np.ndarray
        Numpy array of (x, y) coordinates of AGN candidates
    """

    if dy is None:
        dy = dx

    # Draw from Poisson distribution to determine how many points we will place.
    p = stats.poisson(rate * np.abs(dx - lower_dx) * np.abs(dy - lower_dy)).rvs(random_state=rng)

    # Drop `p` points with uniform x and y coordinates
    x = rng.uniform(lower_dx, dx, size=p)
    y = rng.uniform(lower_dy, dy, size=p)

    # Combine the x and y coordinates.
    coord = np.vstack((x, y))

    return coord


# %%
def generate_mock_cluster(cluster_catalog: Table, color_threshold: float, c_true: float) -> Table:
    cluster_z = cluster_catalog['REDSHIFT']
    cluster_m500 = cluster_catalog['M500']
    cluster_r500 = cluster_catalog['R500'] * u.Mpc
    SZ_center = cluster_catalog['SZ_RA', 'SZ_DEC']
    mask_name = cluster_catalog['MASK_NAME']

    # Background Catalog
    # Read in the mask's WCS for the pixel scale and making SkyCoords
    w = WCS(mask_name)
    mask_pixel_scale = w.proj_plane_pixel_scales()[0]

    # Also get the mask's image size (- 1 to account for the shift between index and length)
    mask_size_x = w.pixel_shape[0] - 1
    mask_size_y = w.pixel_shape[1] - 1
    mask_radius_pix = (
                max_radius * cluster_r500 * cosmo.arcsec_per_kpc_proper(cluster_z).to(mask_pixel_scale.unit / u.Mpc)
                / mask_pixel_scale).value

    # Find the SZ Center for the cluster we are mimicking
    SZ_center_skycoord = SkyCoord(SZ_center['SZ_RA'], SZ_center['SZ_DEC'], unit='deg')

    # Set the bounding box for the object placement
    SZ_center_pix = SZ_center_skycoord.to_pixel(wcs=w, origin=0, mode='wcs')
    upper_x = SZ_center_pix[0] + mask_radius_pix
    upper_y = SZ_center_pix[1] + mask_radius_pix
    lower_x = SZ_center_pix[0] - mask_radius_pix
    lower_y = SZ_center_pix[1] - mask_radius_pix

    # As we aren't using real masks yet, we will crop our data to fit within the image bounds using the image center as reference
    image_center = SkyCoord.from_pixel(np.abs(upper_x - lower_x) / 2, np.abs(upper_y - lower_y) / 2, wcs=w, origin=0,
                                       mode='wcs')

    # Scale the true background rate from arcmin^-2 units to pixel units
    background_rate = c_true / u.arcmin ** 2 * mask_pixel_scale.to(u.arcmin) ** 2
    bkg_coords_pix = poisson_point_process(background_rate, dx=upper_x, dy=upper_y, lower_dx=lower_x, lower_dy=lower_y)
    bkg_cat_df = sdwfs_agn.to_pandas().sample(n=bkg_coords_pix.shape[-1], replace=True, random_state=rng)
    bkg_cat = Table.from_pandas(bkg_cat_df)
    bkg_cat['x'] = bkg_coords_pix[0]
    bkg_cat['y'] = bkg_coords_pix[1]

    bkg_cat = bkg_cat['x', 'y', 'REDSHIFT', 'COMPLETENESS_CORRECTION', f'SELECTION_MEMBERSHIP_{color_threshold:.2f}']
    bkg_cat.rename_columns(['REDSHIFT', f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'],
                           ['galaxy_redshift', 'SELECTION_MEMBERSHIP'])

    # Add flag to background objects
    bkg_cat['CLUSTER_AGN'] = np.full_like(bkg_cat['x'], False)

    # Cluster Catalog
    # Set an array of radii to generate model rates upon
    r_grid = np.linspace(0., max_radius, num=100)

    # Find the maximum rate of our model to use to for homogeneous Poisson process (Using c = 0.0 for a cluster-only model)
    max_rate = np.max(
        model_rate(params=(theta_true, eta_true, zeta_true, beta_true, rc_true, 0.0), z=cluster_z, m=cluster_m500,
                   r500=cluster_r500, radial_dist=r_grid, cluster_id=-1))

    # Convert the max rate from [R_500^-2] units to [pix^-2] units
    max_rate_inv_pix2 = (
                (max_rate / cluster_r500 ** 2) * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc / u.arcmin) ** 2
                * mask_pixel_scale.to(u.arcmin) ** 2)

    # For the cluster, we need to select only objects within a redshift range of the cluster redshift.
    sdwfs_agn_at_z = sdwfs_agn[np.abs(sdwfs_agn['REDSHIFT'] - cluster_catalog['REDSHIFT']) <= 0.1]

    # Generate the homogenous Poisson process (Again, this will need to be done on pixel units in the future.)
    cl_coords = poisson_point_process(max_rate_inv_pix2, dx=upper_x, dy=upper_y, lower_dx=lower_x, lower_dy=lower_y)
    cl_cat_df = sdwfs_agn_at_z.to_pandas().sample(n=cl_coords.shape[-1], replace=True, random_state=rng)
    cl_cat = Table.from_pandas(cl_cat_df)
    cl_cat['x'] = cl_coords[0]
    cl_cat['y'] = cl_coords[1]

    cl_cat = cl_cat['x', 'y', 'REDSHIFT', 'COMPLETENESS_CORRECTION', f'SELECTION_MEMBERSHIP_{color_threshold:.2f}']
    cl_cat.rename_columns(['REDSHIFT', f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'],
                          ['galaxy_redshift', 'SELECTION_MEMBERSHIP'])

    # Find the separations in r500 units
    cluster_agn_skycoord = SkyCoord.from_pixel(cl_cat['x'], cl_cat['y'], wcs=w, origin=0, mode='wcs')
    radii_arcmin = SZ_center_skycoord.separation(cluster_agn_skycoord).to(u.arcmin)
    radii_r500 = radii_arcmin * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc / u.arcmin) / cluster_r500

    # Compute model rates at each candidate position
    rate_at_radius = model_rate(params=(theta_true, eta_true, zeta_true, beta_true, rc_true, 0.), z=cluster_z,
                                m=cluster_m500, r500=cluster_r500, radial_dist=radii_r500.value, cluster_id=-1)

    # Perform rejection sampling
    prob_reject = rate_at_radius / max_rate
    alpha = rng.uniform(0., 1., size=len(rate_at_radius))
    cl_cat = cl_cat[prob_reject >= alpha]

    # Add flag to cluster objects
    cl_cat['CLUSTER_AGN'] = np.full_like(cl_cat['x'], True)

    # Merge the catalogs
    los_cat = vstack([cl_cat, bkg_cat])

    # Add cluster information
    los_cat['SPT_ID'] = cluster_catalog['SPT_ID']
    los_cat['REDSHIFT'] = cluster_z
    los_cat['M500'] = cluster_m500
    los_cat['R500'] = cluster_r500

    # Convert the coordinates to RA/Dec
    los_coords_skycoord = SkyCoord.from_pixel(los_cat['x'], los_cat['y'], wcs=w, origin=0, mode='wcs')
    los_cat['RA'] = los_coords_skycoord.ra
    los_cat['DEC'] = los_coords_skycoord.dec

    # Record the separations of all the objects in angular units and r500 units
    los_cat['RADIAL_SEP_ARCMIN'] = SZ_center_skycoord.separation(los_coords_skycoord).to(u.arcmin)
    los_cat['RADIAL_SEP_R500'] = los_cat['RADIAL_SEP_ARCMIN'] * cosmo.kpc_proper_per_arcmin(cluster_z).to(
        u.Mpc / u.arcmin) / cluster_r500

    # Crop the data to be within 2.5 arcmin of the image center
    image_center_sep = image_center.separation(los_coords_skycoord).to(u.arcmin)
    los_cat = los_cat[image_center_sep <= 2.5 * u.arcmin]

    return los_cat


# %% md
### Good Pixel Fraction Functions
# %%

# %% md
### Generating Model
# %%
def model_rate(params, z, m, r500, radial_dist, cluster_id):
    """
    Our generating model.

    Parameters
    ----------
    params : tuple of floats
        Tuple of parameters.
    radial_dist : array-like
        A vector of radii of objects relative to the cluster center
    cluster_id : int or str
        Used to select correct background prior

    Returns
    -------
    model : np.ndarray
        A surface density profile of objects as a function of radius.
    """

    # Unpack the parameters
    if cluster_id == -1:
        theta, eta, zeta, beta, rc, c0 = params
    else:
        # theta, beta, c0 = (0., 0., *params)
        theta, eta, zeta, beta, rc, c0 = params
    # rc = rc_true

    # In mock generation, we need to be able to skip adding the background surface density redshift relation.
    if cluster_id == -1:
        cz = 0.
    else:
        cz = ((c0 + delta_c(z) * num_clusters) / u.arcmin ** 2)\
             * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc)**2 * r500**2

    # Our amplitude will eventually be more complicated
    a = theta * (1 + z) ** eta * (m / 1e15) ** zeta

    # Our model rate is an amplitude of cluster-specific trends with a radial dependence with a constant background rate.
    model = a * (1 + (radial_dist / rc) ** 2) ** (-1.5 * beta + 0.5) + cz

    return model


# %% md
### Bayesian model functions
# %%
def lnlike(params: tuple[float, ...]):
    # Compute the likelihood value for each cluster
    cluster_like = []
    for cluster in catalog.group_by('SPT_ID').groups:
        cluster_id = cluster['SPT_ID'][0]
        cluster_z = cluster['REDSHIFT'][0]
        cluster_m500 = cluster['M500'][0]
        cluster_r500 = cluster['R500'][0] * u.Mpc
        ri = cluster['RADIAL_SEP_R500'].value

        # Get the selection membership of each object
        # mu_agn = cluster['SELECTION_MEMBERSHIP']
        mu_agn = 1.

        # Compute the model rate at locations of the AGN.
        ni = model_rate(params, cluster_z, cluster_m500, cluster_r500, ri, cluster_id)

        # Compute the ideal model rate at continuous locations
        max_r = 2.5 * u.arcmin * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc / u.arcmin) / cluster_r500
        rall = np.linspace(0., max_r.value, num=10_000)
        nall = model_rate(params, cluster_z, cluster_m500, cluster_r500, rall, cluster_id)

        # We use a Poisson likelihood function
        ln_like_func = np.sum(np.log(ni * ri * mu_agn)) - np.trapz(nall * 2 * np.pi * rall, rall)
        cluster_like.append(ln_like_func)

    # Compute the total likelihood value
    total_ln_like = np.sum(cluster_like)
    return total_ln_like


# %%
def lnprior(params: tuple[float, ...]):
    # Extract the parameters
    # theta, beta, c0 = (0., 0., *params)
    theta, eta, zeta, beta, rc, c0 = params
    # rc = rc_true

    cluster_prior = []
    for cluster in catalog.group_by('SPT_ID').groups:
        # Get the cluster redshift to set the background hyperparameters
        z = cluster['REDSHIFT'][0]
        h_c = agn_prior_surf_den(z) * num_clusters
        h_c_err = agn_prior_surf_den_err(z) * num_clusters

        # Shift background parameter to redshift-dependent value.
        cz = c0 + delta_c(z) * num_clusters

        # Define parameter ranges
        if (0. <= theta <= np.inf and
                -6. <= eta <= 6. and
                -3. <= zeta <= 3. and
                -3. <= beta <= 3. and
                0.05 <= rc <= 0.5 and
                0. <= cz <= np.inf):
            theta_lnprior = 0.
            eta_lnprior = 0.
            zeta_lnprior = 0.
            beta_lnprior = 0.
            rc_lnprior = 0.
            c_lnprior = -0.5 * np.sum((cz - h_c) ** 2 / h_c_err ** 2)
            # c_lnprior = 0.
        else:
            theta_lnprior = -np.inf
            eta_lnprior = -np.inf
            zeta_lnprior = -np.inf
            beta_lnprior = -np.inf
            rc_lnprior = -np.inf
            c_lnprior = -np.inf
        ln_prior_prob = theta_lnprior + eta_lnprior + zeta_lnprior + beta_lnprior + rc_lnprior + c_lnprior
        cluster_prior.append(ln_prior_prob)

    total_lnprior = np.sum(cluster_prior)
    return total_lnprior


# %%
def lnprob(params: tuple[float, ...]):
    # Evaluate log-prior and test if we are within bounds
    lp = lnprior(params)

    if not np.isfinite(lp):
        return -np.inf

    return lnlike(params) + lp


# %% md
## Generate mock catalog
# %%
# Select out a number of clusters to use as examples
n_cl = 4

# Read in the SDWFS IRAGN catalog for use later
sdwfs_agn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')

# Read in the SPT cluster catalog. We will use real data to source our mock cluster properties.
Bocquet = Table.read(f'Data_Repository/Catalogs/SPT/SPT_catalogs/2500d_cluster_sample_Bocquet18.fits')

# For the 20 common clusters between SPT-SZ 2500d and SPTpol 100d surveys we want to update the cluster information from
# the more recent survey. Thus, we will merge the SPT-SZ and SPTpol catalogs together.
Huang = Table.read(f'Data_Repository/Catalogs/SPT/SPT_catalogs/sptpol100d_catalog_huang19.fits')

# First we need to rename several columns in the SPTpol 100d catalog to match the format of the SPT-SZ catalog
Huang.rename_columns(['Dec', 'xi', 'theta_core', 'redshift', 'redshift_unc'],
                     ['DEC', 'XI', 'THETA_CORE', 'REDSHIFT', 'REDSHIFT_UNC'])

# Now, merge the two catalogs
SPTcl = join(Bocquet, Huang, join_type='outer')
SPTcl.sort(keys=['SPT_ID', 'field'])  # Sub-sorting by 'field' puts Huang entries first
SPTcl = unique(SPTcl, keys='SPT_ID', keep='first')  # Keeping Huang entries over Bocquet
SPTcl.sort(keys='SPT_ID')  # Resort by ID.

# Convert masses to [Msun] rather than [Msun/1e14]
SPTcl['M500'] *= 1e14
SPTcl['M500_uerr'] *= 1e14
SPTcl['M500_lerr'] *= 1e14

# Remove any unconfirmed clusters
SPTcl = SPTcl[SPTcl['M500'] > 0.0]

# For our masks, we will co-op the masks for the real clusters.
masks_files = [*glob.glob(f'Data_Repository/Project_Data/SPT-IRAGN/Masks/SPT-SZ_2500d/*.fits'),
               *glob.glob(f'Data_Repository/Project_Data/SPT-IRAGN/Masks/SPTpol_100d/*.fits')]

# Make sure all the masks have matches in the catalog
masks_files = [f for f in masks_files if re.search(r'SPT-CLJ\d+-\d+', f).group(0) in SPTcl['SPT_ID']]

# Select a number of masks at random, sorted to match the order in `full_spt_catalog`.
masks_bank = sorted([masks_files[i] for i in rng.choice(n_cl, size=n_cl)],
                    key=lambda x: re.search(r'SPT-CLJ\d+-\d+', x).group(0))

# Find the corresponding cluster IDs in the SPT catalog that match the masks we chose
spt_catalog_ids = [re.search(r'SPT-CLJ\d+-\d+', mask_name).group(0) for mask_name in masks_bank]
spt_catalog_mask = [np.where(SPTcl['SPT_ID'] == spt_id)[0][0] for spt_id in spt_catalog_ids]
selected_clusters = SPTcl['SPT_ID', 'RA', 'DEC', 'M500', 'REDSHIFT', 'REDSHIFT_UNC', 'THETA_CORE', 'XI', 'field'][
    spt_catalog_mask]

# We'll need the r500 radius for each cluster too.
selected_clusters['R500'] = (3 * selected_clusters['M500'] * u.Msun /
                             (4 * np.pi * 500 *
                              cosmo.critical_density(selected_clusters['REDSHIFT']).to(u.Msun / u.Mpc ** 3))) ** (1 / 3)

# Create cluster names
name_bank = [f'SPT_Mock_{i:03d}' for i in range(n_cl)]

# Combine our data into a catalog
SPT_data = selected_clusters.copy()
SPT_data.rename_columns(['SPT_ID', 'RA', 'DEC'], ['orig_SPT_ID', 'SZ_RA', 'SZ_DEC'])
SPT_data['SPT_ID'] = name_bank
SPT_data['MASK_NAME'] = masks_bank
# %%
# Read in the purity and surface density files
with (open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color.json', 'r') as f,
      open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_background_prior_distributions.json',
           'r') as g):
    sdwfs_purity_data = json.load(f)
    sdwfs_prior_data = json.load(g)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
threshold_bins = sdwfs_prior_data['color_thresholds'][:-1]
# %%
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
# %%
cluster_redshifts = SPT_data['REDSHIFT']
redshift_uncert = SPT_data['REDSHIFT_UNC']

# For a cluster at z = 0.6, the color threshold will be [3.6] - [4.5] = 0.61
color_thresholds = [agn_purity_color(z) for z in cluster_redshifts]

# Set the maximum radius (in r500 units) that we will generate out to.
max_radius = 5.  # [R_500]

# We'll boost the number of objects in our sample by duplicating this cluster by a factor.
num_clusters = 50

# We will set our input (true) parameters to be an arbitrary value for cluster and using an approximation of the expected background surface density using our color threshold.
theta_true = 5.0
eta_true = 4.0
zeta_true = -1.0
beta_true = 1.0
rc_true = 0.1
c0_true = agn_prior_surf_den(0.)
c_truths = np.array([agn_prior_surf_den(z) for z in cluster_redshifts])
c_err_truths = np.array([agn_prior_surf_den_err(z) for z in cluster_redshifts])

# We will amplify the true parameters by the number of clusters in the sample.
theta_true *= num_clusters
c0_true *= num_clusters
c_truths *= num_clusters
c_err_truths *= num_clusters
print(
    f'Input parameters: {theta_true = }, {eta_true = }, {zeta_true = } {beta_true = }, {rc_true = }, {c0_true = :.3f}')
# %% md
### Run cluster realization pipeline
# %%
cluster_cats = []
for cluster_catalog, cluster_color_threshold, bkg_rate_true, bkg_rate_err_true in zip(SPT_data, color_thresholds,
                                                                                      c_truths, c_err_truths):
    cat = generate_mock_cluster(cluster_catalog, cluster_color_threshold, bkg_rate_true)
    cluster_cats.append(cat)

    # Show plot of combined line-of-sight positions
    # cluster_objects = cat[cat['CLUSTER_AGN'].astype(bool)]
    # background_objects = cat[~cat['CLUSTER_AGN'].astype(bool)]
    # _, ax = plt.subplots()
    # ax.scatter(background_objects['x'], background_objects['y'], edgecolors='blue', facecolors='none', alpha=0.4, label='Background')
    # ax.scatter(cluster_objects['x'], cluster_objects['y'], edgecolors='red', facecolors='red', alpha=0.6, label='Cluster')
    # ax.legend()
    # ax.set(title=f'{cluster_catalog["SPT_ID"]} at z = {cluster_catalog["REDSHIFT"]:.2f}', xlabel='x [arcmin]', ylabel='y [arcmin]', xlim=[0, image_width], ylim=[0, image_width], aspect=1)
    # plt.show()

# Combine all catalogs
master_catalog = vstack(cluster_cats)
# %%
cluster_only = master_catalog[master_catalog['CLUSTER_AGN'].astype(bool)]
background_only = master_catalog[~master_catalog['CLUSTER_AGN'].astype(bool)]
len(cluster_only), len(background_only)
# %%
catalog = master_catalog
# %% md
## Apply Bayesian model to refit data
# %%
# Set up walkers
ndim = 6
nwalkers = 50
nsteps = 5000

# Initialize walker positions
pos0 = np.array([
    rng.normal(theta_true, 1e-4, size=nwalkers),
    rng.normal(eta_true, 1e-4, size=nwalkers),
    rng.normal(zeta_true, 1e-4, size=nwalkers),
    rng.normal(beta_true, 1e-4, size=nwalkers),
    rng.normal(rc_true, 1e-4, size=nwalkers),
    rng.normal(c0_true, 1e-4, size=nwalkers)]).T

filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Port_Rebuild_Tests/pure_poisson/emcee_mock_pure_poisson.h5'
backend = emcee.backends.HDFBackend(filename=filename, name=f'full_los_{ndim}param_{n_cl}clusters_r500')
with MultiPool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=lnprob, pool=pool, backend=backend)
    sampler.run_mcmc(pos0, nsteps=nsteps, progress=True)

try:
    print(f'Mean autocorrelation time: {(mean_tau := np.mean(sampler.get_autocorr_time())):.2f} steps\n',
          f'Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}')
except emcee.autocorr.AutocorrError:
    print(f'Mean autocorrelation time: {(mean_tau := np.mean(sampler.get_autocorr_time(quiet=True))):.2f} steps\n',
          f'Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}')
# %%
# Plot chains
samples = sampler.get_chain()
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C_0$']
truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, c0_true]
# truth, label = truths[-1], labels[-1]
fig, axes = plt.subplots(nrows=ndim, figsize=(10, 7), sharex='col')
if ndim == 1:
    axes.plot(samples[:, :, 0], 'k', alpha=0.3)
    axes.axhline(y=truths[-1], c='b')
    axes.set(ylabel=labels[-1], xlim=[0, len(samples)])
    axes.set(xlabel='Steps')
else:
    for i, (ax, label, truth) in enumerate(zip(axes.flatten(), labels, truths)):
        ax.plot(samples[:, :, i], 'k', alpha=0.3)
        ax.axhline(y=truth, c='b')
        ax.set(ylabel=label, xlim=[0, len(samples)])
    axes[-1].set(xlabel='Steps')
fig.savefig(
    f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Port_Rebuild_Tests/pure_poisson/param_chains_full_los_{ndim}param_{n_cl}clusters_r500.pdf')
# %%
# Plot posterior
flat_samples = sampler.get_chain(discard=int(3 * mean_tau), flat=True)
if ndim == 1:
    fig = corner.corner(flat_samples, labels=[labels[-1]], truths=[truths[-1]], show_titles=True,
                        quantiles=[0.16, 0.5, 0.84])
else:
    fig = corner.corner(flat_samples, labels=labels, truths=truths, show_titles=True, quantiles=[0.16, 0.5, 0.84])
fig.savefig(
    f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Port_Rebuild_Tests/pure_poisson/corner_full_los_{ndim}param_{n_cl}clusters_r500.pdf')
