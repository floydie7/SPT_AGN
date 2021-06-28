"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""
import glob
import re
from time import time

import astropy.units as u
import numpy as np
from astro_compendium.utils.k_correction import k_corr_abs_mag
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, join, unique, vstack
from astropy.wcs import WCS
from scipy import stats
from scipy.interpolate import lagrange
from synphot import SpectralElement, SourceSpectrum, units

# Set our cosmology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

# Generate a random seed
# cluster_seed, object_seed = np.random.default_rng().integers(1024, size=2)
cluster_seed = 890
object_seed = 930
print(f'Cluster Seed: {cluster_seed}\t Object Seed: {object_seed}')

# Set our random number generators
cluster_rng = np.random.default_rng(cluster_seed)  # Previously 123
object_rng = np.random.default_rng(object_seed)


def poisson_point_process(model, dx, dy=None, lower_dx=0, lower_dy=0):
    """
    Uses a spatial Poisson point process to generate AGN candidate coordinates.

    Parameters
    ----------
    model : float
        The model rate used in the Poisson distribution to determine the number of points being placed.
    dx, dy : int, Optional
        Upper bound on x- and y-axes respectively. If only `dx` is provided then `dy` = `dx`.
    lower_dx, lower_dy : int, Optional
        Lower bound on x- and y-axes respectively. If not provided, a default of 0 will be used

    Returns
    -------
    coord : ndarray
        Numpy array of (x, y) coordinates of AGN candidates
    """

    if dy is None:
        dy = dx

    # Draw from Poisson distribution to determine how many points we will place.
    p = stats.poisson(model * np.abs(dx - lower_dx) * np.abs(dy - lower_dy)).rvs()

    # Drop `p` points with uniform x and y coordinates
    x = object_rng.uniform(lower_dx, dx, size=p)
    y = object_rng.uniform(lower_dy, dy, size=p)

    # Combine the x and y coordinates.
    coord = np.vstack((x, y))

    return coord


def luminosity_function(abs_mag, redshift):
    """
    Assef+11 QLF using luminosity and density evolution.

    Parameters
    ----------
    abs_mag : array-like
        Rest-frame J-band absolute magnitude.
    redshift : array-like
        Cluster redshift

    Returns
    -------
    Phi : ndarray
        Luminosity density

    """

    # Set up the luminosity and density evolution using the fits from Assef+11 Table 2
    z_i = [0.25, 0.5, 1., 2., 4.]
    m_star_z_i = [-23.51, -24.64, -26.10, -27.08]
    phi_star_z_i = [-3.41, -3.73, -4.17, -4.65, -5.77]
    m_star = lagrange(z_i[1:], m_star_z_i)
    log_phi_star = lagrange(z_i, phi_star_z_i)

    # L/L_*(z) = 10**(0.4 * (M_*(z) - M))
    L_L_star = 10 ** (0.4 * (m_star(redshift) - abs_mag))

    # Phi*(z) = 10**(log(Phi*(z))
    phi_star = 10 ** log_phi_star(redshift) * (cosmo.h / u.Mpc) ** 3

    # QLF slopes
    alpha1 = -3.35  # alpha in Table 2
    alpha2 = -0.37  # beta in Table 2

    Phi = 0.4 * np.log(10) * L_L_star * phi_star * (L_L_star ** -alpha1 + L_L_star ** -alpha2) ** -1

    return Phi


def model_rate(params, z, m, r500, r_r500, j_mag):
    """
    Our generating model (for the cluster term only).

    Parameters
    ----------
    params : tuple
        Tuple of (theta, eta, zeta, beta, rc, C)
    z : float
        The redshift of the cluster
    m : float
        The M500 mass of the cluster
    r500 : float
        The r500 radius of the cluster
    r_r500 : array-like
        A vector of radii of objects within the cluster normalized by the cluster's r500
    j_mag : array-like
        A vector of J-band absolute magnitudes to be used in the luminosity function

    Returns
    -------
    model : ndarray
        A surface density profile of objects as a function of radius and luminosity.
    """

    # Unpack our parameters
    theta, eta, zeta, beta, rc = params

    # Luminosity function number
    LF = cosmo.angular_diameter_distance(z) ** 2 * r500 * luminosity_function(j_mag, z)

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta * LF

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc) ** 2) ** (-1.5 * beta + 0.5)

    return model.value


start_time = time()
# <editor-fold desc="Parameter Set up">

# Number of clusters to generate
n_cl = 249 + 57

# Set parameter values

theta_true = 0.8  # Amplitude
eta_true = 4.0  # Redshift slope
zeta_true = -1.0  # Mass slope
beta_true = 1.0  # Radial slope
rc_true = 0.1  # Core radius (in r500)
C_true = 0.376  # Background AGN surface density (in arcmin^-2)

# Set the maximum radius we will generate objects to as a factor of r500
max_radius = 5.0

# Set cluster center positional uncertainty
median_cluster_pos_uncert = 0.214 * u.arcmin

# SPT's 150 GHz beam size
SZ_theta_beam = 1.2 * u.arcmin
# </editor-fold>

# <editor-fold desc="Data Generation">
# Read in the SPT cluster catalog. We will use real data to source our mock cluster properties.
Bocquet = Table.read('Data_Repository/Catalogs/SPT/SPT_catalogs/2500d_cluster_sample_Bocquet18.fits')

# For the 20 common clusters between SPT-SZ 2500d and SPTpol 100d surveys we want to update the cluster information from
# the more recent survey. Thus, we will merge the SPT-SZ and SPTpol catalogs together.
Huang = Table.read('Data_Repository/Catalogs/SPT/SPT_catalogs/sptpol100d_catalog_huang19.fits')

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
masks_files = [*glob.glob('Data_Repository/Project_Data/SPT-IRAGN/Masks/SPT-SZ_2500d/*.fits'),
               *glob.glob('Data_Repository/Project_Data/SPT-IRAGN/Masks/SPTpol_100d/*.fits')]

# Make sure all the masks have matches in the catalog
masks_files = [f for f in masks_files if re.search(r'SPT-CLJ\d+-\d+', f).group(0) in SPTcl['SPT_ID']]

# Select a number of masks at random, sorted to match the order in `full_spt_catalog`.
masks_bank = sorted([masks_files[i] for i in cluster_rng.choice(n_cl, size=n_cl)],
                    key=lambda x: re.search(r'SPT-CLJ\d+-\d+', x).group(0))

# Find the corresponding cluster IDs in the SPT catalog that match the masks we chose
spt_catalog_ids = [re.search(r'SPT-CLJ\d+-\d+', mask_name).group(0) for mask_name in masks_bank]
spt_catalog_mask = [np.where(SPTcl['SPT_ID'] == spt_id)[0][0] for spt_id in spt_catalog_ids]
selected_clusters = SPTcl['SPT_ID', 'RA', 'DEC', 'M500', 'REDSHIFT', 'THETA_CORE', 'XI'][spt_catalog_mask]

# We'll need the r500 radius for each cluster too.
selected_clusters['R500'] = (3 * selected_clusters['M500'] * u.Msun /
                             (4 * np.pi * 500 *
                              cosmo.critical_density(selected_clusters['REDSHIFT']).to(u.Msun / u.Mpc ** 3))) ** (1 / 3)

# Create cluster names
name_bank = ['SPT_Mock_{:03d}'.format(i) for i in range(n_cl)]

# Combine our data into a catalog
SPT_data = Table([name_bank, selected_clusters['RA'], selected_clusters['DEC'], selected_clusters['M500'],
                  selected_clusters['R500'], selected_clusters['REDSHIFT'], selected_clusters['THETA_CORE'],
                  selected_clusters['XI'], masks_bank, selected_clusters['SPT_ID']],
                 names=['SPT_ID', 'SZ_RA', 'SZ_DEC', 'M500', 'R500', 'REDSHIFT', 'THETA_CORE', 'XI', 'MASK_NAME',
                        'orig_SPT_ID'])

# Check that we have the correct mask and cluster data matched up. If so, we can drop the original SPT_ID column
assert np.all([spt_id in mask_name for spt_id, mask_name in zip(SPT_data['orig_SPT_ID'], SPT_data['MASK_NAME'])])
del SPT_data['orig_SPT_ID']

# Set up grid of radial positions to place AGN on (normalized by r500)
r_dist_r500 = np.linspace(0, max_radius, num=200)

# To provide data for the photometric and AGN sample membership we will need to read in the real data catalog.
# To help provide abstraction from the data, we will only read in the three relevant columns and then randomize the
# order of the rows.
real_data = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')
real_data.keep_columns(['COMPLETENESS_CORRECTION', 'SELECTION_MEMBERSHIP', 'J_ABS_MAG'])
real_data = Table(object_rng.permutation(real_data), names=real_data.colnames)
# </editor-fold>

catalog_start_time = time()
params_true = (theta_true, eta_true, zeta_true, beta_true, rc_true)

cluster_sample = SPT_data.copy()

AGN_cats = []
for cluster in cluster_sample:
    spt_id = cluster['SPT_ID']
    mask_name = cluster['MASK_NAME']
    z_cl = cluster['REDSHIFT']
    m500_cl = cluster['M500'] * u.Msun
    r500_cl = cluster['R500'] * u.Mpc
    SZ_center = cluster['SZ_RA', 'SZ_DEC']
    SZ_theta_core = cluster['THETA_CORE'] * u.arcmin
    SZ_xi = cluster['XI']

    # Read in the mask's WCS for the pixel scale and making SkyCoords
    w = WCS(mask_name)
    try:
        assert w.pixel_scale_matrix[0, 1] == 0.
        mask_pixel_scale = w.pixel_scale_matrix[1, 1] * w.wcs.cunit[1]
    except AssertionError:
        cd = w.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        mask_pixel_scale = cd_diag[1, 1] * w.wcs.cunit[1]

    # Also get the mask's image size (- 1 to account for the shift between index and length)
    mask_size_x = w.pixel_shape[0] - 1
    mask_size_y = w.pixel_shape[1] - 1
    mask_radius_pix = (
            max_radius * r500_cl * cosmo.arcsec_per_kpc_proper(z_cl).to(u.deg / u.Mpc) / mask_pixel_scale).value

    # Find the SZ Center for the cluster we are mimicking
    SZ_center_skycoord = SkyCoord(SZ_center['SZ_RA'], SZ_center['SZ_DEC'], unit='deg')

    # Set the bounding box for the object placement
    SZ_center_pix = SZ_center_skycoord.to_pixel(wcs=w, origin=0, mode='wcs')
    upper_x = SZ_center_pix[0] + mask_radius_pix
    upper_y = SZ_center_pix[1] + mask_radius_pix
    lower_x = SZ_center_pix[0] - mask_radius_pix
    lower_y = SZ_center_pix[1] - mask_radius_pix

    # Generate a grid of J-band absolute magnitudes on which we will compute model rates to determine the maximum rate
    # They will be derived from the empirical 4.5 um magnitude limits k-corrected into J-band absolute magnitudes
    faint_end_45_apmag = 17.46  # Vega mag
    bright_end_45_apmag = 10.45  # Vega mag
    irac_45_filter = SpectralElement.from_file('Data_Repository/filter_curves/Spitzer_IRAC/080924ch2trans_full.txt',
                                               wave_unit=u.um)
    flamingos_j_filter = SpectralElement.from_file('Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                                                   'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt', wave_unit=u.nm)
    qso2_sed = SourceSpectrum.from_file('Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                        wave_unit=u.Angstrom, flux_unit=units.FLAM)
    faint_end_j_absmag = k_corr_abs_mag(faint_end_45_apmag, z=z_cl, f_lambda_sed=qso2_sed,
                                        zero_pt_obs_band=179.7 * u.Jy, zero_pt_em_band='vega',
                                        obs_filter=irac_45_filter, em_filter=flamingos_j_filter, cosmo=cosmo)
    bright_end_j_absmag = k_corr_abs_mag(bright_end_45_apmag, z=z_cl, f_lambda_sed=qso2_sed,
                                         zero_pt_obs_band=179.7 * u.Jy, zero_pt_em_band='vega',
                                         obs_filter=irac_45_filter, em_filter=flamingos_j_filter, cosmo=cosmo)
    j_grid = np.linspace(bright_end_j_absmag, faint_end_j_absmag, num=200)

    # Calculate the model values for the AGN candidates in the cluster
    model_cluster_agn = model_rate(params_true, z_cl, m500_cl, r500_cl, r_dist_r500, j_grid)

    # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
    # the cluster. Then convert to pix^-2 units.
    max_rate = np.max(model_cluster_agn)  # r500^-2 units
    max_rate_inv_pix2 = ((max_rate / r500_cl ** 2) * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) ** 2
                         * mask_pixel_scale.to(u.arcmin) ** 2)

    # Simulate the AGN using the spatial Poisson point process.
    cluster_agn_coords_pix = poisson_point_process(max_rate_inv_pix2, dx=upper_x, dy=upper_y,
                                                   lower_dx=lower_x, lower_dy=lower_y)

    # Now we will assign each cluster object a random completeness value, degree of membership, and J-band abs. mag.
    data_idx = object_rng.integers(len(real_data), size=cluster_agn_coords_pix.shape[1])
    cluster_agn_completeness = real_data['COMPLETENESS_CORRECTION'][data_idx]
    cluster_agn_selection_membership = real_data['SELECTION_MEMBERSHIP'][data_idx]
    cluster_agn_j_abs_mag = real_data['J_ABS_MAG'][data_idx]

    # Find the radius of each point placed scaled by the cluster's r500 radius
    cluster_agn_skycoord = SkyCoord.from_pixel(cluster_agn_coords_pix[0], cluster_agn_coords_pix[1],
                                               wcs=w, origin=0, mode='wcs')
    radii_arcmin = SZ_center_skycoord.separation(cluster_agn_skycoord).to(u.arcmin)
    radii_r500 = radii_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl

    # Filter the candidates through the model to establish the radial trend in the data.
    rate_at_rad = model_rate(params_true, z_cl, m500_cl, r500_cl, radii_r500, cluster_agn_j_abs_mag)

    # Our rejection rate is the model rate at the radius scaled by the maximum rate
    prob_reject = rate_at_rad / max_rate

    # Draw a random number for each candidate
    alpha = object_rng.uniform(0, 1, len(rate_at_rad))

    # Perform the rejection sampling
    cluster_agn_final = cluster_agn_skycoord[prob_reject >= alpha]
    cluster_agn_final_pix = np.array(cluster_agn_final.to_pixel(w, origin=0, mode='wcs'))

    # Apply the rejection sampling on the photometric information for cluster objects as well
    cluster_agn_completeness = cluster_agn_completeness[prob_reject >= alpha]
    cluster_agn_selection_membership = cluster_agn_selection_membership[prob_reject >= alpha]
    cluster_agn_j_abs_mag = cluster_agn_j_abs_mag[prob_reject >= alpha]

    # Generate background sources using a Poisson point process but skipping the rejection sampling step from above.
    background_rate = C_true / u.arcmin ** 2 * mask_pixel_scale.to(u.arcmin) ** 2
    background_agn_pix = poisson_point_process(background_rate, dx=upper_x, dy=upper_y,
                                               lower_dx=lower_x, lower_dy=lower_y)

    # For each background AGN we will also need a random completeness value, degree of membership, and J-band abs. mag.
    data_idx = object_rng.integers(len(real_data), size=background_agn_pix.shape[1])
    background_agn_completeness = real_data['COMPLETENESS_CORRECTION'][data_idx]
    background_agn_selection_membership = real_data['SELECTION_MEMBERSHIP'][data_idx]
    background_agn_j_abs_mag = real_data['J_ABS_MAG'][data_idx]

    # Concatenate the cluster sources with the background sources
    line_of_sight_agn_pix = np.hstack((cluster_agn_final_pix, background_agn_pix))

    # Set up the table of objects
    AGN_list = Table([line_of_sight_agn_pix[0], line_of_sight_agn_pix[1]], names=['x_pixel', 'y_pixel'])
    AGN_list['SPT_ID'] = spt_id
    AGN_list['SZ_RA'] = SZ_center['SZ_RA']
    AGN_list['SZ_DEC'] = SZ_center['SZ_DEC']
    AGN_list['M500'] = m500_cl
    AGN_list['REDSHIFT'] = z_cl
    AGN_list['R500'] = r500_cl
    AGN_list['COMPLETENESS_CORRECTION'] = np.hstack((cluster_agn_completeness, background_agn_completeness))
    AGN_list['SELECTION_MEMBERSHIP'] = np.hstack((cluster_agn_selection_membership,
                                                  background_agn_selection_membership))
    AGN_list['J_ABS_MAG'] = np.hstack((cluster_agn_j_abs_mag, background_agn_j_abs_mag))

    # Create a flag indicating if the object is a cluster member
    AGN_list['Cluster_AGN'] = np.concatenate((np.full_like(cluster_agn_final_pix[0], True),
                                              np.full_like(background_agn_pix[0], False)))

    # Convert the pixel coordinates to RA/Dec coordinates
    agn_coords_skycoord = SkyCoord.from_pixel(AGN_list['x_pixel'], AGN_list['y_pixel'], wcs=w, origin=0, mode='wcs')
    AGN_list['RA'] = agn_coords_skycoord.ra
    AGN_list['DEC'] = agn_coords_skycoord.dec

    # <editor-fold desc="Miscentering">
    # Shift the cluster center away from the true center within the 1-sigma SZ positional uncertainty
    cluster_pos_uncert = np.sqrt(SZ_theta_beam ** 2 + SZ_theta_core ** 2) / SZ_xi
    AGN_list['CENTER_POS_UNC_ARCMIN_1sigma'] = cluster_pos_uncert
    offset_SZ_center = cluster_rng.multivariate_normal(
        (SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
        np.eye(2) * cluster_pos_uncert.to_value(u.deg) ** 2)
    offset_SZ_center_skycoord = SkyCoord(offset_SZ_center[0], offset_SZ_center[1], unit='deg')
    AGN_list['OFFSET_RA'] = offset_SZ_center_skycoord.ra
    AGN_list['OFFSET_DEC'] = offset_SZ_center_skycoord.dec

    # Decrease the positional uncertainty to half of the true value
    cluster_pos_uncert_half = cluster_pos_uncert / 2
    half_offset_SZ_center = cluster_rng.multivariate_normal(
        (SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
        np.eye(2) * cluster_pos_uncert_half.to_value(u.deg) ** 2)
    half_offset_SZ_center_skycoord = SkyCoord(half_offset_SZ_center[0], half_offset_SZ_center[1], unit='deg')
    AGN_list['HALF_OFFSET_RA'] = half_offset_SZ_center_skycoord.ra
    AGN_list['HALF_OFFSET_DEC'] = half_offset_SZ_center_skycoord.dec

    # Decrease the positional uncertainty to 75% of the true value
    cluster_pos_uncert_075 = cluster_pos_uncert * 0.75
    threequarters_offset_SZ_center = cluster_rng.multivariate_normal(
        (SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
        np.eye(2) * cluster_pos_uncert_half.to_value(u.deg) ** 2)
    threequarters_offset_SZ_center_skycoord = SkyCoord(threequarters_offset_SZ_center[0],
                                                       threequarters_offset_SZ_center[1], unit='deg')
    AGN_list['075_OFFSET_RA'] = threequarters_offset_SZ_center_skycoord.ra
    AGN_list['075_OFFSET_DEC'] = threequarters_offset_SZ_center_skycoord.dec
    # </editor-fold>

    # Calculate the radii of the final AGN scaled by the cluster's r500 radius
    r_final_arcmin = SZ_center_skycoord.separation(agn_coords_skycoord).to(u.arcmin)
    r_final_r500 = r_final_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl
    AGN_list['RADIAL_SEP_ARCMIN'] = r_final_arcmin
    AGN_list['RADIAL_SEP_R500'] = r_final_r500

    # <editor-fold, desc="Miscentered radial distances">
    # Also calculate the radial distances based on the offset center.
    r_final_arcmin_offset = offset_SZ_center_skycoord.separation(agn_coords_skycoord).to(u.arcmin)
    r_final_r500_offset = r_final_arcmin_offset * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl
    AGN_list['RADIAL_SEP_ARCMIN_OFFSET'] = r_final_arcmin_offset
    AGN_list['RADIAL_SEP_R500_OFFSET'] = r_final_r500_offset

    # Also, calculate the radial distances based on the half-offset center
    r_final_arcmin_half_offset = half_offset_SZ_center_skycoord.separation(agn_coords_skycoord).to(u.arcmin)
    r_final_r500_half_offset = r_final_arcmin_half_offset * cosmo.kpc_proper_per_arcmin(z_cl).to(
        u.Mpc / u.arcmin) / r500_cl
    AGN_list['RADIAL_SEP_ARCMIN_HALF_OFFSET'] = r_final_arcmin_half_offset
    AGN_list['RADIAL_SEP_R500_HALF_OFFSET'] = r_final_r500_half_offset

    r_final_arcmin_075_offset = threequarters_offset_SZ_center_skycoord.separation(agn_coords_skycoord).to(u.arcmin)
    r_final_r500_075_offset = r_final_arcmin_075_offset * cosmo.kpc_proper_per_arcmin(z_cl).to(
        u.Mpc / u.arcmin) / r500_cl
    AGN_list['RADIAL_SEP_ARCMIN_075_OFFSET'] = r_final_arcmin_075_offset
    AGN_list['RADIAL_SEP_R500_075_OFFSET'] = r_final_r500_075_offset
    # </editor-fold>

    # Select only objects within the max_radius
    AGN_list = AGN_list[AGN_list['RADIAL_SEP_R500'] <= max_radius]

    # Read in the original (full) mask
    full_mask_image, full_mask_header = fits.getdata(mask_name, header=True)

    # Select the image to mask the data on
    mask_image = full_mask_image
    AGN_list['MASK_NAME'] = mask_name

    # Remove all objects that are outside of the image bounds
    AGN_list = AGN_list[np.all([0 <= AGN_list['x_pixel'],
                                AGN_list['x_pixel'] <= mask_size_x,
                                0 <= AGN_list['y_pixel'],
                                AGN_list['y_pixel'] <= mask_size_y], axis=0)]

    # Pass the cluster catalog through the mask to insure all objects are on image.
    AGN_list = AGN_list[np.where(mask_image[np.floor(AGN_list['y_pixel']).astype(int),
                                            np.floor(AGN_list['x_pixel']).astype(int)] == 1)]

    # Perform a rejection sampling based on the completeness value for each object.
    alpha = object_rng.uniform(0, 1, size=len(AGN_list))
    prob_reject = 1 / AGN_list['COMPLETENESS_CORRECTION']
    AGN_list = AGN_list[prob_reject >= alpha]

    AGN_cats.append(AGN_list)

# Stack the individual cluster catalogs into a single master catalog
outAGN = vstack(AGN_cats)

# Reorder the columns in the cluster for ascetic reasons.
outAGN = outAGN['SPT_ID', 'SZ_RA', 'SZ_DEC', 'OFFSET_RA', 'OFFSET_DEC', 'HALF_OFFSET_RA', 'HALF_OFFSET_DEC',
                '075_OFFSET_RA', '075_OFFSET_DEC', 'x_pixel', 'y_pixel', 'RA', 'DEC',
                'REDSHIFT', 'M500', 'R500', 'RADIAL_SEP_ARCMIN', 'RADIAL_SEP_R500', 'RADIAL_SEP_ARCMIN_OFFSET',
                'RADIAL_SEP_R500_OFFSET', 'RADIAL_SEP_ARCMIN_HALF_OFFSET', 'RADIAL_SEP_R500_HALF_OFFSET',
                'RADIAL_SEP_ARCMIN_075_OFFSET', 'RADIAL_SEP_R500_075_OFFSET', 'MASK_NAME',
                'COMPLETENESS_CORRECTION', 'SELECTION_MEMBERSHIP', 'J_ABS_MAG', 'Cluster_AGN']
outAGN.write(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Final_tests/LF_tests/'
             f'mock_AGN_catalog_t{theta_true:.3f}_e{eta_true:.2f}_z{zeta_true:.2f}_b{beta_true:.2f}_rc{rc_true:.3f}'
             f'_C{C_true:.3f}_maxr{max_radius:.2f}'
             f'_clseed{cluster_seed}_objseed{object_seed}_fuzzy_selection_J_abs_mag_compl_rej_samp.fits', overwrite=True)

# Print out statistics
number_of_clusters = len(outAGN.group_by('SPT_ID').groups.keys)
total_number = len(outAGN)
total_number_comp_corrected = outAGN['COMPLETENESS_CORRECTION'].sum()
total_number_corrected = np.sum(outAGN['COMPLETENESS_CORRECTION'] * outAGN['SELECTION_MEMBERSHIP'])
number_per_cluster = total_number_corrected / number_of_clusters
median_z = np.median(outAGN['REDSHIFT'])
median_m = np.median(outAGN['M500'])
print(f"""Mock Catalog
Parameters:\t{params_true + (C_true,)}
Number of clusters:\t{number_of_clusters}
Objects Selected:\t{total_number}
Objects selected (completeness corrected):\t{total_number_comp_corrected:.2f}
Objects Selected (comp + membership corrected):\t{total_number_corrected:.2f}
Objects per cluster (comp + mem corrected):\t{number_per_cluster:.2f}
Median Redshift:\t{median_z:.2f}
Median Mass:\t{median_m:.2e}""")

print('Run time: {:.2f}s'.format(time() - catalog_start_time))
print('Total run time: {:.2f}s'.format(time() - start_time))
