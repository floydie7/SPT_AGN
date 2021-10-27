"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""
import glob
import json
import pickle
import re
from argparse import ArgumentParser
from time import time

import astropy.units as u
import numpy as np
from k_correction import k_corr_abs_mag, k_corr_ap_mag
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, join, unique, vstack
from astropy.wcs import WCS
from scipy import stats
from scipy.integrate import quad, quad_vec
from scipy.interpolate import lagrange, interp1d
from synphot import SpectralElement, SourceSpectrum, units
from schwimmbad import MPIPool

hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'

# Set our cosmology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

# Generate a random seed
# cluster_seed, object_seed = np.random.default_rng().integers(1024, size=2)
cluster_seed = 890
object_seed = 930

# Set our random number generators
cluster_rng = np.random.default_rng(cluster_seed)  # Previously 123
object_rng = np.random.default_rng(object_seed)


# Provide a random variable for the luminosity function
class LFpdf(stats.rv_continuous):
    def __init__(self, z, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._z = z
        self._normalization = self._norm(self._z)

    def _norm(self, z):
        _a, _b = self._get_support()
        return quad(lambda x: luminosity_function(x, z).value, a=_a, b=_b)[0]

    def _pdf(self, abs_mag, *args):
        return luminosity_function(abs_mag, self._z).value / self._normalization


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
    Phi : astropy.units.Quantity
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


def membership_degree(ch1_ch2_color, ch1_ch2_color_err, ch1_ch2_color_cut, color_number_dist, color_bin_extrema):
    """
    Computes the degree of membership.

    Parameters
    ----------
    ch1_ch2_color : array-like, Table
        Array of [3.6] - [4.5] colors for each object.
    ch1_ch2_color_err : array-like, Table
        Array of [3.6] - [4.5] color errors for each object.
    ch1_ch2_color_cut : float
        [3.6] - [4.5] color cut above which objects will be selected as AGN.
    color_number_dist : interp1d
        A callable interpolation function of the color number distribution from a reference field sample.
    color_bin_extrema : array-like
        A list or tuple of the minimum and maximum of the color bins.

    Returns
    -------
    membership : array-like
        An array of degrees of membership for each object.
    """

    # Unpack the extrema
    color_min, color_max = color_bin_extrema

    # Convolve the error distribution for each object with the overall number count distribution
    def object_integrand(x):
        return stats.norm(loc=ch1_ch2_color, scale=ch1_ch2_color_err).pdf(x) * color_number_dist(x)

    membership_numer = quad_vec(object_integrand, a=ch1_ch2_color_cut, b=color_max)[0]
    membership_denom = quad_vec(object_integrand, a=color_min, b=color_max)[0]
    membership = membership_numer / membership_denom

    return membership


def generate_mock_cluster(cluster):
    """Task function to create the line-of-sight cluster catalogs."""
    original_spt_id = cluster['orig_SPT_ID']
    spt_id = cluster['SPT_ID']
    mask_name = cluster['MASK_NAME']
    z_cl = cluster['REDSHIFT']
    m500_cl = cluster['M500'] * u.Msun
    r500_cl = cluster['R500'] * u.Mpc
    SZ_center = cluster['SZ_RA', 'SZ_DEC']
    SZ_theta_core = cluster['THETA_CORE'] * u.arcmin
    SZ_xi = cluster['XI']
    spt_field = cluster['field']

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

    # Draw J-band luminosities for each object from the luminosity function at the cluster redshift
    lf_cl_rv = LFpdf(z=z_cl, a=bright_end_j_absmag, b=faint_end_j_absmag)
    cluster_agn_j_abs_mag = lf_cl_rv.rvs(size=cluster_agn_coords_pix.shape[1])

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

    # Perform the rejection sampling for both the object coordinates and associated luminosities
    cluster_agn_final = cluster_agn_skycoord[prob_reject >= alpha]
    cluster_agn_final_pix = np.array(cluster_agn_final.to_pixel(w, origin=0, mode='wcs'))
    cluster_agn_j_abs_mag_final = cluster_agn_j_abs_mag[prob_reject >= alpha]

    # To get the cluster AGN we must first filter for objects near the cluster redshift
    cluster_color_z = AGN_color_z.T[np.abs(AGN_color_z[0] - z_cl) < delta_z]

    # Draw colors and object redshifts from Stern Wedge subsample of the SDWFS the color--redshift plane
    cluster_obj_z, cluster_obj_colors = cluster_rng.choice(cluster_color_z, size=len(cluster_agn_final), replace=True).T

    # Draw color errors from the range of color errors in the survey. NB this may change in the future
    min_color_err = sptpol_min_color_err if spt_field == 'SPTPOL_100d' else sptsz_min_color_err
    max_color_err = sptpol_max_color_err if spt_field == 'SPTPOL_100d' else sptsz_max_color_err
    cluster_obj_color_errors = object_rng.uniform(min_color_err, max_color_err, size=len(cluster_agn_final))

    # Generate background sources using a Poisson point process but skipping the rejection sampling step from above.
    background_rate = C_true / u.arcmin ** 2 * mask_pixel_scale.to(u.arcmin) ** 2
    background_agn_pix = poisson_point_process(background_rate, dx=upper_x, dy=upper_y,
                                               lower_dx=lower_x, lower_dy=lower_y)

    # Draw luminosities for all the background objects. As the real data does not know the true redshifts of the objects
    # it assumes all objects are at the cluster's redshift. We will mimic this here by using the LF at `z_cl`.
    background_agn_j_abs_mag = lf_cl_rv.rvs(size=background_agn_pix.shape[1])

    # Draw colors and object redshifts from the full SDWFS color--redshift plane
    background_obj_z, background_obj_colors = cluster_rng.choice(SDWFS_color_z.T, size=background_agn_pix.shape[1],
                                                                 replace=True).T

    # Draw color errors from the range of color errors in the survey as before.
    background_obj_color_errors = object_rng.uniform(min_color_err, max_color_err, size=background_agn_pix.shape[1])

    # Concatenate the cluster sources with the background sources
    line_of_sight_agn_pix = np.hstack((cluster_agn_final_pix, background_agn_pix))
    line_of_sight_agn_j_abs_mag = np.hstack((cluster_agn_j_abs_mag_final, background_agn_j_abs_mag))
    line_of_sight_agn_colors = np.hstack((cluster_obj_colors, background_obj_colors))
    line_of_sight_agn_color_errors = np.hstack((cluster_obj_color_errors, background_obj_color_errors))
    line_of_sight_agn_obj_z = np.hstack((cluster_obj_z, background_obj_z))

    # Create a flag indicating if the object is a cluster member
    line_of_sight_agn_cluster_mem = np.hstack((np.full_like(cluster_agn_final_pix[0], True),
                                               np.full_like(background_agn_pix[0], False)))

    # Set up the table of objects
    AGN_list = Table([line_of_sight_agn_pix[0], line_of_sight_agn_pix[1],
                      line_of_sight_agn_j_abs_mag,
                      line_of_sight_agn_colors, line_of_sight_agn_color_errors,
                      line_of_sight_agn_obj_z,
                      line_of_sight_agn_cluster_mem],
                     names=['x_pixel', 'y_pixel', 'J_ABS_MAG', 'I1_I2', 'I1_I2_ERR', 'OBJ_REDSHIFT', 'CLUSTER_AGN'])
    AGN_list['SPT_ID'] = spt_id
    AGN_list['SZ_RA'] = SZ_center['SZ_RA']
    AGN_list['SZ_DEC'] = SZ_center['SZ_DEC']
    AGN_list['M500'] = m500_cl
    AGN_list['REDSHIFT'] = z_cl
    AGN_list['R500'] = r500_cl

    # Compute 4.5 um apparent magnitudes from the J-band absolute magnitudes
    AGN_list['I2_APMAG'] = k_corr_ap_mag(AGN_list['J_ABS_MAG'], z=z_cl, f_lambda_sed=qso2_sed,
                                         zero_pt_obs_band=179.7 * u.Jy, zero_pt_em_band='vega',
                                         obs_filter=irac_45_filter, em_filter=flamingos_j_filter, cosmo=cosmo)

    # Pull up the cluster's completeness curve and build interpolator
    completeness_data = sptcl_comp_sim[original_spt_id]
    comp_sim_mag_bins = sptcl_comp_sim['magnitude_bins'][:-1]
    completeness_funct = interp1d(comp_sim_mag_bins, completeness_data, kind='linear')

    # Measure completeness values and corrections for all objects
    AGN_list['COMPLETENESS_VALUE'] = completeness_funct(AGN_list['I2_APMAG'])
    AGN_list['COMPLETENESS_CORRECTION'] = 1 / AGN_list['COMPLETENESS_VALUE']

    # Compute the degrees of membership for each object
    AGN_list['SELECTION_MEMBERSHIP'] = membership_degree(AGN_list['I1_I2'], AGN_list['I1_I2_ERR'],
                                                         ch1_ch2_color_cut=0.7,
                                                         color_number_dist=color_probability_distribution,
                                                         color_bin_extrema=(color_bin_min, color_bin_max))

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
        np.eye(2) * cluster_pos_uncert_075.to_value(u.deg) ** 2)
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
    AGN_list['MASK_NAME'] = mask_name.replace(hcc_prefix, '')

    # Remove all objects that are outside of the image bounds
    AGN_list = AGN_list[np.all([0 <= AGN_list['x_pixel'],
                                AGN_list['x_pixel'] <= mask_size_x,
                                0 <= AGN_list['y_pixel'],
                                AGN_list['y_pixel'] <= mask_size_y], axis=0)]

    # Pass the cluster catalog through the mask to insure all objects are on image.
    AGN_list = AGN_list[np.where(mask_image[np.floor(AGN_list['y_pixel']).astype(int),
                                            np.floor(AGN_list['x_pixel']).astype(int)] == 1)]

    # Perform a rejection sampling based on the completeness value for each object.
    if not args.no_rejection:
        alpha = object_rng.uniform(0, 1, size=len(AGN_list))
        prob_reject = 1 / AGN_list['COMPLETENESS_CORRECTION']
        AGN_list = AGN_list[prob_reject >= alpha]

    return AGN_list


start_time = time()
# <editor-fold desc="Parameter Set up">
parser = ArgumentParser(description='Creates mock catalogs with input parameter set')
parser.add_argument('theta', help='Amplitude of cluster term', type=float)
parser.add_argument('--no-rejection', help='Turns off the secondary rejection sampling in the generation',
                    action='store_true')
args = parser.parse_args()

# Number of clusters to generate
n_cl = 308

# Set parameter values
theta_true = args.theta  # Amplitude
eta_true = 4.0  # Redshift slope
zeta_true = -1.0  # Mass slope
beta_true = 1.0  # Radial slope
rc_true = 0.1  # Core radius (in r500)
C_true = 0.333  # Background AGN surface density (in arcmin^-2)

# Set the maximum radius we will generate objects to as a factor of r500
max_radius = 5.0

# Set cluster center positional uncertainty
median_cluster_pos_uncert = 0.214 * u.arcmin

# SPT's 150 GHz beam size
SZ_theta_beam = 1.2 * u.arcmin

# Set 4.5 um magnitude bounds to define luminosity range
faint_end_45_apmag = 17.46  # Vega mag
bright_end_45_apmag = 10.45  # Vega mag

# Set cluster redshift error for color--redshift selection
delta_z = 0.05

# Set min and max values for SPT-SZ and SPTpol 100d IRAC color errors
sptsz_min_color_err, sptsz_max_color_err = 0.02, 0.16
sptpol_min_color_err, sptpol_max_color_err = 0.05, 0.22
# </editor-fold>

# <editor-fold desc="Data Generation">
# Read in the SPT cluster catalog. We will use real data to source our mock cluster properties.
Bocquet = Table.read(f'{hcc_prefix}Data_Repository/Catalogs/SPT/SPT_catalogs/2500d_cluster_sample_Bocquet18.fits')

# For the 20 common clusters between SPT-SZ 2500d and SPTpol 100d surveys we want to update the cluster information from
# the more recent survey. Thus, we will merge the SPT-SZ and SPTpol catalogs together.
Huang = Table.read(f'{hcc_prefix}Data_Repository/Catalogs/SPT/SPT_catalogs/sptpol100d_catalog_huang19.fits')

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
masks_files = [*glob.glob(f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/Masks/SPT-SZ_2500d/*.fits'),
               *glob.glob(f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/Masks/SPTpol_100d/*.fits')]

# Make sure all the masks have matches in the catalog
masks_files = [f for f in masks_files if re.search(r'SPT-CLJ\d+-\d+', f).group(0) in SPTcl['SPT_ID']]

# Select a number of masks at random, sorted to match the order in `full_spt_catalog`.
masks_bank = sorted([masks_files[i] for i in cluster_rng.choice(n_cl, size=n_cl)],
                    key=lambda x: re.search(r'SPT-CLJ\d+-\d+', x).group(0))

# Find the corresponding cluster IDs in the SPT catalog that match the masks we chose
spt_catalog_ids = [re.search(r'SPT-CLJ\d+-\d+', mask_name).group(0) for mask_name in masks_bank]
spt_catalog_mask = [np.where(SPTcl['SPT_ID'] == spt_id)[0][0] for spt_id in spt_catalog_ids]
selected_clusters = SPTcl['SPT_ID', 'RA', 'DEC', 'M500', 'REDSHIFT', 'THETA_CORE', 'XI', 'field'][spt_catalog_mask]

# We'll need the r500 radius for each cluster too.
selected_clusters['R500'] = (3 * selected_clusters['M500'] * u.Msun /
                             (4 * np.pi * 500 *
                              cosmo.critical_density(selected_clusters['REDSHIFT']).to(u.Msun / u.Mpc ** 3))) ** (1 / 3)

# Create cluster names
name_bank = ['SPT_Mock_{:03d}'.format(i) for i in range(n_cl)]

# Combine our data into a catalog
SPT_data = selected_clusters.copy()
SPT_data.rename_columns(['SPT_ID', 'RA', 'DEC'], ['orig_SPT_ID', 'SZ_RA', 'SZ_DEC'])
SPT_data['SPT_ID'] = name_bank
SPT_data['MASK_NAME'] = masks_bank

# Check that we have the correct mask and cluster data matched up.
assert np.all([spt_id in mask_name for spt_id, mask_name in zip(SPT_data['orig_SPT_ID'], SPT_data['MASK_NAME'])])

# Set up grid of radial positions to place AGN on (normalized by r500)
r_dist_r500 = np.linspace(0, max_radius, num=200)

# For the luminosities read in the filters and SED from which we will perform k-corrections on
irac_45_filter = SpectralElement.from_file(f'{hcc_prefix}Data_Repository/filter_curves/Spitzer_IRAC/'
                                           f'080924ch2trans_full.txt',
                                           wave_unit=u.um)
flamingos_j_filter = SpectralElement.from_file(f'{hcc_prefix}Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                                               'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt', wave_unit=u.nm)
qso2_sed = SourceSpectrum.from_file(f'{hcc_prefix}Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                    wave_unit=u.Angstrom, flux_unit=units.FLAM)

# To provide the IRAC colors (and object redshifts) for the objects we will need to create a realization of the SDWFS
# color--redshift plane from which we can later resample to assign values to the objects.
# Load in the SDWFS color-redshift distributions
with open(f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
          'SDWFS_color_redshift_kde_agn_weighted.pkl', 'rb') as f:
    kde_dict = pickle.load(f)
SDWFS_kde = kde_dict['SDWFS_kde']
AGN_kde = kde_dict['AGN_kde']

# Generate realizations
SDWFS_color_z = SDWFS_kde.resample(size=2000)
AGN_color_z = AGN_kde.resample(size=2000)

# Read in the completeness dictionaries
comp_sim_dir = f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim'
with open(f'{comp_sim_dir}/SPT-SZ_2500d/Results/SPTSZ_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json', 'r') as f, \
        open(f'{comp_sim_dir}/SPTpol_100d/Results/SPTpol_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json', 'r') as g:
    sptsz_comp_sim = json.load(f)
    sptpol_comp_sim = json.load(g)

# Because the SPT-SZ database still uses the original observed SPT IDs we need to update them to the official IDs
with open(f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_observed_to_official_ids.json', 'r') as f:
    obs_to_off_id = json.load(f)
for obs_id, off_id in obs_to_off_id.items():
    sptsz_comp_sim[off_id] = sptsz_comp_sim.pop(obs_id)

# Merge the dictionaries together replacing SPTpol/SSDF curves with SPT-SZ/targeted curves if available
sptcl_comp_sim = {**sptpol_comp_sim, **sptsz_comp_sim}

# For the membership degree calculation we need to read in the SDWFS color distribution
# Read in the number count distribution file
with open(f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
          'SDWFS_number_count_distribution_normed.json', 'r') as f:
    field_number_distribution = json.load(f)
field_number_counts = field_number_distribution['normalized_number_counts']
color_bins = field_number_distribution['color_bins']
color_bin_min, color_bin_max = np.min(color_bins), np.max(color_bins)

# Create an interpolation of our number count distribution, this will get passed to the membership degree function
color_probability_distribution = interp1d(color_bins, field_number_counts)

# </editor-fold>

catalog_start_time = time()
params_true = (theta_true, eta_true, zeta_true, beta_true, rc_true)

cluster_sample = SPT_data.copy()

# Run the catalog generation in parallel
with MPIPool() as pool:
    AGN_cats = list(pool.map(generate_mock_cluster, cluster_sample))

# Stack the individual cluster catalogs into a single master catalog
outAGN = vstack(AGN_cats)

# # Reorder the columns in the cluster for ascetic reasons.
# outAGN = outAGN['SPT_ID', 'SZ_RA', 'SZ_DEC', 'OFFSET_RA', 'OFFSET_DEC', 'HALF_OFFSET_RA', 'HALF_OFFSET_DEC',
#                 '075_OFFSET_RA', '075_OFFSET_DEC', 'x_pixel', 'y_pixel', 'RA', 'DEC',
#                 'REDSHIFT', 'M500', 'R500', 'RADIAL_SEP_ARCMIN', 'RADIAL_SEP_R500', 'RADIAL_SEP_ARCMIN_OFFSET',
#                 'RADIAL_SEP_R500_OFFSET', 'RADIAL_SEP_ARCMIN_HALF_OFFSET', 'RADIAL_SEP_R500_HALF_OFFSET',
#                 'RADIAL_SEP_ARCMIN_075_OFFSET', 'RADIAL_SEP_R500_075_OFFSET', 'MASK_NAME',
#                 'COMPLETENESS_CORRECTION', 'SELECTION_MEMBERSHIP', 'J_ABS_MAG', 'Cluster_AGN']
outAGN.write(f'{hcc_prefix}Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Final_tests/LF_tests/'
             f'variable_theta/'
             f'mock_AGN_catalog_t{theta_true:.3f}_e{eta_true:.2f}_z{zeta_true:.2f}_b{beta_true:.2f}_rc{rc_true:.3f}'
             f'_C{C_true:.3f}_maxr{max_radius:.2f}_clseed{cluster_seed}_objseed{object_seed}'
             f'_photometry_weighted_kde{"no_rejection" if args.no_rejection else "rejection"}.fits', overwrite=True)

# Print out statistics
number_of_clusters = len(outAGN.group_by('SPT_ID').groups.keys)
total_number = len(outAGN)
total_number_comp_corrected = outAGN['COMPLETENESS_CORRECTION'].sum()
total_number_corrected = np.sum(outAGN['COMPLETENESS_CORRECTION'] * outAGN['SELECTION_MEMBERSHIP'])
number_per_cluster = total_number_corrected / number_of_clusters
cluster_objs_corrected = np.sum(outAGN['COMPLETENESS_CORRECTION'][outAGN['CLUSTER_AGN'].astype(bool)]
                                * outAGN['SELECTION_MEMBERSHIP'][outAGN['CLUSTER_AGN'].astype(bool)])
background_objs_corrected = np.sum(outAGN['COMPLETENESS_CORRECTION'][~outAGN['CLUSTER_AGN'].astype(bool)]
                                   * outAGN['SELECTION_MEMBERSHIP'][~outAGN['CLUSTER_AGN'].astype(bool)])
median_z = np.median(outAGN['REDSHIFT'])
median_m = np.median(outAGN['M500'])

print(f"""Mock Catalog ({"no rejection sampling" if args.no_rejection else "rejection sampling"})
Cluster Seed: {cluster_seed}\tObject Seed: {object_seed}
Parameters:\t{params_true + (C_true,)}
Number of clusters:\t{number_of_clusters}
Objects Selected:\t{total_number}
Objects selected (completeness corrected):\t{total_number_comp_corrected:.2f}
Objects Selected (comp + membership corrected):\t{total_number_corrected:.2f}
Objects per cluster (comp + mem corrected):\t{number_per_cluster:.2f}
Cluster Objects (corrected):\t{cluster_objs_corrected:.2f}
Background Objects (corrected):\t{background_objs_corrected:.2f}
Median Redshift:\t{median_z:.2f}
Median Mass:\t{median_m:.2e}""")

print('Run time: {:.2f}s'.format(time() - catalog_start_time))
print('Total run time: {:.2f}s'.format(time() - start_time))
