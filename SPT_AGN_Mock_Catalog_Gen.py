"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""
import glob
import json
import re
from time import time

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, join, unique, vstack
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from scipy import stats
from scipy.interpolate import lagrange, interp1d
from synphot import SpectralElement, SourceSpectrum, units

from k_correction import k_corr_abs_mag

# hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
hcc_prefix = ''

# Set our cosmology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

# Set up rng
seed = 3775
rng = np.random.default_rng(seed)
print(f'Using RNG seed: {seed}')

tez_pattern = re.compile(r'[tez](-*\d+.\d+|\d+)')

# Set up the luminosity and density evolution using the fits from Assef+11 Table 2
z_i = [0.25, 0.5, 1., 2., 4.]
m_star_z_i = [-23.51, -24.64, -26.10, -27.08]
# m_star_z_i = [-24.52, -25.16, -25.81, -25.10]  # PLE
phi_star_z_i = [-3.41, -3.73, -4.17, -4.65, -5.77]
m_star = lagrange(z_i[1:], m_star_z_i)
log_phi_star = lagrange(z_i, phi_star_z_i)


# Provide a random variable for the luminosity function
class LFpdf(stats.rv_continuous):
    def __init__(self, z, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._z = z
        self._normalization = self._norm(self._z)

    def _norm(self, z):
        _a, _b = self._get_support()
        int_grid = np.linspace(_a, _b, num=200)
        # return quad(lambda x: luminosity_function(x, z).value, a=_a, b=_b)[0]
        return np.trapz(luminosity_function(int_grid, z).value, int_grid)

    def _pdf(self, abs_mag, *args):
        return luminosity_function(abs_mag, self._z).value / self._normalization

    def _cdf(self, x, *args):
        xx = np.atleast_1d(x)
        _a, _b = self._get_support()
        res = np.zeros_like(xx)
        for ii, mag in enumerate(xx):
            mag_range = np.arange(_a, mag + 0.1, 0.1)
            res[ii] = np.sum(self._pdf(mag_range))
        return res

    def _ppf(self, q, *args):
        _a, _b = self._get_support()
        mag_range = np.linspace(_a - 5, _b + 5, num=100)
        cdf_range = self._cdf(mag_range)
        ppf_interp = interp1d(cdf_range, mag_range)
        return ppf_interp(q)

    def _rvs(self, *args, size=None, random_state=None):
        if size is None:
            size = 1
        if random_state is None:
            random_state = np.random.default_rng()

        rands = random_state.uniform(size=size)

        return self._ppf(rands)


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

    # L/L_*(z) = 10**(0.4 * (M_*(z) - M))
    L_L_star = 10 ** (0.4 * (m_star(redshift) - abs_mag))

    # Phi*(z) = 10**(log(Phi*(z))
    phi_star = 10 ** log_phi_star(redshift) * (cosmo.h / u.Mpc) ** 3
    # phi_star = 10 ** (-4.53) * (cosmo.h / u.Mpc)**3  # PLE

    # QLF slopes
    alpha1 = -3.35  # alpha in Table 2
    alpha2 = -0.37  # beta in Table 2
    # alpha1 = -3.30  # PLE
    # alpha2 = -1.42  # PLE

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
    model : np.ndarray
        A surface density profile of objects as a function of radius and luminosity.
    """

    # Unpack our parameters
    theta, eta, zeta, beta, rc = params

    # Luminosity function number
    LF = cosmo.angular_diameter_distance(z) ** 2 * r500 * np.trapz(luminosity_function(j_mag, z), j_mag, axis=0)
    # LF = 1.

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta * LF

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc) ** 2) ** (-1.5 * beta + 0.5)

    return model  # .value


def generate_mock_cluster(cluster: Table, color_threshold: float, c_true: float) -> Table:
    """Task function to create the line-of-sight cluster catalogs."""
    spt_id = cluster['SPT_ID']
    mask_name = cluster['MASK_NAME']
    z_cl = cluster['REDSHIFT']
    m500_cl = cluster['M500'] * u.Msun
    r500_cl = cluster['R500'] * u.Mpc
    SZ_center = cluster['SZ_RA', 'SZ_DEC']
    SZ_theta_core = cluster['THETA_CORE'] * u.arcmin
    SZ_xi = cluster['XI']

    # Make a cut in the SDWFS catalog to only include objects with selection memberships >= 50%
    sdwfs_agn_mu_cut = sdwfs_agn[sdwfs_agn[f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'] >= 0.5]
    # sdwfs_agn_mu_cut = sdwfs_agn

    # Read in the mask's WCS for the pixel scale and making SkyCoords
    w = WCS(mask_name)
    mask_pixel_scale = w.proj_plane_pixel_scales()[0]

    # Also get the mask's image size (- 1 to account for the shift between index and length)
    mask_size_x = w.pixel_shape[0] - 1
    mask_size_y = w.pixel_shape[1] - 1
    mask_radius_pix = (max_radius * r500_cl * cosmo.arcsec_per_kpc_proper(z_cl).to(mask_pixel_scale.unit / u.Mpc)
                       / mask_pixel_scale).value

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

    # Background Catalog
    # Generate background sources using a Poisson point process but skipping the rejection sampling step from above.
    background_rate = c_true / u.arcmin ** 2 * mask_pixel_scale.to(u.arcmin) ** 2
    background_agn_pix = poisson_point_process(background_rate, dx=upper_x, dy=upper_y,
                                               lower_dx=lower_x, lower_dy=lower_y)

    # For each background AGN we will also need a random completeness value, degree of membership, and J-band abs. mag.
    bkg_cat_df = sdwfs_agn_mu_cut.to_pandas().sample(n=background_agn_pix.shape[-1], replace=True, random_state=rng)
    bkg_cat = Table.from_pandas(bkg_cat_df)
    bkg_cat['x_pixel'] = background_agn_pix[0]
    bkg_cat['y_pixel'] = background_agn_pix[1]

    # Filter for only the columns we care about
    bkg_cat = bkg_cat['x_pixel', 'y_pixel', 'REDSHIFT', 'COMPLETENESS_CORRECTION',
                      f'SELECTION_MEMBERSHIP_{color_threshold:.2f}', 'J_ABS_MAG']
    bkg_cat.rename_columns(['REDSHIFT', f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'],
                           ['galaxy_redshift', 'SELECTION_MEMBERSHIP'])

    # Add flag to background objects
    bkg_cat['CLUSTER_AGN'] = np.full_like(bkg_cat['x_pixel'], False)

    # Cluster Catalog
    # Calculate the model values for the AGN candidates in the cluster
    model_cluster_agn = model_rate(params_true, z_cl, m500_cl, r500_cl, r_grid, j_grid)
    # if spt_id == 'SPT_Mock_000':
    #     fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #     ax.plot_surface(R, J, model_cluster_agn)
    #     ax.set(xlabel=r'$r/r_{500}$', ylabel=r'$M_J$', zlabel=r'$N(r,M_J)$')
    #     # ax.view_init(elev=0, azim=0)
    #     plt.show()

    # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
    # the cluster. Then convert to pix^-2 units.
    max_rate = np.max(model_cluster_agn)  # r500^-2 units
    max_rate_inv_pix2 = ((max_rate / r500_cl ** 2) * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) ** 2
                         * mask_pixel_scale.to(u.arcmin) ** 2)

    # Simulate the AGN using the spatial Poisson point process.
    cluster_agn_coords_pix = poisson_point_process(max_rate_inv_pix2, dx=upper_x, dy=upper_y,
                                                   lower_dx=lower_x, lower_dy=lower_y)

    # For the cluster, we need to select only objects within a redshift range of the cluster redshift.
    sdwfs_agn_at_z = sdwfs_agn_mu_cut[np.abs(sdwfs_agn_mu_cut['REDSHIFT'] - z_cl) <= delta_z]

    # plot_lf_cluster(sdwfs_agn_at_z['J_ABS_MAG'], z_cl, r500_cl, cluster_id='Raw SDWFS', before=True)

    # Now we will assign each cluster object a random completeness value, degree of membership, and J-band abs. mag.
    cl_cat_df = sdwfs_agn_at_z.to_pandas().sample(n=cluster_agn_coords_pix.shape[-1], replace=True,
                                                  random_state=rng)
    cl_cat = Table.from_pandas(cl_cat_df)
    cl_cat['x_pixel'] = cluster_agn_coords_pix[0]
    cl_cat['y_pixel'] = cluster_agn_coords_pix[1]

    # Filter for only the columns we care about
    cl_cat = cl_cat['x_pixel', 'y_pixel', 'REDSHIFT', 'COMPLETENESS_CORRECTION',
                    f'SELECTION_MEMBERSHIP_{color_threshold:.2f}', 'J_ABS_MAG']
    cl_cat.rename_columns(['REDSHIFT', f'SELECTION_MEMBERSHIP_{color_threshold:.2f}'],
                          ['galaxy_redshift', 'SELECTION_MEMBERSHIP'])

    # # Draw J-band luminosities for each object from the luminosity function at the cluster redshift
    # lf_cl_rv = LFpdf(z=z_cl, a=bright_end_j_absmag, b=faint_end_j_absmag)
    # cluster_agn_j_abs_mag = lf_cl_rv.rvs(size=cluster_agn_coords_pix.shape[-1], random_state=rng)
    #
    # # For testing, overwrite the empirical J-band absolute magnitudes with the ones drawn directly from the LF
    # cl_cat['J_ABS_MAG'] = cluster_agn_j_abs_mag

    # Do a quick rejection sampling method to build the luminosities
    # lf_norm = np.trapz(luminosity_function(j_grid, z_cl), j_grid)
    # abs_mag_cans = rng.uniform(bright_end_j_absmag, faint_end_j_absmag, size=10 * len(cl_cat))
    # lf_at_mag = luminosity_function(abs_mag_cans, z_cl) / lf_norm
    # alpha = rng.uniform(0, 1, size=abs_mag_cans.size)
    # cluster_agn_j_abs_mag = abs_mag_cans[lf_at_mag >= alpha][:len(cl_cat)]
    #
    # cl_cat['J_ABS_MAG'] = cluster_agn_j_abs_mag
    #
    # plot_lf_cluster(cl_cat['J_ABS_MAG'], z_cl, r500_cl, spt_id, before=True)

    # Find the radius of each point placed scaled by the cluster's r500 radius
    cluster_agn_skycoord = SkyCoord.from_pixel(cl_cat['x_pixel'], cl_cat['y_pixel'], wcs=w, origin=0, mode='wcs')
    radii_arcmin = SZ_center_skycoord.separation(cluster_agn_skycoord).to(u.arcmin)
    radii_r500 = radii_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl

    # Filter the candidates through the model to establish the radial trend in the data.
    rate_at_rad = model_rate(params_true, z_cl, m500_cl, r500_cl, radii_r500, j_grid)

    # Our rejection rate is the model rate at the radius scaled by the maximum rate
    prob_reject = rate_at_rad / max_rate

    # Draw a random number for each candidate
    alpha = rng.uniform(0., 1., size=len(rate_at_rad))

    # Perform the rejection sampling
    cl_cat = cl_cat[prob_reject >= alpha]

    # Add flag to cluster objects
    cl_cat['CLUSTER_AGN'] = np.full_like(cl_cat['x_pixel'], True)

    # plot_lf_cluster(cl_cat['J_ABS_MAG'], z_cl, r500_cl, spt_id, before=False)

    # Concatenate the cluster sources with the background sources
    los_cat = vstack([cl_cat, bkg_cat])

    try:
        los_cat['SPT_ID'] = spt_id
    except TypeError:
        print(f'{params_true}\n{spt_id = }: {len(cl_cat) = }, {len(bkg_cat) = }')
        return los_cat

    los_cat['SZ_RA'] = SZ_center_skycoord.ra
    los_cat['SZ_DEC'] = SZ_center_skycoord.dec
    los_cat['M500'] = m500_cl
    los_cat['REDSHIFT'] = z_cl
    los_cat['R500'] = r500_cl
    los_cat['MASK_NAME'] = mask_name

    # Convert the pixel coordinates to RA/Dec coordinates
    los_coords_skycoord = SkyCoord.from_pixel(los_cat['x_pixel'], los_cat['y_pixel'], wcs=w, origin=0, mode='wcs')
    los_cat['RA'] = los_coords_skycoord.ra
    los_cat['DEC'] = los_coords_skycoord.dec

    # Calculate the radii of the final AGN scaled by the cluster's r500 radius
    los_cat['RADIAL_SEP_ARCMIN'] = SZ_center_skycoord.separation(los_coords_skycoord).to(u.arcmin)
    los_cat['RADIAL_SEP_R500'] = (los_cat['RADIAL_SEP_ARCMIN'] * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin)
                                  / r500_cl)

    # <editor-fold desc="Miscentering">
    # # Shift the cluster center away from the true center within the 1-sigma SZ positional uncertainty
    # cluster_pos_uncert = np.sqrt(SZ_theta_beam ** 2 + SZ_theta_core ** 2) / SZ_xi
    # los_cat['CENTER_POS_UNC_ARCMIN_1sigma'] = cluster_pos_uncert
    # offset_SZ_center = rng.multivariate_normal((SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
    #                                            np.eye(2) * cluster_pos_uncert.to_value(u.deg) ** 2)
    # offset_SZ_center_skycoord = SkyCoord(offset_SZ_center[0], offset_SZ_center[1], unit='deg')
    # los_cat['OFFSET_RA'] = offset_SZ_center_skycoord.ra
    # los_cat['OFFSET_DEC'] = offset_SZ_center_skycoord.dec
    #
    # # Also calculate the radial distances based on the offset center.
    # los_cat['RADIAL_SEP_ARCMIN_OFFSET'] = (offset_SZ_center_skycoord.separation(agn_coords_skycoord).to(u.arcmin))
    # los_cat['RADIAL_SEP_R500_OFFSET'] = (los_cat['RADIAL_SEP_ARCMIN_OFFSET'] * cosmo.kpc_proper_per_arcmin(z_cl)
    #                                      .to(u.Mpc / u.arcmin) / r500_cl)
    #
    # # Decrease the positional uncertainty to 75% of the true value
    # cluster_pos_uncert_075 = cluster_pos_uncert * 0.75
    # threequarters_offset_SZ_center = rng.multivariate_normal(
    #     (SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
    #     np.eye(2) * cluster_pos_uncert_075.to_value(u.deg) ** 2)
    # threequarters_offset_SZ_center_skycoord = SkyCoord(threequarters_offset_SZ_center[0],
    #                                                    threequarters_offset_SZ_center[1], unit='deg')
    # los_cat['075_OFFSET_RA'] = threequarters_offset_SZ_center_skycoord.ra
    # los_cat['075_OFFSET_DEC'] = threequarters_offset_SZ_center_skycoord.dec
    #
    # # Calculate the radial distances based on the 3/4-offset center
    # los_cat['RADIAL_SEP_ARCMIN_075_OFFSET'] = (threequarters_offset_SZ_center_skycoord.separation(agn_coords_skycoord)
    #                                            .to(u.arcmin))
    # los_cat['RADIAL_SEP_R500_075_OFFSET'] = (los_cat['RADIAL_SEP_ARCMIN_075_OFFSET'] * cosmo.kpc_proper_per_arcmin(z_cl)
    #                                          .to(u.Mpc / u.arcmin) / r500_cl)
    #
    # # Decrease the positional uncertainty to half of the true value
    # cluster_pos_uncert_half = cluster_pos_uncert / 2
    # half_offset_SZ_center = rng.multivariate_normal((SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
    #                                                 np.eye(2) * cluster_pos_uncert_half.to_value(u.deg) ** 2)
    # half_offset_SZ_center_skycoord = SkyCoord(half_offset_SZ_center[0], half_offset_SZ_center[1], unit='deg')
    # los_cat['HALF_OFFSET_RA'] = half_offset_SZ_center_skycoord.ra
    # los_cat['HALF_OFFSET_DEC'] = half_offset_SZ_center_skycoord.dec
    #
    # # Also, calculate the radial distances based on the half-offset center
    # los_cat['RADIAL_SEP_ARCMIN_HALF_OFFSET'] = (half_offset_SZ_center_skycoord.separation(agn_coords_skycoord)
    #                                             .to(u.arcmin))
    # los_cat['RADIAL_SEP_R500_HALF_OFFSET'] = (los_cat['RADIAL_SEP_ARCMIN_HALF_OFFSET']
    #                                           * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl)
    # </editor-fold>

    # Select only objects within the max_radius
    los_cat = los_cat[los_cat['RADIAL_SEP_R500'] <= max_radius]

    # Read in the mask
    mask_image, mask_header = fits.getdata(mask_name, header=True)

    # Remove all objects that are outside of the image bounds
    los_cat = los_cat[np.all([0 <= los_cat['x_pixel'], los_cat['x_pixel'] <= mask_size_x,
                              0 <= los_cat['y_pixel'], los_cat['y_pixel'] <= mask_size_y], axis=0)]

    # Pass the cluster catalog through the mask to insure all objects are on image.
    mask_image = mask_image.astype(bool)
    los_cat = los_cat[mask_image[np.floor(los_cat['y_pixel']).astype(int), np.floor(los_cat['x_pixel']).astype(int)]]

    # # Perform a rejection sampling based on the completeness value for each object.
    alpha = rng.uniform(0, 1, size=len(los_cat))
    prob_reject = 1 / los_cat['COMPLETENESS_CORRECTION']
    los_cat['COMPLETENESS_REJECT'] = prob_reject >= alpha

    # image_center = SkyCoord.from_pixel(mask_size_x / 2, mask_size_y / 2, wcs=w, origin=0, mode='wcs')
    # los_coords = SkyCoord(los_cat['RA'], los_cat['DEC'], unit=u.deg)
    # image_center_sep = image_center.separation(los_coords).to(u.arcmin)
    # los_cat = los_cat[image_center_sep <= 2.5 * u.arcmin]

    # if spt_id == 'SPT_Mock_000':
    #     plot_mock_cluster(los_cat, cluster)

    return los_cat


def plot_lf_cluster(j_mag, redshift: float, r500: float, cluster_id: str, before: bool, weight=1.):
    j_bins = np.arange(j_mag.min(), j_mag.max() + 0.25, 0.25)
    j_bin_centers = np.diff(j_bins) / 2 + j_bins[:-1]

    dV = cosmo.comoving_volume(redshift)
    # dV = cosmo.angular_diameter_distance(redshift)**2 * r500
    dM = (j_bins[-1] - j_bins[0]) / (j_bins.size - 1)
    print(f'{cluster_id}: {dV = }, {dM = }')

    lf_at_z = luminosity_function(j_bin_centers, redshift)
    j_hist, _ = np.histogram(j_mag, bins=j_bins)

    title = f'{cluster_id} at z = {redshift:.2f} ({"before" if before else "after"})'

    _, ax = plt.subplots()
    ax.scatter(j_bin_centers, j_hist / dV / dM)
    # ax.plot(j_bin_centers, lf_at_z * dM)
    ax.set(title=title, xlabel=r'$M_J$', ylabel=r'$\Phi(M_J, z) dM_J$', yscale='log')
    plt.show()


def plot_mock_cluster(cat, cluster_catalog):
    # Show plot of combined line-of-sight positions
    cluster_objects = cat[cat['CLUSTER_AGN'].astype(bool)]
    background_objects = cat[~cat['CLUSTER_AGN'].astype(bool)]
    mask_img, mask_hdr = fits.getdata(cluster_catalog['MASK_NAME'], header=True)
    wcs = WCS(mask_hdr)
    _, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(mask_img, origin='lower', cmap='Greys_r')
    ax.scatter(background_objects['RA'], background_objects['DEC'], edgecolors='blue', facecolors='blue', alpha=0.4,
               label='Background', transform=ax.get_transform('world'))
    ax.scatter(cluster_objects['RA'], cluster_objects['DEC'], edgecolors='red', facecolors='red', alpha=0.6,
               label='Cluster', transform=ax.get_transform('world'))
    ax.scatter(cluster_catalog['SZ_RA'], cluster_catalog['SZ_DEC'], marker='+', c='k', s=50, label='Cluster Center',
               transform=ax.get_transform('world'))
    ax.legend()
    ax.set(title=f'{cluster_catalog["SPT_ID"]} at z = {cluster_catalog["REDSHIFT"]:.2f}', xlabel='Right Ascension',
           ylabel='Declination', aspect=1)
    plt.show()


def print_catalog_stats(catalog):
    number_of_clusters = len(catalog.group_by('SPT_ID').groups.keys)
    total_number = len(catalog)
    total_number_comp_corrected = catalog['COMPLETENESS_CORRECTION'].sum()
    total_number_corrected = np.sum(catalog['COMPLETENESS_CORRECTION'] * catalog['SELECTION_MEMBERSHIP'])
    number_per_cluster = total_number_corrected / number_of_clusters
    cluster_objs_corrected = np.sum(catalog['COMPLETENESS_CORRECTION'][catalog['CLUSTER_AGN'].astype(bool)])
    background_objs_corrected = np.sum(catalog['COMPLETENESS_CORRECTION'][~catalog['CLUSTER_AGN'].astype(bool)])
    median_z = np.median(catalog['REDSHIFT'])
    median_m = np.median(catalog['M500'])

    print(f"""Parameters:\t{params_true + (c0_true,)}
    Number of clusters:\t{number_of_clusters:,}
    Objects Selected:\t{total_number:,}
    Objects selected (completeness corrected):\t{total_number_comp_corrected:,.2f}
    Objects Selected (comp + membership corrected):\t{total_number_corrected:,.2f}
    Objects per cluster (comp + mem corrected):\t{number_per_cluster:,.2f}
    Cluster Objects (corrected):\t{cluster_objs_corrected:,.2f}
    Background Objects (corrected):\t{background_objs_corrected:,.2f}
    SNR (Cluster objects / Background objects):\t{cluster_objs_corrected / background_objs_corrected:,.2f}
    Median Redshift:\t{median_z:.2f}
    Median Mass:\t{median_m:.2e}""")


start_time = time()

# Read in the purity and surface density files
with (open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f,
      open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
           'SDWFS_background_prior_distributions_mu_cut_updated_cuts.json', 'r') as g):
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


# Read in the SNR-theta fit library
with open('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/eta_zeta_slopes/'
          'snr_to_theta_fits.json', 'r') as f:
    snr_theta_fits = json.load(f)
for name, fit_data in snr_theta_fits.items():
    snr_theta_fits[name] = np.poly1d(fit_data)

# <editor-fold desc="Parameter Set up">
# parser = ArgumentParser(description='Creates mock catalogs with input parameter set')
# parser.add_argument('theta', help='Amplitude of cluster term', type=float)
# args = parser.parse_args()

# Number of clusters to generate
n_cl = 308

# We'll boost the number of objects in our sample by duplicating this cluster by a factor.
cluster_amp = 1.

# Set parameter values
# theta_true = 50.0  # Amplitude
# eta_true = 4.0  # Redshift slope
# zeta_true = -1.0  # Mass slope
beta_true = 1.0  # Radial slope
rc_true = 0.1  # Core radius (in r500)
c0_true = agn_prior_surf_den(0.)  # Background AGN surface density (in arcmin^-2)

theta_range = np.arange(0.1, 7., 0.5)
eta_range = [-5., -3., 0., 3., 4., 5.]
zeta_range = [-2., -1, 0., 1., 2.]

# Using our targeted SNR, determine the cluster amplitude parameter needed.
target_snr = 0.23
targeted_snr_theta = Table(rows=[[name, theta_snr(target_snr)] for name, theta_snr in snr_theta_fits.items()],
                           names=['catalog', 'theta'])

# We will amplify our true parameters to increase the SNR
# theta_true *= cluster_amp
# c0_true *= cluster_amp

# Set the maximum radius we will generate objects to as a factor of r500
max_radius = 5.0

# Set cluster center positional uncertainty
median_cluster_pos_uncert = 0.214 * u.arcmin

# SPT's 150 GHz beam size
SZ_theta_beam = 1.2 * u.arcmin

# Set 4.5 um magnitude bounds to define luminosity range
faint_end_45_apmag = 17.48  # Vega mag
bright_end_45_apmag = 10.45  # Vega mag

# Set cluster redshift error for color--redshift selection
delta_z = 0.1
# </editor-fold>

# <editor-fold desc="Data Generation">
# Read in the SDWFS IRAGN catalog to use to populate photometric information from
sdwfs_agn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')

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

# Select a number of masks at random, sorted to match the order in `SPTcl`.
masks_bank = sorted([masks_files[i] for i in rng.choice(n_cl, size=n_cl, replace=False)],
                    key=lambda x: re.search(r'SPT-CLJ\d+-\d+', x).group(0))

# Find the corresponding cluster IDs in the SPT catalog that match the masks we chose
spt_catalog_ids = [re.search(r'SPT-CLJ\d+-\d+', mask_name).group(0) for mask_name in masks_bank]
spt_catalog_mask = [np.where(SPTcl['SPT_ID'] == spt_id)[0][0] for spt_id in spt_catalog_ids]
SPT_data = SPTcl['SPT_ID', 'RA', 'DEC', 'M500', 'REDSHIFT', 'THETA_CORE', 'XI', 'field'][spt_catalog_mask]

# We'll need the r500 radius for each cluster too.
SPT_data['R500'] = (3 * SPT_data['M500'] * u.Msun / (4 * np.pi * 500 * cosmo.critical_density(SPT_data['REDSHIFT'])
                                                     .to(u.Msun / u.Mpc ** 3))) ** (1 / 3)

# Create cluster names
name_bank = [f'SPT_Mock_{i:03d}' for i in range(n_cl)]

# Combine our data into a catalog
SPT_data.rename_columns(['SPT_ID', 'RA', 'DEC'], ['orig_SPT_ID', 'SZ_RA', 'SZ_DEC'])
SPT_data['SPT_ID'] = name_bank
SPT_data['MASK_NAME'] = masks_bank

# Remove the clusters with z < 0.2 as we have problems assigning photometric data from SDWFS in the redshift bin
SPT_data = SPT_data[SPT_data['REDSHIFT'] >= 0.2]

# Check that we have the correct mask and cluster data matched up.
assert np.all([spt_id in mask_name for spt_id, mask_name in zip(SPT_data['orig_SPT_ID'], SPT_data['MASK_NAME'])])

# Set up grid of radial positions to place AGN on (normalized by r500)
r_grid = np.linspace(0., max_radius, num=200)

# For the luminosities read in the filters and SED from which we will perform k-corrections on
irac_36_filter = SpectralElement.from_file(f'{hcc_prefix}Data_Repository/filter_curves/Spitzer_IRAC/'
                                           f'080924ch2trans_full.txt', wave_unit=u.um)
irac_45_filter = SpectralElement.from_file(f'{hcc_prefix}Data_Repository/filter_curves/Spitzer_IRAC/'
                                           f'080924ch2trans_full.txt', wave_unit=u.um)
flamingos_j_filter = SpectralElement.from_file(f'{hcc_prefix}Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                                               'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt', wave_unit=u.nm)
qso2_sed = SourceSpectrum.from_file(f'{hcc_prefix}Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                    wave_unit=u.Angstrom, flux_unit=units.FLAM)
# </editor-fold>

catalog_start_time = time()
# for theta_true, eta_true, zeta_true in np.array(np.meshgrid(theta_range, eta_range, zeta_range)).T.reshape(-1, 3):
for eta_true, zeta_true in np.array(np.meshgrid(eta_range, zeta_range)).T.reshape(-1, 2):
    # For our chosen redshift and mass slopes, use the amplitude value that produces a cluster-to-background SNR of 13.
    theta_true = snr_theta_fits[f'{(eta_true, zeta_true)}'](target_snr)

    # theta_true, eta_true, zeta_true = 6.6, -5.0, 1.0
    params_true = (theta_true, eta_true, zeta_true, beta_true, rc_true)
    # params_true = (6.6, -5.0, 1.0, 1.0, 0.1)

    # Find the appropriate color thresholds for our clusters
    color_thresholds = [agn_purity_color(z) for z in SPT_data['REDSHIFT']]

    # Set the redshift dependent background rates
    c_truths = np.array([agn_prior_surf_den(z) for z in SPT_data['REDSHIFT']])
    c_err_truths = np.array([agn_prior_surf_den_err(z) for z in SPT_data['REDSHIFT']])

    # We will amplify the true parameters by the number of clusters in the sample.
    c_truths *= cluster_amp
    c_err_truths *= cluster_amp

    # # Run the catalog generation in parallel
    # with MPIPool() as pool:
    #     AGN_cats = list(pool.map(generate_mock_cluster, cluster_sample))
    AGN_cats = []
    for cluster_catalog, cluster_color_threshold, bkg_rate_true in zip(SPT_data, color_thresholds, c_truths):
        AGN_cats.append(generate_mock_cluster(cluster_catalog, cluster_color_threshold, bkg_rate_true))

    # Stack the individual cluster catalogs into a single master catalog
    outAGN = vstack(AGN_cats)
    filename = (
        f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/eta_zeta_slopes/'
        f'targeted_snr/308cl/snr_0.23/'
        f'mock_AGN_catalog_t{theta_true:.3f}_e{eta_true:.2f}_z{zeta_true:.2f}_b{beta_true:.2f}_rc{rc_true:.3f}'
        f'_C{c0_true:.3f}_maxr{max_radius:.2f}_seed{seed}_{n_cl}x{cluster_amp}_photComp_tez_grid.fits')
    outAGN.write(filename, overwrite=True)
    print(filename)

    # Print out statistics
    print(f'RNG Seed: {seed}')
    print('Mock Catalog (no rejection sampling)')
    print_catalog_stats(outAGN)

# print('-----\n')
# outAGN_rejection = outAGN[outAGN['COMPLETENESS_REJECT'].astype(bool)]
# print('Mock Catalog (with rejection sampling)')
# print_catalog_stats(outAGN_rejection)

print(f'Total run time: {time() - start_time:.2f}s')
