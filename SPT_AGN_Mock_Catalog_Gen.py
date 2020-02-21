"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""
import glob
import re
from itertools import product
from time import time

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, join, unique, vstack
from astropy.wcs import WCS
from scipy import stats
from scipy.spatial.distance import cdist

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

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


def poisson_point_process(model, dx, dy=None, lower_dx=None, lower_dy=None):
    """
    Uses a spatial Poisson point process to generate AGN candidate coordinates.

    :param model: The model rate used in the Poisson distribution to determine the number of points being placed.
    :param dx: Upper bound on x-axis (lower bound is set to 0).
    :param dy: Upper bound on y-axis (lower bound is set to 0).
    :return coord: 2d numpy array of (x,y) coordinates of AGN candidates.
    """

    if lower_dx is None:
        lower_dx = 0
    if lower_dy is None:
        lower_dy = 0

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


def model_rate(z, m, r500, r_r500, params):
    """
    Our generating model.

    :param z: Redshift of the cluster
    :param m: M_500 mass of the cluster
    :param r500: r500 radius of the cluster
    :param r_r500: A vector of radii of objects within the cluster normalized by the cluster's r500
    :param params: Tuple of (theta, eta, zeta, beta, background)
    :return model: A surface density profile of objects as a function of radius
    """

    # Unpack our parameters
    theta, eta, zeta, beta, rc = params

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc) ** 2) ** (-1.5 * beta + 0.5)

    return model.value


def good_pixel_fraction(r, z, r500, image_name, center):
    # Read in the mask file and the mask file's WCS
    image, header = fits.getdata(image_name, header=True)
    image_wcs = WCS(header)

    # From the WCS get the pixel scale
    try:
        assert image_wcs.pixel_scale_matrix[0, 1] == 0.
        pix_scale = (image_wcs.pixel_scale_matrix[1, 1] * image_wcs.wcs.cunit[1]).to(u.arcsec)
    except AssertionError:
        cd = image_wcs.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        pix_scale = (cd_diag[1, 1] * image_wcs.wcs.cunit[1]).to(u.arcsec)

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

    # Convert our radius to pixels
    r_pix = r * r500 * cosmo.arcsec_per_kpc_proper(z).to(u.arcsec / u.Mpc) / pix_scale

    # Because we potentially integrate to larger radii than can be fit on the image we will need to increase the size of
    # our mask. To do this, we will pad the mask with a zeros out to the radius we need.
    # Find the width needed to pad the image to include the largest radius inside the image.
    width = ((int(round(np.max(r_pix.value) - center_pix[1])),
              int(round(np.max(r_pix.value) - (image.shape[0] - center_pix[1])))),
             (int(round(np.max(r_pix.value) - center_pix[0])),
              int(round(np.max(r_pix.value) - (image.shape[1] - center_pix[0])))))

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
    for i in np.arange(len(r_pix) - 1):
        pix_ring = large_image[np.where((r_pix[i] <= image_dists) & (image_dists < r_pix[i + 1]))]

        # Calculate the fraction
        good_pix_frac.append(np.sum(pix_ring) / len(pix_ring))

    return good_pix_frac


start_time = time()
# <editor-fold desc="Parameter Set up">

# Number of clusters to generate
n_cl = 238 + 55

# Set parameter values
# theta_list = np.arange(0., 10.)  # Amplitude.
theta_list = [10.84455443075414, 7.816208056554819, 6.040439268937491, 4.7112880155415615, 3.6418306647959326,
              2.6480608140771227, 2.153967368352823, 5.713202810466703, 4.654610477666601, 3.5618592216479232,
              2.692166590729115, 1.9319515427241647, 1.3294805204765892, 1.0955208036479187, 3.3288375689987593,
              2.573018795011335, 1.9384357865188957, 1.5051493498570752, 1.1056519674231713, 0.8332998742039303,
              0.6522056035137009, 1.7550384483896142, 1.367467599476686, 0.9700707193485799, 0.816643386076246,
              0.5635011047350256, 0.4500476814847764, 0.33671300715004504, 0.9811311359799195, 0.8094857691852279,
              0.5613426537155676, 0.4617062947605525, 0.3381850258330566, 0.23036825390562257, 0.1707340940143034,
              0.503918376040121, 0.35201288564602606, 0.29524313094438276, 0.23819861030919015, 0.16562369205603047,
              0.13633072402820387, 0.09865472513656726]
eta_list = np.arange(0., 6.)  # Redshift slope
zeta_list = np.arange(-1.75, 0, 0.25)  # Mass slope
beta_true = 1.0  # Radial slope
C_true = 0.371  # Background AGN surface density
rc_true = 0.1  # Core radius (in r500)

# Set the maximum radius we will generate objects to as a factor of r500
max_radius = 5.0

# Set cluster center positional uncertainty
median_cluster_pos_uncert = 0.214 * u.arcmin

# SPT's 150 GHz beam size
SZ_theta_beam = 1.2 * u.arcmin
# </editor-fold>

# <editor-fold desc="Data Generation">
# Read in the SPT cluster catalog. We will use real data to source our mock cluster properties.
bocquet = Table.read('Data/2500d_cluster_sample_Bocquet18.fits')
bocquet = bocquet[bocquet['M500'] != 0.0]  # So we only include confirmed clusters with measured masses.
# bocquet = bocquet[bocquet['REDSHIFT'] >= 0.5]
bocquet['M500'] *= 1e14  # So that our masses are in Msun instead of 1e14*Msun

# Read in the SPTpol 100d cluster catalog to include those clusters.
huang = Table.read('Data/sptpol100d_catalog_huang19.fits')
huang = huang[huang['M500'] > 0.0]  # Nick uses a '-1.0' for non-confirmations for redshift and mass.
huang['M500'] *= 1e14  # Same reason as Bocquet's catalog.

# Standardize the column names in the Huang catalog to match the Bocquet catalog
huang.rename_column('Dec', 'DEC')
huang.rename_column('redshift', 'REDSHIFT')
huang.rename_column('redshift_unc', 'REDSHIFT_UNC')
huang.rename_column('xi', 'XI')
huang.rename_column('theta_core', 'THETA_CORE')

# Merge the two catalogs
full_spt_catalog = join(bocquet, huang, join_type='outer')
full_spt_catalog.sort(keys=['SPT_ID', 'field'])  # Sub-sorting by 'field' puts Huang entries first
full_spt_catalog = unique(full_spt_catalog, keys='SPT_ID', keep='first')  # Keeping Huang entries over Bocquet
full_spt_catalog.sort(keys='SPT_ID')  # Resort by ID.

# For our masks, we will co-op the masks for the real clusters.
masks_files = [*glob.glob('Data/Masks/*.fits'),
               *glob.glob('Data/SPTPol/masks/*.fits')]

# Make sure all the masks have matches in the catalog
masks_files = [f for f in masks_files if re.search(r'SPT-CLJ\d+-\d+', f).group(0) in full_spt_catalog['SPT_ID']]

# Select a number of masks at random, sorted to match the order in `full_spt_catalog`.
masks_bank = sorted([masks_files[i] for i in cluster_rng.choice(n_cl, size=n_cl)],
                    key=lambda x: re.search(r'SPT-CLJ\d+-\d+', x).group(0))

# Find the corresponding cluster IDs in the SPT catalog that match the masks we chose
spt_catalog_ids = [re.search(r'SPT-CLJ\d+-\d+', mask_name).group(0) for mask_name in masks_bank]
# spt_catalog_idx = np.any([full_spt_catalog['SPT_ID'] == catalog_id for catalog_id in spt_catalog_ids], axis=0)
# spt_catalog_mask = np.isin(full_spt_catalog['SPT_ID'], spt_catalog_ids, assume_unique=True)
spt_catalog_mask = [np.where(full_spt_catalog['SPT_ID'] == spt_id)[0][0] for spt_id in spt_catalog_ids]
selected_clusters = full_spt_catalog['SPT_ID', 'RA', 'DEC', 'M500', 'REDSHIFT', 'THETA_CORE', 'XI'][spt_catalog_mask]

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
# </editor-fold>

for theta_true, (eta_true, zeta_true) in zip(theta_list, product(eta_list, zeta_list)):
    # for eta_true, zeta_true in product(eta_list, zeta_list):
    #     for theta_true in theta_list:
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

        # Calculate the model values for the AGN candidates in the cluster
        model_cluster_agn = model_rate(z_cl, m500_cl, r500_cl, r_dist_r500, params_true)

        # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
        # the cluster.
        max_rate = np.max(model_cluster_agn)  # r500^-2 units
        max_rate_inv_pix2 = ((max_rate / r500_cl ** 2) * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) ** 2
                             * mask_pixel_scale.to(u.arcmin) ** 2)

        # Set the bounding box for the object placement
        SZ_center_pix = SZ_center_skycoord.to_pixel(wcs=w, origin=0, mode='wcs')
        upper_x = SZ_center_pix[0] + mask_radius_pix
        upper_y = SZ_center_pix[1] + mask_radius_pix
        lower_x = SZ_center_pix[0] - mask_radius_pix
        lower_y = SZ_center_pix[1] - mask_radius_pix

        # Simulate the AGN using the spatial Poisson point process.
        cluster_agn_coords_pix = poisson_point_process(max_rate_inv_pix2, dx=upper_x, dy=upper_y,
                                                       lower_dx=lower_x, lower_dy=lower_y)

        # Find the radius of each point placed scaled by the cluster's r500 radius
        cluster_agn_skycoord = SkyCoord.from_pixel(cluster_agn_coords_pix[0], cluster_agn_coords_pix[1],
                                                   wcs=w, origin=0, mode='wcs')
        radii_arcmin = SZ_center_skycoord.separation(cluster_agn_skycoord).to(u.arcmin)
        radii_r500 = radii_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl

        # Filter the candidates through the model to establish the radial trend in the data.
        rate_at_rad = model_rate(z_cl, m500_cl, r500_cl, radii_r500, params_true)

        # Our rejection rate is the model rate at the radius scaled by the maximum rate
        prob_reject = rate_at_rad / max_rate

        # Draw a random number for each candidate
        alpha = object_rng.uniform(0, 1, len(rate_at_rad))

        # Perform the rejection sampling
        cluster_agn_final = cluster_agn_skycoord[np.where(prob_reject >= alpha)]
        cluster_agn_final_pix = np.array(cluster_agn_final.to_pixel(w, origin=0, mode='wcs'))

        # Generate background sources
        background_rate = C_true / u.arcmin ** 2 * mask_pixel_scale.to(u.arcmin) ** 2
        background_agn_pix = poisson_point_process(background_rate, dx=upper_x, dy=upper_y,
                                                   lower_dx=lower_x, lower_dy=lower_y)

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

        # Create a flag indicating if the object is a cluster member
        AGN_list['Cluster_AGN'] = np.concatenate((np.full_like(cluster_agn_final_pix[0], True),
                                                  np.full_like(background_agn_pix[0], False)))

        # Convert the pixel coordinates to RA/Dec coordinates
        agn_coords_skycoord = SkyCoord.from_pixel(AGN_list['x_pixel'], AGN_list['y_pixel'], wcs=w, origin=0,
                                                  mode='wcs')
        AGN_list['RA'] = agn_coords_skycoord.ra
        AGN_list['DEC'] = agn_coords_skycoord.dec

        # Shift the cluster center away from the true center within the 1-sigma SZ positional uncertainty
        cluster_pos_uncert = np.sqrt(SZ_theta_beam ** 2 + SZ_theta_core ** 2) / SZ_xi
        AGN_list['CENTER_POS_UNC_ARCMIN_1sigma'] = cluster_pos_uncert
        offset_SZ_center = cluster_rng.multivariate_normal(
            (SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
            np.eye(2) * cluster_pos_uncert.to_value(u.deg) ** 2)
        offset_SZ_center_skycoord = SkyCoord(offset_SZ_center[0], offset_SZ_center[1], unit='deg')
        AGN_list['OFFSET_RA'] = offset_SZ_center_skycoord.ra
        AGN_list['OFFSET_DEC'] = offset_SZ_center_skycoord.dec

        # Additionally, decrease the positional uncertainty to half of the true value
        cluster_pos_uncert_half = cluster_pos_uncert / 2
        half_offset_SZ_center = cluster_rng.multivariate_normal(
            (SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
            np.eye(2) * cluster_pos_uncert_half.to_value(u.deg) ** 2)
        half_offset_SZ_center_skycoord = SkyCoord(half_offset_SZ_center[0], half_offset_SZ_center[1], unit='deg')
        AGN_list['HALF_OFFSET_RA'] = half_offset_SZ_center_skycoord.ra
        AGN_list['HALF_OFFSET_DEC'] = half_offset_SZ_center_skycoord.dec

        cluster_pos_uncert_075 = cluster_pos_uncert * 0.75
        threequarters_offset_SZ_center = cluster_rng.multivariate_normal(
            (SZ_center_skycoord.ra.value, SZ_center_skycoord.dec.value),
            np.eye(2) * cluster_pos_uncert_half.to_value(u.deg) ** 2)
        threequarters_offset_SZ_center_skycoord = SkyCoord(threequarters_offset_SZ_center[0],
                                                           threequarters_offset_SZ_center[1], unit='deg')
        AGN_list['075_OFFSET_RA'] = threequarters_offset_SZ_center_skycoord.ra
        AGN_list['075_OFFSET_DEC'] = threequarters_offset_SZ_center_skycoord.dec

        # Calculate the radii of the final AGN scaled by the cluster's r500 radius
        r_final_arcmin = SZ_center_skycoord.separation(agn_coords_skycoord).to(u.arcmin)
        r_final_r500 = r_final_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl
        AGN_list['RADIAL_SEP_ARCMIN'] = r_final_arcmin
        AGN_list['RADIAL_SEP_R500'] = r_final_r500

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

        # Pass the cluster catalog through the quarter mask to insure all objects are on image.
        AGN_list = AGN_list[np.where(mask_image[np.floor(AGN_list['y_pixel']).astype(int),
                                                np.floor(AGN_list['x_pixel']).astype(int)] == 1)]

        AGN_cats.append(AGN_list)

    # Stack the individual cluster catalogs into a single master catalog
    outAGN = vstack(AGN_cats)

    # Reorder the columns in the cluster for ascetic reasons.
    outAGN = outAGN['SPT_ID', 'SZ_RA', 'SZ_DEC', 'OFFSET_RA', 'OFFSET_DEC', 'HALF_OFFSET_RA', 'HALF_OFFSET_DEC',
                    '075_OFFSET_RA', '075_OFFSET_DEC', 'x_pixel', 'y_pixel', 'RA', 'DEC',
                    'REDSHIFT', 'M500', 'R500', 'RADIAL_SEP_ARCMIN', 'RADIAL_SEP_R500', 'RADIAL_SEP_ARCMIN_OFFSET',
                    'RADIAL_SEP_R500_OFFSET', 'RADIAL_SEP_ARCMIN_HALF_OFFSET', 'RADIAL_SEP_R500_HALF_OFFSET',
                    'RADIAL_SEP_ARCMIN_075_OFFSET', 'RADIAL_SEP_R500_075_OFFSET', 'MASK_NAME', 'Cluster_AGN']

    print('\n------\nparameters: {param}\nTotal number of clusters: {cl} \t Total number of objects: {agn}'
          .format(param=params_true + (C_true,), cl=len(outAGN.group_by('SPT_ID').groups.keys), agn=len(outAGN)))
    # Path(f'Data/MCMC/Mock_Catalog/Catalogs/Final_tests/Slope_tests/trial_2/'
    #      f'e{eta_true:.2f}_z{zeta_true:.2f}').mkdir(parents=True, exist_ok=True)
    outAGN.write(
        f'Data/MCMC/Mock_Catalog/Catalogs/Final_tests/Slope_tests/trial_6/realistic/'
        f'mock_AGN_catalog_t{theta_true:.3f}_e{eta_true:.2f}_z{zeta_true:.2f}_b{beta_true:.2f}'
        f'_C{C_true:.3f}_rc{rc_true:.3f}_maxr{max_radius:.2f}'
        f'_clseed{cluster_seed}_objseed{object_seed}_slope_test.cat',
        format='ascii', overwrite=True)

    print('Run time: {:.2f}s'.format(time() - catalog_start_time))
print('Total run time: {:.2f}s'.format(time() - start_time))
