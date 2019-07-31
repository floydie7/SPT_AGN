"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""

import re
from itertools import product
from os import listdir
from time import time

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from scipy import stats
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from small_poisson import small_poisson

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Set our cosmology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

# Generate a random seed
# rand_seed = np.random.randint(1024)
rand_seed = 890
print('Random Seed: {}'.format(rand_seed))

# Set our random seed
np.random.seed(rand_seed)  # Previously 123


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
    x = np.random.uniform(lower_dx, dx, size=p)
    y = np.random.uniform(lower_dy, dy, size=p)

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
    theta, eta, zeta, beta = params

    # The cluster's core radius in units of r500
    rc_r500 = 0.1 * u.Mpc / r500

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc_r500) ** 2) ** (-1.5 * beta + 0.5)

    return model.value


def good_pixel_fraction(r, z, r500, image_name, center):
    # Read in the mask file and the mask file's WCS
    image, header = fits.getdata(image_name, header=True)
    image_wcs = WCS(header)

    # From the WCS get the pixel scale
    pix_scale = (image_wcs.pixel_scale_matrix[1, 1] * u.deg).to(u.arcsec)

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
n_cl = 195

# Set parameter values
theta_list = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 12.0]
# theta_true = 12     # Amplitude.
eta_true = 1.2       # Redshift slope
zeta_true = -1.0     # Mass slope
beta_true = 0.5      # Radial slope
C_true = 0.371       # Background AGN surface density

for theta_true in theta_list:
    params_true = (theta_true, eta_true, zeta_true, beta_true)

    # Set the maximum radius we will generate objects to as a factor of r500
    max_radius = 5.0

    # Number of bins to use to plot our sampled data points
    num_bins = 30
    # </editor-fold>

    # <editor-fold desc="Data Generation">
    # Read in the SPT cluster catalog. We will use real data to source our mock cluster properties.
    bocquet = Table.read('Data/2500d_cluster_sample_Bocquet18.fits')
    bocquet = bocquet[bocquet['M500'] != 0.0]  # So we only include confirmed clusters with measured masses.
    bocquet = bocquet[bocquet['REDSHIFT'] >= 0.5]
    bocquet['M500'] *= 1e14  # So that our masses are in Msun instead of 1e14*Msun

    # For our masks, we will co-op the masks for the real clusters.
    mask_dir = 'Data/Masks/'
    masks_files = [f for f in listdir(mask_dir) if not f.startswith('.') and f not in ['no_masks', 'quarter_masks']]

    # Make sure all the masks have matches in the catalog
    masks_files = [f for f in masks_files if re.search('SPT-CLJ(.+?)_', f).group(0)[:-1] in bocquet['SPT_ID']]

    # Select a number of masks at random
    masks_bank = np.sort([mask_dir + masks_files[i] for i in np.random.choice(n_cl, size=n_cl, replace=False)])

    # Find the corresponding cluster IDs in Bocquet that match the masks we chose
    bocquet_ids = [re.search('SPT-CLJ(.+?)_', mask_name).group(0)[:-1] for mask_name in masks_bank]
    bocquet_idx = np.any([bocquet['SPT_ID'] == b_id for b_id in bocquet_ids], axis=0)
    selected_clusters = bocquet['SPT_ID', 'RA', 'DEC', 'M500', 'REDSHIFT'][bocquet_idx]

    # We'll need the r500 radius for each cluster too.
    selected_clusters['r500'] = (3 * selected_clusters['M500'] * u.Msun /
                                 (4 * np.pi * 500 *
                                  cosmo.critical_density(selected_clusters['REDSHIFT']).to(u.Msun / u.Mpc**3)))**(1/3)

    # Create cluster names
    name_bank = ['SPT_Mock_{:03d}'.format(i) for i in range(n_cl)]

    # Combine our data into a catalog
    SPT_data = Table([name_bank, selected_clusters['RA'], selected_clusters['DEC'], selected_clusters['M500'],
                      selected_clusters['r500'], selected_clusters['REDSHIFT'], masks_bank, selected_clusters['SPT_ID']],
                     names=['SPT_ID', 'SZ_RA', 'SZ_DEC', 'M500', 'r500', 'REDSHIFT', 'MASK_NAME', 'orig_SPT_ID'])

    # Check that we have the correct mask and cluster data matched up. If so, we can drop the original SPT_ID column
    assert np.all([spt_id in mask_name for spt_id, mask_name in zip(SPT_data['orig_SPT_ID'], SPT_data['MASK_NAME'])])
    del SPT_data['orig_SPT_ID']

    # Set up grid of radial positions to place AGN on (normalized by r500)
    r_dist_r500 = np.linspace(0, max_radius, num=200)
    # </editor-fold>
    #%%
    cluster_sample = SPT_data

    hist_heights = {}
    hist_scaled_areas = {}
    hist_errors = {}
    hist_models = {}
    gpfs = {}

    AGN_cats = []
    for cluster in cluster_sample:
        spt_id = cluster['SPT_ID']
        mask_name = cluster['MASK_NAME']
        z_cl = cluster['REDSHIFT']
        m500_cl = cluster['M500'] * u.Msun
        r500_cl = cluster['r500'] * u.Mpc
        SZ_center = cluster['SZ_RA', 'SZ_DEC']

        # Read in the mask's WCS for the pixel scale and making SkyCoords
        w = WCS(mask_name)
        mask_pixel_scale = w.pixel_scale_matrix[1, 1] * u.deg

        # Also get the mask's image size (- 1 to account for the shift between index and length)
        mask_size_x = w.pixel_shape[0] - 1
        mask_size_y = w.pixel_shape[1] - 1
        mask_radius_pix = (max_radius * r500_cl * cosmo.arcsec_per_kpc_proper(z_cl).to(u.deg/u.Mpc) / mask_pixel_scale).value

        # Find the SZ Center for the cluster we are mimicking
        SZ_center_skycoord = SkyCoord(SZ_center['SZ_RA'], SZ_center['SZ_DEC'], unit='deg')

        # Calculate the model values for the AGN candidates in the cluster
        model_cluster_agn = model_rate(z_cl, m500_cl, r500_cl, r_dist_r500, params_true)

        # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
        # the cluster.
        max_rate = np.max(model_cluster_agn)  # r500^-2 units
        max_rate_inv_pix2 = ((max_rate / r500_cl**2) * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin)**2
                             * mask_pixel_scale.to(u.arcmin)**2)

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
        radii_r500 = radii_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc/u.arcmin) / r500_cl

        # Filter the candidates through the model to establish the radial trend in the data.
        rate_at_rad = model_rate(z_cl, m500_cl, r500_cl, radii_r500, params_true)

        # Our rejection rate is the model rate at the radius scaled by the maximum rate
        prob_reject = rate_at_rad / max_rate

        # Draw a random number for each candidate
        alpha = np.random.uniform(0, 1, len(rate_at_rad))

        # Perform the rejection sampling
        cluster_agn_final = cluster_agn_skycoord[np.where(prob_reject >= alpha)]
        cluster_agn_final_pix = np.array(cluster_agn_final.to_pixel(w, origin=0, mode='wcs'))

        # Generate background sources
        background_rate = C_true / u.arcmin**2 * mask_pixel_scale.to(u.arcmin)**2
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
        AGN_list['r500'] = r500_cl

        # Create a flag indicating if the object is a cluster member
        AGN_list['Cluster_AGN'] = np.concatenate((np.full_like(cluster_agn_final_pix[0], True),
                                                  np.full_like(background_agn_pix[0], False)))

        # Convert the pixel coordinates to RA/Dec coordinates
        agn_coords_skycoord = SkyCoord.from_pixel(AGN_list['x_pixel'], AGN_list['y_pixel'], wcs=w, origin=0, mode='wcs')
        AGN_list['RA'] = agn_coords_skycoord.ra
        AGN_list['DEC'] = agn_coords_skycoord.dec

        # Calculate the radii of the final AGN scaled by the cluster's r500 radius
        r_final_arcmin = SZ_center_skycoord.separation(agn_coords_skycoord).to(u.arcmin)
        r_final_r500 = r_final_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl
        AGN_list['radial_arcmin'] = r_final_arcmin
        AGN_list['radial_r500'] = r_final_r500

        # Select only objects within the max_radius
        AGN_list = AGN_list[AGN_list['radial_r500'] <= max_radius]

        # Read in the original (full) mask
        full_mask_image, full_mask_header = fits.getdata(mask_name, header=True)

        # Select the image to mask the data on
        mask_image = full_mask_image
        AGN_list['MASK_NAME'] = mask_name

        # <editor-fold desc="Flat Mask">
        # Generate the flat mask version
        # no_mask_image = np.ones_like(full_mask_image)
        #
        # # Write the new mask out to disk for use in fitting later
        # no_mask_name = mask_dir + 'no_masks/' + re.search('SPT(.+?).fits', mask_name).group(0)
        # fits.PrimaryHDU(no_mask_image, header=full_mask_header).writeto(no_mask_name, overwrite=True)
        # AGN_list['MASK_NAME'] = no_mask_name
        #
        # # Select the image to mask the data on
        # mask_image = no_mask_image
        # </editor-fold>

        # <editor-fold desc="Quarter Mask">
        # Generate the quarter mask version
        # quarter_mask_name = mask_dir + 'quarter_masks/' + re.search('SPT(.+?).fits', mask_name).group(0)
        # quarter_mask_image = np.ones_like(full_mask_image)
        #
        # # Write the new mask out to disk for use in fitting later
        # quarter_mask_area = quarter_mask_image[SZ_center_pix[1].round().astype(int):, SZ_center_pix[0].round().astype(int):]
        # quarter_mask_image[SZ_center_pix[1].round().astype(int):,
        #                    SZ_center_pix[0].round().astype(int):] = np.zeros_like(quarter_mask_area)
        # fits.PrimaryHDU(data=quarter_mask_image, header=full_mask_header).writeto(quarter_mask_name, overwrite=True)
        # AGN_list['MASK_NAME'] = quarter_mask_name
        #
        # # Select the image to mask the data on
        # mask_image = quarter_mask_image
        # </editor-fold>

        # Remove all objects that are outside of the image bounds
        AGN_list = AGN_list[np.all([0 <= AGN_list['x_pixel'],
                                    AGN_list['x_pixel'] <= mask_size_x,
                                    0 <= AGN_list['y_pixel'],
                                    AGN_list['y_pixel'] <= mask_size_y], axis=0)]

        # Pass the cluster catalog through the quarter mask to insure all objects are on image.
        AGN_list = AGN_list[np.where(mask_image[AGN_list['y_pixel'].round().astype(int),
                                                AGN_list['x_pixel'].round().astype(int)] == 1)]

        AGN_cats.append(AGN_list)
    #%%
        # <editor-fold desc="Diagnostics">
        # ------- The rest of this loop is dedicated to diagnostics of the sample --------
        # Create a histogram of the objects in the cluster using evenly spaced bins on radius
        radial_bins = np.linspace(0, max_radius, num=num_bins)
        hist, bin_edges = np.histogram(AGN_list['radial_r500'], bins=radial_bins)

        # Compute area in terms of r500^2
        area_edges = np.pi * bin_edges ** 2
        area = np.diff(area_edges)

        # Calculate the good pixel fraction for each annulus area
        SZ_center = AGN_list['SZ_RA', 'SZ_DEC'][0]
        gpf = good_pixel_fraction(bin_edges, z_cl, r500_cl, AGN_list['MASK_NAME'][0], SZ_center)
        # gpf = 1.0

        # Scale our area by the good pixel fraction
        scaled_area = area * gpf

        # Use small-N Poisson error of counts in each bin normalized by the area of the bin
        count_err = small_poisson(hist)
        err = [count_err_ul / scaled_area for count_err_ul in count_err]
        np.nan_to_num(err, copy=False)

        # Calculate the model for this cluster
        rall = np.linspace(0, np.max(bin_edges), num=200)
        background_rate_r500 = C_true / u.arcmin**2 * cosmo.arcsec_per_kpc_proper(z_cl).to(u.arcmin / u.Mpc)**2 * r500_cl**2
        model_cl = model_rate(z_cl, m500_cl, r500_cl, rall, params_true) + background_rate_r500
        gpf_intep_func = interp1d(bin_edges, np.insert(gpf, 0, 1.), kind='cubic')
        gpf_rall = gpf_intep_func(rall)

        # Drop model values for bins that do not have any area
        # r_zero = np.min(bin_edges[np.where(np.array(gpf) <= 0.4)])
        # model_cl[np.where(gpf_rall <= 0.4)] = np.nan
        # model_cl = model_cl.value / gpf_rall

        # Store the binned data into the dictionaries
        # gpfs.update({spt_id: gpf})
        hist_heights[spt_id] = hist
        hist_scaled_areas[spt_id] = scaled_area
        hist_errors[spt_id] = err
        hist_models[spt_id] = model_cl
        # </editor-fold>
    #%%
    # Stack the individual cluster catalogs into a single master catalog
    outAGN = vstack(AGN_cats)

    # Reorder the columns in the cluster for ascetic reasons.
    outAGN = outAGN['SPT_ID', 'SZ_RA', 'SZ_DEC', 'x_pixel', 'y_pixel', 'RA', 'DEC', 'REDSHIFT', 'M500', 'r500',
                    'radial_arcmin', 'radial_r500', 'MASK_NAME', 'Cluster_AGN']

    print('\n------\nparameters: {param}\nTotal number of clusters: {cl} \t Total number of objects: {agn}'
          .format(param=params_true, cl=len(outAGN.group_by('SPT_ID').groups.keys), agn=len(outAGN)))
    outAGN.write('Data/MCMC/Mock_Catalog/Catalogs/Signal-Noise_tests/theta_varied/'
                 'mock_AGN_catalog_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_C{C:.3f}'
                 '_maxr{maxr:.2f}_seed{seed}_full_mask.cat'
                 .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true,
                         maxr=max_radius, nbins=num_bins, seed=rand_seed),
                 format='ascii', overwrite=True)
    #%%
    # <editor-fold desc="Diagnostic Plots">
    # -------- Diagnostic Plots --------
    # Average the cluster histograms
    stacked_heights = np.nansum(np.array(list(hist_heights.values())), axis=0)
    stacked_areas = np.nansum(np.array(list(hist_scaled_areas.values())), axis=0)
    stacked_hist = stacked_heights / stacked_areas

    # Find the errors using the fractional Poisson error in the bin.
    frac_err = np.sqrt(stacked_heights) / stacked_heights
    stacked_err = frac_err * stacked_hist

    # Average the cluster models
    stacked_model = np.nanmean(list(hist_models.values()), axis=0)

    # Find the scatter on the models
    stacked_model_err = np.nanstd(list(hist_models.values()), axis=0)

    # A grid of radii for the data to be plotted on
    bin_edges = np.linspace(0, max_radius, num=num_bins)
    bins = (bin_edges[1:len(bin_edges)] - bin_edges[0:len(bin_edges)-1]) / 2. + bin_edges[0:len(bin_edges)-1]

    # A grid of radii for the model to be plotted on
    rall = np.linspace(0, np.max(bin_edges), num=200)

    # A quick chi2 fit of the mean model to find the redshift and mass of the "cluster" it corresponds to
    # f = lambda r, z, m: model_rate(z, m*u.Msun, (3 * m*u.Msun / (4 * np.pi * 500 *
    #                                                              cosmo.critical_density(z).to(u.Msun / u.Mpc**3)))**(1/3),
    #                                r, max_radius, params_true)
    # model_z_m, model_cov = op.curve_fit(f, rall[:75], stacked_model[:75], sigma=stacked_model_err[:75],
    #                                     bounds=([0.5, 0.2e15], [1.7, 1.8e15]))
    # model_z_m_err = np.sqrt(np.diag(model_cov))
    # print('Mean model: z = {z:.2f} +/- {z_err:.2e}\tm500 = {m:.2e} +/- {m_err:.3e} Msun'
    #       .format(z=model_z_m[0], z_err=model_z_m_err[0], m=model_z_m[1], m_err=model_z_m_err[1]))

    # Overplot the normalized binned data with the model rate
    fig, ax = plt.subplots()
    ax.errorbar(bins, stacked_hist, yerr=stacked_err, fmt='o', color='C1',
                label='Mock AGN Candidate Surface Density')
    ax.plot(rall, stacked_model, color='C0', label='Model Rate')
    ax.fill_between(rall, y1=stacked_model+stacked_model_err, y2=stacked_model-stacked_model_err, color='C0', alpha=0.2)
    ax.set(title=r'Comparison of Sampled Points to Model $\theta$ = {theta}'.format(theta=theta_true),
           xlabel=r'$r/r_{{500}}$', ylabel=r'Rate per cluster [$r_{{500}}^{-2}$]')
    ax.legend()
    # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/'
    #             'mock_AGN_binned_check_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_C{C:.3f}_maxr{maxr:.2f}_seed{seed}'
    #             '_flat_mask_refresh.pdf'
    #             .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true,
    #                     maxr=max_radius, seed=rand_seed), format='pdf')
    ax.set(xlim=[0, 3.0])
    fig.savefig('Data/MCMC/Mock_Catalog/Plots/Signal-Noise_tests/theta_varied/'
                'mock_AGN_binned_check_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_C{C:.3f}_maxr{maxr:.2f}_seed{seed}'
                '_full_mask_zoom.pdf'
                .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true,
                        maxr=max_radius, seed=rand_seed), format='pdf')

    # fig, axarr = plt.subplots(nrows=15, ncols=13, figsize=(50, 50))
    # for cluster_id, ax in zip(hist_heights.keys(), axarr.flatten()):
    #     surface_den = hist_heights[cluster_id] / hist_scaled_areas[cluster_id]
    #     surface_den_err = hist_errors[cluster_id]
    #     cluster_model = hist_models[cluster_id]
    #
    #     ax.errorbar(bins, surface_den, yerr=surface_den_err[::-1], fmt='o', color='C1',
    #                 label='Mock AGN Candidate Surface Density')
    #     ax.plot(rall, cluster_model, color='C0', label='Model Rate')
    #     ax.set(title='Comparison of Sampled Points to Model\n{}'.format(cluster_id),
    #            xlabel=r'$r/r_{{500}}$', ylabel=r'Rate per cluster [$r_{{500}}^{-2}$]')
    # plt.tight_layout()
    # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/'
    #             'mock_AGN_binned_check_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_C{C:.3f}_maxr{maxr:.2f}_nbins{nbins}'
    #             '_seed{seed}_all_clusters_data_to_5r500_flat_mask_applied.pdf'
    #             .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true,
    #                     maxr=max_radius, nbins=num_bins, seed=rand_seed),
    #             format='pdf')
    plt.show()
    # </editor-fold>

    print('Run time: {:.2f}s'.format(time() - start_time))
