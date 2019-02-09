"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""

from __future__ import print_function, division

import re
from itertools import product
from time import time

from os import listdir

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


def poisson_point_process(model, dx, dy=None):
    """
    Uses a spatial Poisson point process to generate AGN candidate coordinates.

    :param model: The model rate used in the Poisson distribution to determine the number of points being placed.
    :param dx: Upper bound on x-axis (lower bound is set to 0).
    :param dy: Upper bound on y-axis (lower bound is set to 0).
    :return coord: 2d numpy array of (x,y) coordinates of AGN candidates.
    """

    if dy is None:
        dy = dx

    # Draw from Poisson distribution to determine how many points we will place.
    p = stats.poisson(model * dx * dy).rvs()

    # Drop `p` points with uniform x and y coordinates
    x = stats.uniform.rvs(0, dx, size=p)
    y = stats.uniform.rvs(0, dy, size=p)

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
    pix_area = []
    for i in np.arange(len(r_pix) - 1):
        pix_ring = large_image[np.where((r_pix[i] <= image_dists) & (image_dists < r_pix[i + 1]))]

        # Calculate the fraction
        good_pix_frac.append(np.sum(pix_ring) / len(pix_ring))
        pix_area.append(np.sum(pix_ring) * pix_scale * pix_scale)

    return good_pix_frac, pix_area

start_time = time()
# <editor-fold desc="Parameter Set up">

# Number of clusters to generate
n_cl = 1

# Dx is set to 5 to mimic an IRAC image's width in arcmin.
Dx = 5.  # In arcmin

# Set parameter values
theta_true = 12000     # Amplitude.
eta_true = 1.2       # Redshift slope
zeta_true = -1.0     # Mass slope
beta_true = 0.5      # Radial slope
C_true = 0.371       # Background AGN surface density

params_true = (theta_true, eta_true, zeta_true, beta_true)

# Set the maximum radius we will generate objects to
max_radius = 5.0

# Number of bins to use to plot our sampled data points
num_bins = 30

# For a preliminary mask, we will use a perfect 5'x5' image with a dummy WCS
# Set the pixel scale and size of the image
pixel_scale = 0.000239631 * u.deg  # Standard pixel scale for SPT IRAC images (0.8626716 arcsec)
# </editor-fold>

# <editor-fold desc="Data Generation">
# Make the mask data
bocquet = Table.read('Data/2500d_cluster_sample_Bocquet18.fits')  # For SZ centers

# For our masks, we will co-op the masks for the real clusters.
mask_dir = 'Data/Masks/'
masks_files = [f for f in listdir(mask_dir) if not f.startswith('.')]

# Make sure all the masks have matches in the catalog
masks_files = [f for f in masks_files if re.search('SPT-CLJ(.+?)_', f).group(0)[:-1] in bocquet['SPT_ID']]

# Select a number of masks at random
masks_bank = [mask_dir + masks_files[i] for i in np.random.randint(n_cl, size=n_cl)]

# Set up grid of radial positions (normalized by r500)
r_dist_r500 = np.logspace(-2, np.log10(max_radius), 200)

# Draw mass and redshift distribution from a uniform distribution as well.
mass_dist = np.random.uniform(0.2e15, 1.8e15, n_cl)
z_dist = np.random.uniform(0.5, 1.7, n_cl)

# Create cluster names
name_bank = ['SPT_Mock_{:03d}'.format(i) for i in range(n_cl)]
SPT_data = Table([name_bank, z_dist, mass_dist, masks_bank], names=['SPT_ID', 'REDSHIFT', 'M500', 'MASK_NAME'])

# We'll need the r500 radius for each cluster too.
SPT_data['r500'] = (3 * SPT_data['M500'] * u.Msun /
                    (4 * np.pi * 500 * cosmo.critical_density(SPT_data['REDSHIFT']).to(u.Msun / u.Mpc**3)))**(1/3)
# </editor-fold>

cluster_sample = SPT_data

hist_heights = {}
hist_scaled_areas = {}
hist_errors = {}
hist_models = {}

AGN_cats = []
for cluster in cluster_sample:
    spt_id = cluster['SPT_ID']
    mask_name = cluster['MASK_NAME']
    z_cl = cluster['REDSHIFT']
    m500_cl = cluster['M500'] * u.Msun
    r500_cl = cluster['r500'] * u.Mpc

    # Read in the mask's WCS for the pixel scale and making SkyCoords
    w = WCS(mask_name)
    mask_pixel_scale = fits.getval(mask_name, 'PXSCAL2') * u.arcsec

    # Also get the mask's image size (- 1 to account for the shift between index and length)
    mask_size_x = fits.getval(mask_name, 'NAXIS1') - 1
    mask_size_y = fits.getval(mask_name, 'NAXIS2') - 1

    # Find the SZ Center for the cluster we are mimicking
    bocquet_id = re.search('SPT-CLJ(.+?)_', mask_name).group(0)[:-1]
    SZ_center = bocquet['RA', 'DEC'][np.where(bocquet['SPT_ID'] == bocquet_id)]
    SZ_center_skycoord = SkyCoord(SZ_center['RA'], SZ_center['DEC'], unit='deg')

    # Calculate the model values for the AGN candidates in the cluster
    model_cluster_agn = model_rate(z_cl, m500_cl, r500_cl, r_dist_r500, params_true)

    # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
    # the cluster.
    max_rate = np.max(model_cluster_agn)  # r500^-2 units
    max_rate_inv_pix2 = (max_rate * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin)**2
                         / r500_cl**2 * mask_pixel_scale.to(u.arcmin)**2)

    # Simulate the AGN using the spatial Poisson point process.
    cluster_agn_coords_pix = poisson_point_process(max_rate_inv_pix2, dx=mask_size_x, dy=mask_size_y)

    # Find the radius of each point placed scaled by the cluster's r500 radius
    # cluster_agn_coords_pix = cluster_agn_coords * u.arcmin / mask_pixel_scale.to(u.arcmin)
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
    background_agn_pix = poisson_point_process(background_rate, dx=mask_size_x, dy=mask_size_y)

    # Concatenate the cluster sources with the background sources
    line_of_sight_agn_pix = np.hstack((cluster_agn_final_pix, background_agn_pix))

    # Set up the table of objects
    AGN_list = Table([line_of_sight_agn_pix[0], line_of_sight_agn_pix[1]], names=['x_pixel', 'y_pixel'])
    AGN_list['SPT_ID'] = spt_id
    AGN_list['SZ_RA'] = SZ_center['RA']
    AGN_list['SZ_DEC'] = SZ_center['DEC']
    AGN_list['M500'] = m500_cl
    AGN_list['REDSHIFT'] = z_cl
    AGN_list['r500'] = r500_cl
    AGN_list['MASK_NAME'] = mask_name

    # Create a flag indicating if the object is a cluster member
    AGN_list['Cluster_AGN'] = np.concatenate((np.full_like(cluster_agn_final_pix[0], True),
                                              np.full_like(background_agn_pix[0], False)))

    # Read in the mask and check if the object is on a good pixel of the mask
    mask_image = fits.getdata(mask_name)
    AGN_list = AGN_list[np.where(mask_image[AGN_list['y_pixel'].round().astype(int),
                                            AGN_list['x_pixel'].round().astype(int)] == 1)]

    # Convert the pixel coordinates to RA/Dec coordinates
    agn_coords_skycoord = SkyCoord.from_pixel(AGN_list['x_pixel'], AGN_list['y_pixel'], wcs=w, origin=0, mode='wcs')
    AGN_list['RA'] = agn_coords_skycoord.ra
    AGN_list['DEC'] = agn_coords_skycoord.dec

    # Calculate the radii of the final AGN scaled by the cluster's r500 radius
    r_final_arcmin = SZ_center_skycoord.separation(agn_coords_skycoord).to(u.arcmin)
    r_final_r500 = r_final_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl
    AGN_list['radial_arcmin'] = r_final_arcmin
    AGN_list['radial_r500'] = r_final_r500

    AGN_cats.append(AGN_list)

    # <editor-fold desc="Diagnostics">
    # ------- The rest of this loop is dedicated to diagnostics of the sample --------
    # Create a histogram of the objects in the cluster using evenly spaced bins on radius
    hist, bin_edges = np.histogram(AGN_list['radial_r500'], bins='auto')
    num_bins = len(bin_edges-1)

    # Compute area in terms of r500^2
    area_edges = np.pi * bin_edges ** 2
    area = np.diff(area_edges)

    # Calculate the good pixel fraction for each annulus area (We can do this here for now as all mock clusters use
    # the same mask. If we source from real masks we'll need to move this up into the loop.)
    # For our center set a dummy center at (0,0)
    SZ_center = AGN_list['SZ_RA', 'SZ_DEC'][0]
    gpf, pixel_area = good_pixel_fraction(bin_edges, z_cl, r500_cl, mask_name, SZ_center)

    # Scale our area by the good pixel fraction
    scaled_area = area * gpf

    # Use small-N Poisson error of counts in each bin normalized by the area of the bin
    count_err = small_poisson(hist)
    err = [count_err_ul / scaled_area for count_err_ul in count_err]
    np.nan_to_num(err, copy=False)

    # Calculate the model for this cluster
    rall = np.linspace(0, np.max(bin_edges), num=200)
    background_rate_r500 = C_true / u.arcmin ** 2 * cosmo.arcsec_per_kpc_proper(z_cl).to(u.arcmin / u.Mpc) ** 2 * r500_cl ** 2
    model_cl = model_rate(z_cl, m500_cl, r500_cl, rall, params_true) + background_rate_r500
    # gpf_rall, _ = good_pixel_fraction(rall, z_cl, r500_cl, mask_name, SZ_center)

    # Drop model values for bins that do not have any area
    # r_zero = np.min(bin_edges[np.where(scaled_area == 0)])
    # model_cl[np.where(rall >= r_zero)] = np.nan
    # model_cl = model_cl.value / np.insert(gpf_rall, 0, 1.)

    # Store the binned data into the dictionaries
    hist_heights.update({spt_id: hist})
    hist_scaled_areas.update({spt_id: scaled_area})
    hist_errors.update({spt_id: err})
    hist_models.update({spt_id: model_cl})
    # </editor-fold>

# Stack the individual cluster catalogs into a single master catalog
outAGN = vstack(AGN_cats)

# Reorder the columns in the cluster for ascetic reasons.
outAGN = outAGN['SPT_ID', 'SZ_RA', 'SZ_DEC', 'x_pixel', 'y_pixel', 'RA', 'DEC', 'REDSHIFT', 'M500', 'r500',
                'radial_arcmin', 'radial_r500', 'MASK_NAME', 'Cluster_AGN']

print('\n------\nparameters: {param}\nTotal number of clusters: {cl} \t Total number of objects: {agn}'
      .format(param=params_true, cl=len(outAGN.group_by('SPT_ID').groups.keys), agn=len(outAGN)))
outAGN.write('Data/MCMC/Mock_Catalog/Catalogs/pre-final_tests/'
             'mock_AGN_catalog_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_C{C:.3f}'
             '_maxr{maxr:.2f}_seed{seed}_gpf_fixed_single_cluster.cat'
             .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true,
                     maxr=max_radius, nbins=num_bins, seed=rand_seed),
             format='ascii', overwrite=True)

# <editor-fold desc="Diagnostic Plots">
# -------- Diagnostic Plots --------
# AGN Candidates
sz_center_pix = w.wcs_world2pix(AGN_list['SZ_RA'], AGN_list['SZ_DEC'], 0)
cluster_agn = AGN_list[np.where(AGN_list['Cluster_AGN'].astype(bool))]
backgound_agn = AGN_list[np.where(~AGN_list['Cluster_AGN'].astype(bool))]
fig, ax = plt.subplots(subplot_kw={'projection': w})
ax.imshow(mask_image, origin='lower', cmap='gray_r')
ax.plot(sz_center_pix[0], sz_center_pix[1], 'w+')
ax.scatter(cluster_agn['x_pixel'], cluster_agn['y_pixel'], edgecolor='cyan', facecolor='none', alpha=1.,
           label='Cluster AGN')
ax.scatter(backgound_agn['x_pixel'], backgound_agn['y_pixel'], edgecolor='red', facecolor='none', alpha=1.,
           label='Background AGN')
ax.coords[0].set_major_formatter('hh:mm:ss.s'); ax.coords[1].set_major_formatter('dd:mm:ss')
ax.set(title='Sample Cluster Line-of-sight Generation',
       xlabel='Right Ascension', ylabel='Declination')
ax.legend(handletextpad=0.001)
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/example_cluster'
            '_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_C{C:.3f}'
            '_maxr{maxr:.2f}_seed{seed}_mask{spt_id}_gpf_fixed_single_cluster.pdf'
            .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true,
                    maxr=max_radius, nbins=num_bins, seed=rand_seed, spt_id=spt_id),
            format='pdf')
plt.show()

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
# bin_edges = np.logspace(-2, np.log10(max_radius), num=num_bins+1)
bins = (bin_edges[1:len(bin_edges)] - bin_edges[0:len(bin_edges)-1]) / 2. + bin_edges[0:len(bin_edges)-1]

# A grid of radii for the model to be plotted on
# rall = np.logspace(-2, np.log10(max_radius+2), 200)

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
ax.set(title='Comparison of Sampled Points to Model (Stacked Sample)',
       xlabel=r'$r/r_{{500}}$', ylabel=r'Rate per cluster [$r_{{500}}^{-2}$]')
ax.legend()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/'
            'mock_AGN_binned_check_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_C{C:.3f}_maxr{maxr:.2f}_nbins{nbins}'
            '_seed{seed}_model_nan_0area_gpf_fixed_single_cluster.pdf'
            .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true,
                    maxr=max_radius, nbins=num_bins, seed=rand_seed),
            format='pdf')
plt.show()
print('Run time: {:.2f}s'.format(time() - start_time))
# </editor-fold>
