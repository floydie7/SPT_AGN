"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""

from __future__ import print_function, division

from itertools import product

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from scipy import stats
from scipy.spatial.distance import cdist
from os import listdir
from os.path import isfile, join

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Set our cosmology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

# Set our random seed
np.random.seed(123)


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
    # print('Number of points placed: ', p)

    # Drop `p` points with uniform x and y coordinates
    x = stats.uniform.rvs(0, dx, size=p)
    y = stats.uniform.rvs(0, dy, size=p)

    # Combine the x and y coordinates.
    coord = np.vstack((x, y))

    return coord


def model_rate(z, m, r500, r_r500, maxr, params):
    """
    Our generating model.

    :param z: Redshift of the cluster
    :param m: M_500 mass of the cluster
    :param r500: r500 radius of the cluster
    :param maxr: maximum radius in units of r500 to consider
    :param r_r500: A vector of radii of objects within the cluster normalized by the cluster's r500
    :param params: Tuple of (theta, eta, zeta, beta, background)
    :return model: A surface density profile of objects as a function of radius
    """

    # Unpack our parameters
    theta, eta, zeta, beta = params

    # theta = theta / u.Mpc**2 * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc/u.arcmin)**2

    # Convert our background surface density from angular units into units of r500^-2
    background = 0.371 / u.arcmin ** 2 * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc) ** 2 * r500 ** 2
    # print(background)

    # r_r500 = r * u.arcmin * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc/u.arcmin) / r500

    # The cluster's core radius in units of r500
    rc_r500 = 0.1 * u.Mpc / r500

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z) ** eta * (m / (1e15 * u.Msun)) ** zeta
    # print('a = {}'.format(a))

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc_r500) ** 2) ** (-1.5 * beta + 0.5) + background

    # We impose a cut off of all objects with a radius greater than 1.1r500
    model[r_r500 > maxr] = 0.

    return model.value


def good_pixel_fraction(r, z, r500, image_name, center):
    # Read in the mask file and the mask file's WCS
    image, header = fits.getdata(image_name, header=True)
    image_wcs = WCS(header)

    # From the WCS get the pixel scale and the size of the image
    pix_scale = (image_wcs.pixel_scale_matrix[1, 1] * u.deg).to(u.arcsec)
    xlen = header['NAXIS1']
    ylen = header['NAXIS2']

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center)

    # Convert our radius to pixels
    r_pix = r * r500 * cosmo.arcsec_per_kpc_proper(z).to(u.arcsec / u.Mpc) / pix_scale

    # find the distances from center pixel to all other pixels
    image_coords = np.array(list(product(range(xlen), range(ylen))))

    center_coord = np.asanyarray(center_pix)

    image_dists = cdist(image_coords, center_coord).reshape(image.shape)

    # select all pixels that are within the annulus
    good_pix_frac = []
    for i in np.arange(len(r_pix) - 1):
        pix_ring = image[np.where((image_dists > r[i]) & (image_dists <= r[i + 1]))]

        # Calculate the fraction
        good_pix_frac.append(np.sum(pix_ring) / len(pix_ring))

    return good_pix_frac


# Number of clusters to generate
n_cl = 195

# Dx is set to 5 to mimic an IRAC image's width in arcmin.
Dx = 5.  # In arcmin

# Set parameter values
theta_true = 0.3     # Amplitude.
eta_true = 1.2       # Redshift slope
zeta_true = -1.0     # Mass slope
beta_true = 0.5      # Radial slope
C_true = 0.371       # Background AGN surface density

params_true = (theta_true, eta_true, zeta_true, beta_true)

max_radius = 1.5

# Set up grid of radial positions (normalized by r500)
r_dist_r500 = np.linspace(0, 2, 100)  # from 0 - 2r500

# Draw mass and redshift distribution from a uniform distribution as well.
mass_dist = np.random.uniform(0.2e15, 1.8e15, n_cl)
z_dist = np.random.uniform(0.5, 1.7, n_cl)

# TODO: Try this later
# # For our masks, we will co-op the masks for the real clusters.
# masks_files = [f for f in listdir('Data/Masks') if isfile(join('Data/Masks', f))]
#
# # Select a number of masks at random
# masks_bank = [masks_files[i] for i in np.random.randint(n_cl, size=n_cl)]

# For a preliminary mask, we will use a perfect 5'x5' image with a dummy WCS
# Set the pixel scale and size of the image
pixel_scale = 0.000239631 * u.deg  # Standard pixel scale for SPT IRAC images (0.8626716 arcsec)
mask_size = Dx * u.arcmin / pixel_scale.to(u.arcmin)

# Make the mask data
mask_data = np.ones(shape=(mask_size, mask_size))

# Create an HDU
mask_hdu = fits.PrimaryHDU(data=mask_data)

# Fill in the header with WCS information
header = mask_hdu.header
header['CRPIX1'] = round(mask_size.value/2.)
header['CRPIX2'] = round(mask_size.value/2.)
header['CRVAL1'] = 0.
header['CRVAL2'] = 0.
header['CDELT1'] = -pixel_scale.value
header['CDELT2'] = pixel_scale.value
header['CTYPE1'] = 'RA---TAN'
header['CTYPE2'] = 'DEC---TAN'

# Write the mask to disk
mask_file = 'Data/MCMC/Mock_Catalog/mock_flat_mask.fits'
mask_hdu.writeto(mask_file, overwrite=True)

# Create cluster names
name_bank = ['SPT_Mock_{:03d}'.format(i) for i in range(n_cl)]
SPT_data = Table([name_bank, z_dist, mass_dist], names=['SPT_ID', 'REDSHIFT', 'M500'])
SPT_data['MASK_NAME'] = mask_file

# We'll need the r500 radius for each cluster too.
SPT_data['r500'] = (3 * SPT_data['M500'] /
                    (4 * np.pi * 500 * cosmo.critical_density(SPT_data['REDSHIFT']).to(u.Msun / u.Mpc**3)))**(1/3)
SPT_data_z = SPT_data[np.where(SPT_data['REDSHIFT'] >= 0.75)]
SPT_data_m = SPT_data[np.where(SPT_data['M500'] <= 5e14)]

cluster_sample = SPT_data

AGN_cats = []
for cluster in cluster_sample:
    spt_id = cluster['SPT_ID']
    mask_name = cluster['MASK_NAME']
    z_cl = cluster['REDSHIFT']
    m500_cl = cluster['M500'] * u.Msun
    r500_cl = cluster['r500'] * u.Mpc
    print("---\nCluster Data: z = {z:.2f}, M500 = {m:.2e}, r500 = {r:.2f}".format(z=z_cl, m=m500_cl, r=r500_cl))

    # Calculate the model values for the AGN candidates in the cluster
    rad_model = model_rate(z_cl, m500_cl, r500_cl, r_dist_r500, max_radius, params_true)

    # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
    # the cluster.
    max_rate = np.max(rad_model)
    max_rate_arcmin2 = max_rate * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin)**2 / r500_cl**2
    # print('Max rate: {}'.format(max_rate_arcmin2))
    # max_rate_list.append(max_rate)

    # Simulate the AGN using the spatial Poisson point process.
    agn_coords = poisson_point_process(max_rate_arcmin2, Dx)

    # Find the radius of each point placed scaled by the cluster's r500 radius
    radii_arcmin = np.sqrt((agn_coords[0] - Dx / 2.) ** 2 + (agn_coords[1] - Dx / 2.) ** 2) * u.arcmin
    radii_r500 = radii_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc/u.arcmin) / r500_cl

    # Filter the candidates through the model to establish the radial trend in the data.
    rate_at_rad = model_rate(z_cl, m500_cl, r500_cl, radii_r500, max_radius, params_true)

    # Our rejection rate is the model rate at the radius scaled by the maximum rate
    prob_reject = rate_at_rad / max_rate

    # Draw a random number for each candidate
    alpha = np.random.uniform(0, 1, len(rate_at_rad))

    x_final = agn_coords[0][np.where(prob_reject >= alpha)]
    y_final = agn_coords[1][np.where(prob_reject >= alpha)]
    print('Number of points in final selection: {}'.format(len(x_final)))

    # Calculate the radii of the final AGN scaled by the cluster's r500 radius
    r_final_arcmin = np.sqrt((x_final - Dx / 2.) ** 2 + (y_final - Dx / 2.) ** 2) * u.arcmin
    r_final_r500 = r_final_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl

    if r_final_r500.size != 0:
        # Create a table of our output objects
        AGN_list = Table([r_final_arcmin, r_final_r500], names=['radial_arcmin', 'radial_r500'])
        AGN_list['SPT_ID'] = spt_id
        AGN_list['M500'] = m500_cl
        AGN_list['REDSHIFT'] = z_cl
        AGN_list['r500'] = r500_cl
        AGN_list['MASK_NAME'] = mask_name

        AGN_cat = AGN_list['SPT_ID', 'REDSHIFT', 'M500', 'r500', 'radial_arcmin', 'radial_r500', 'MASK_NAME']
        AGN_cats.append(AGN_cat)

outAGN = vstack(AGN_cats)

print('\n------\nparameters: {param}\nTotal number of clusters: {cl} \t Total number of objects: {agn}'
      .format(param=params_true, cl=len(outAGN.group_by('SPT_ID').groups.keys), agn=len(outAGN)))
# outAGN.write('Data/MCMC/Mock_Catalog/Catalogs/Cutoff_Radius/'
#              'mock_AGN_catalog_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_maxr{maxr:.2f}.cat'
#              .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, maxr=max_radius),
#              format='ascii', overwrite=True)

# Diagnostic Plots
# Model Rate
# fig, ax = plt.subplots()
# ax.plot(r_dist_r500, rad_model)
# ax.set(title='Model rate', xlabel=r'$r/r_{500}$', ylabel=r'$N(r)$ [$r_{500}^{-2}$]')
# plt.show()

# AGN Candidates
# fig, ax = plt.subplots()
# ax.scatter(agn_coords[0], agn_coords[1], edgecolor='b', facecolor='none', alpha=0.5)
# ax.set_aspect(1.0)
# ax.set(title=r'Spatial Poisson Point Process with $N_{{max}} = {:.2f}/r_{{500}}^2$'.format(max_rate_arcmin2),
#        xlabel=r'$x$ (arcmin)', ylabel=r'$y$ (arcmin)', xlim=[0, Dx], ylim=[0, Dx])
# plt.show()

# Selected AGN
# fig, ax = plt.subplots()
# ax.scatter(x_final, y_final, edgecolor='b', facecolor='none', alpha=0.5)
# ax.set_aspect(1.0)
# ax.set(title='Filtered SPPP', xlabel=r'$x$ (arcmin)', ylabel=r'$y$ (arcmin)', xlim=[0, Dx], ylim=[0, Dx])
# plt.show()

# Histogram of source counts per cluster
hist, bin_edges = np.histogram(outAGN['radial_r500'])
# hist, bin_edges = np.histogram(r_final_r500)
bins = (bin_edges[1:len(bin_edges)] - bin_edges[0:len(bin_edges)-1]) / 2. + bin_edges[0:len(bin_edges)-1]
# plt.hist(outAGN['radial_r500'], weights=np.full(len(outAGN['radial_r500']), 1/n_cl))
# plt.hist(r_final_r500)
# plt.show()

# Compute area in terms of r500^2
area_edges = np.pi * bin_edges**2
area = area_edges[1:len(area_edges)] - area_edges[0:len(area_edges)-1]

# Calculate the good pixel fraction for each annulus area


# Scale the histogram counts by the number of clusters and the area
rate_per_clust = hist / n_cl / area

# Use Poisson error of counts in each bin normalized by the area of the bin
err = np.sqrt(hist) / n_cl / area

# A grid of radii for the model to be plotted on
rall = np.linspace(0, 2.0, 100)

# Overplot the normalized binned data with the model rate
fig, ax = plt.subplots()
ax.errorbar(bins, rate_per_clust, yerr=err, fmt='o', color='C1', label='Filtered SPPP Points Normalized by Area')
ax.plot(rall, model_rate(np.median(outAGN['REDSHIFT']), np.median(outAGN['M500'])*u.Msun,
                         np.median(outAGN['r500'])*u.Mpc, rall, max_radius,
                         (theta_true, eta_true, zeta_true, beta_true)),
        color='C0', label='Model Rate')
# ax.plot(r_dist_r500, model_rate(z_cl, m500_cl, r500_cl, r_dist_r500, (2.01, 2.25, -1.29, 0.29)), color='C1', label='fit model')
ax.set(title='Comparison of Sampled Points to Model',
       xlabel=r'$r/r_{{500}}$', ylabel=r'Rate per cluster [$r_{{500}}^{-2}$]')
ax.legend()
# plt.show()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/Cutoff_Radius/'
            'mock_AGN_binned_check_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_maxr{maxr:.2f}_new.pdf'
            .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, maxr=max_radius), format='pdf')
