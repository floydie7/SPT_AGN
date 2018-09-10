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
from small_poisson import small_poisson

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
    :param r_r500: A vector of radii of objects within the cluster normalized by the cluster's r500
    :param maxr: maximum radius in units of r500 to consider
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

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r_r500 / rc_r500) ** 2) ** (-1.5 * beta + 0.5) + background

    # We impose a cut off of all objects with a radius greater than 1.1r500
    model[r_r500 > maxr] = 0.

    return model.value


def good_pixel_fraction(r, z, r500, image_name, center):
    # Read in the mask file and the mask file's WCS
    image, header = fits.getdata(image_name, header=True)
    image_wcs = WCS(header)

    # From the WCS get the pixel scale
    pix_scale = (image_wcs.pixel_scale_matrix[1, 1] * u.deg).to(u.arcsec)

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center['RA'], center['DEC'], 0)

    # Convert our radius to pixels
    r_pix = r * r500 * cosmo.arcsec_per_kpc_proper(z).to(u.arcsec / u.Mpc) / pix_scale

    # Because we potentially integrate to larger radii than can be fit on the image we will need to increase the size of
    # our mask. To do this, we will pad the mask with a zeros out to the radius we need.
    # Find the width needed to pad the image to include the largest radius inside the image.
    width = (int(np.max(r_pix) - image.shape[0] // 2), int(np.max(r_pix) - image.shape[1] // 2))

    # Insure that we are adding a non-negative padding width.
    if (width[0] <= 0) or (width[1] <= 0):
        width = (0, 0)

    large_image = np.pad(image, pad_width=width, mode='constant', constant_values=0)

    # find the distances from center pixel to all other pixels
    image_coords = np.array(list(product(range(large_image.shape[0]), range(large_image.shape[1]))))

    center_coord = np.asanyarray(center_pix).T + np.array(width) + 1

    image_dists = cdist(image_coords, center_coord).reshape(large_image.shape)

    # select all pixels that are within the annulus
    good_pix_frac = []
    for i in np.arange(len(r_pix) - 1):
        pix_ring = large_image[np.where((image_dists > r_pix[i]) & (image_dists <= r_pix[i + 1]))]

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

# For a preliminary mask, we will use a perfect 5'x5' image with a dummy WCS
# Set the pixel scale and size of the image
pixel_scale = 0.000239631 * u.deg  # Standard pixel scale for SPT IRAC images (0.8626716 arcsec)
mask_size = Dx * u.arcmin / pixel_scale.to(u.arcmin)

# Make the mask data
mask_data = np.ones(shape=(round(mask_size.value), round(mask_size.value)))

# Create a header with a dummy WCS
w = WCS(naxis=2)
w.wcs.crpix = [mask_size.value // 2., mask_size.value // 2.]
w.wcs.cdelt = np.array([-pixel_scale.value, pixel_scale.value])
w.wcs.crval = [0., 0.]
w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
mask_header = w.to_header()

# Create an HDU
mask_hdu = fits.PrimaryHDU(data=mask_data, header=mask_header)

# Write the mask to disk
mask_file = 'Data/MCMC/Mock_Catalog/mock_flat_mask.fits'
mask_hdu.writeto(mask_file, overwrite=True)

# Set up grid of radial positions (normalized by r500)
r_dist_r500 = np.linspace(0, 2, 100)  # from 0 - 2r500

# Draw mass and redshift distribution from a uniform distribution as well.
mass_dist = np.random.uniform(0.2e15, 1.8e15, n_cl)
z_dist = np.random.uniform(0.5, 1.7, n_cl)

# Create cluster names
name_bank = ['SPT_Mock_{:03d}'.format(i) for i in range(n_cl)]
SPT_data = Table([name_bank, z_dist, mass_dist], names=['SPT_ID', 'REDSHIFT', 'M500'])
SPT_data['MASK_NAME'] = mask_file

# We'll need the r500 radius for each cluster too.
SPT_data['r500'] = (3 * SPT_data['M500'] /
                    (4 * np.pi * 500 * cosmo.critical_density(SPT_data['REDSHIFT']).to(u.Msun / u.Mpc**3)))**(1/3)
# SPT_data_z = SPT_data[np.where(SPT_data['REDSHIFT'] >= 0.75)]
# SPT_data_m = SPT_data[np.where(SPT_data['M500'] <= 5e14)]

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
    # print("---\nCluster Data: z = {z:.2f}, M500 = {m:.2e}, r500 = {r:.2f}".format(z=z_cl, m=m500_cl, r=r500_cl))

    # Calculate the model values for the AGN candidates in the cluster
    rad_model = model_rate(z_cl, m500_cl, r500_cl, r_dist_r500, max_radius, params_true)

    # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
    # the cluster.
    max_rate = np.max(rad_model)
    max_rate_arcmin2 = max_rate * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin)**2 / r500_cl**2

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
    # print('Number of points in final selection: {}'.format(len(x_final)))

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

        # Reorder the columns in the cluster for ascetic reasons.
        AGN_cat = AGN_list['SPT_ID', 'REDSHIFT', 'M500', 'r500', 'radial_arcmin', 'radial_r500', 'MASK_NAME']
        AGN_cats.append(AGN_cat)

        # Create a histogram of the objects in the cluster using evenly spaced bins on radius
        hist, bin_edges = np.histogram(AGN_cat['radial_r500'], bins=np.linspace(0, max_radius, num=10))

        # Compute area in terms of r500^2
        area_edges = np.pi * bin_edges ** 2
        area = area_edges[1:len(area_edges)] - area_edges[0:len(area_edges) - 1]

        # Calculate the good pixel fraction for each annulus area (We can do this here for now as all mock clusters use
        # the same mask. If we source from real masks we'll need to move this up into the loop.)
        # For our center set a dummy center at (0,0)
        SZ_center = Table([[0], [0]], names=['RA', 'DEC'])
        gpf = good_pixel_fraction(bin_edges, z_cl, r500_cl, mask_file, SZ_center)

        # Scale our area by the good pixel fraction
        scaled_area = area * gpf

        # Use small-N Poisson error of counts in each bin normalized by the area of the bin
        count_err = small_poisson(hist)
        err = [count_err_ul / scaled_area for count_err_ul in count_err]
        np.nan_to_num(err, copy=False)

        # Calculate the model for this cluster
        rall = np.linspace(0, 2, num=100)
        model_cl = model_rate(z_cl, m500_cl, r500_cl, rall, max_radius, params_true)

        # Store the binned data into the dictionaries
        hist_heights.update({spt_id: hist})
        hist_scaled_areas.update({spt_id: scaled_area})
        hist_errors.update({spt_id: err})
        hist_models.update({spt_id: model_cl})

outAGN = vstack(AGN_cats)

print('\n------\nparameters: {param}\nTotal number of clusters: {cl} \t Total number of objects: {agn}'
      .format(param=params_true, cl=len(outAGN.group_by('SPT_ID').groups.keys), agn=len(outAGN)))
# outAGN.write('Data/MCMC/Mock_Catalog/Catalogs/Cutoff_Radius/'
#              'mock_AGN_catalog_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_maxr{maxr:.2f}.cat'
#              .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, maxr=max_radius),
#              format='ascii', overwrite=True)

# Diagnostic Plots
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

# Average the cluster histograms
stacked_heights = np.sum(np.array(list(hist_heights.values())), axis=0)
stacked_areas = np.sum(np.array(list(hist_scaled_areas.values())), axis=0)
stacked_hist = stacked_heights / stacked_areas

# Find the errors using the fractional Poisson error in the bin.
stacked_err = (np.sqrt(stacked_heights) / stacked_heights) * stacked_hist

# Average the cluster models
stacked_model = np.nanmean(list(hist_models.values()), axis=0)

# Find the scatter on the models
stacked_model_err = np.nanstd(list(hist_models.values()), axis=0)

# A grid of radii for the data to be plotted on
bin_edges = np.linspace(0, max_radius, num=10)
bins = (bin_edges[1:len(bin_edges)] - bin_edges[0:len(bin_edges)-1]) / 2. + bin_edges[0:len(bin_edges)-1]

# A grid of radii for the model to be plotted on
rall = np.linspace(0, 2.0, 100)

# Overplot the normalized binned data with the model rate
fig, ax = plt.subplots()
ax.errorbar(bins, stacked_hist, yerr=stacked_err[::-1, ...], fmt='o', color='C1',
            label='Mock AGN Candidate Surface Density')
ax.plot(rall, stacked_model, color='C0', label='Model Rate')
ax.fill_between(rall, y1=stacked_model+stacked_model_err, y2=stacked_model-stacked_model_err, color='C0', alpha=0.2)
ax.set(title='Comparison of Sampled Points to Model (Stacked Sample)',
       xlabel=r'$r/r_{{500}}$', ylabel=r'Rate per cluster [$r_{{500}}^{-2}$]')
ax.legend()
plt.show()
# fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/New_Stacking/'
#             'mock_AGN_binned_check_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_maxr{maxr:.2f}'
#             '_gpf_stacked_sumN_sumA_frac_pois_err.pdf'
#             .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, maxr=max_radius), format='pdf')

# Pull 5 random clusters to see how their data compares to their model
# cluster_ids = np.random.choice(list(hist_heights.keys()), size=5)
for cluster_id in list(hist_heights.keys()):
    # Grab the right cluster's histogram height, scaled area, error, and model
    cl_hist_height = hist_heights[cluster_id]
    cl_scaled_area = hist_scaled_areas[cluster_id]
    cl_error = hist_errors[cluster_id]
    cl_model = hist_models[cluster_id]

    # For identification purposes grab the cluster's redshift, mass, and r500 for the title of the plot
    cl_z = outAGN[outAGN['SPT_ID'] == cluster_id]['REDSHIFT'][0]
    cl_m500 = outAGN[outAGN['SPT_ID'] == cluster_id]['M500'][0]
    cl_r500 = outAGN[outAGN['SPT_ID'] == cluster_id]['r500'][0]

    fig, ax = plt.subplots()
    ax.errorbar(bins, cl_hist_height / cl_scaled_area, yerr=cl_error[::-1, ...], fmt='o', color='C1',
                label='AGN candidates per Scaled Area')
    ax.plot(rall, cl_model, color='C0', label='Model')
    ax.set(title='Comparison of Sampled Points to Model\n'
                 r'ID: {id}  $z$ = {z:.2f}, $M_{{500}}$ = {m:.2e} $M_\odot$, $r_{{500}}$ = {r:.2f} Mpc'
           .format(id=cluster_id, z=cl_z, m=cl_m500, r=cl_r500),
           xlabel=r'$r/r_{{500}}$', ylabel=r'Rate [$r_{{500}}^{-2}$]')
    ax.legend()
    plt.show()
    # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/New_Stacking/'
    #             'mock_AGN_binned_check_t{theta:.2f}_e{eta:.2f}_z{zeta:.2f}_b{beta:.2f}_maxr{maxr:.2f}_gpf_mock{id}.pdf'
    #             .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, maxr=max_radius, id=cluster_id),
    #             format='pdf')
