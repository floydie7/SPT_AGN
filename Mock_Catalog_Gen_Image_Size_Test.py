"""
Mock_Catalog_Gen_Image_Size_Test.py
Author: Benjamin Floyd

Attempting to find the issue the mock catalog generation has at large radii.
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
from matplotlib.ticker import AutoMinorLocator

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
    center_pix = image_wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

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

    # The center pixel's coordinate needs to be transformed into the large image system
    # We shifted the origin by (x_pad + 1, y_pad + 1). This is the padding width plus 1 for (x, y) = (0, 0).
    center_coord = np.array(center_pix) + np.array(width) + 1
    center_coord = center_coord.reshape((1, 2))

    # Compute the distance matrix. The entries are a_ij = sqrt((x_j - cent_x)^2 + (y_i - cent_y)^2)
    image_dists = cdist(image_coords, center_coord).reshape(large_image.shape)

    # select all pixels that are within the annulus
    good_pix_frac = []
    pix_area = []
    for i in np.arange(len(r_pix) - 1):
        pix_ring = large_image[np.where((r_pix[i] <= image_dists) & (image_dists < r_pix[i + 1]))]

        # Calculate the fraction
        good_pix_frac.append(np.sum(pix_ring) / len(pix_ring))
        pix_area.append(np.sum(pix_ring) * pix_scale * pix_scale)

    return good_pix_frac, pix_area


# Number of clusters to generate
n_cl = 195

# Set parameter values
theta_true = 12     # Amplitude.
eta_true = 1.2       # Redshift slope
zeta_true = -1.0     # Mass slope
beta_true = 0.5      # Radial slope
C_true = 0.371       # Background AGN surface density

params_true = (theta_true, eta_true, zeta_true, beta_true)

max_radius = 1.0

# Number of bins to use to plot our sampled data points
num_bins = 20

# Dx is set to the image's width in arcmin.
for Dx in [5, 25]:
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
    SPT_data['r500'] = (3 * SPT_data['M500'] * u.Msun/
                        (4 * np.pi * 500 * cosmo.critical_density(SPT_data['REDSHIFT']).to(u.Msun / u.Mpc**3)))**(1/3)
    # SPT_data_z = SPT_data[np.where(SPT_data['REDSHIFT'] >= 0.75)]
    # SPT_data_m = SPT_data[np.where(SPT_data['M500'] <= 5e14)]
    r500_arcmin = SPT_data['r500'] * cosmo.arcsec_per_kpc_proper(SPT_data['REDSHIFT']).to(u.arcmin / u.Mpc)

    SPT_data_r500 = SPT_data[np.where(r500_arcmin <= 2.5 * u.arcmin)]

    cluster_sample = SPT_data_r500

    if Dx == 5:
        Dx_5_hist_heights = {}
        Dx_5_hist_scaled_areas = {}
        Dx_5_hist_errors = {}
        Dx_5_hist_models = {}
        Dx_5_hist_raw_areas = {}
        Dx_5_hist_gpf = {}
        Dx_5_hist_pix_area = {}
    else:
        Dx_25_hist_heights = {}
        Dx_25_hist_scaled_areas = {}
        Dx_25_hist_errors = {}
        Dx_25_hist_models = {}
        Dx_25_hist_raw_areas = {}
        Dx_25_hist_gpf = {}
        Dx_25_hist_pix_area = {}

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

        # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass
        # of the cluster.
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
        # print('Number of points in final selection: {}'.format(len(cluster_back_x)))

        # Calculate the radii of the final AGN scaled by the cluster's r500 radius
        r_final_arcmin = np.sqrt((x_final - Dx / 2.) ** 2 + (y_final - Dx / 2.) ** 2) * u.arcmin
        r_final_r500 = r_final_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin) / r500_cl

        if r_final_r500.size != 0:
            # Create a table of our output objects
            AGN_list = Table([r_final_arcmin, r_final_r500], names=['radial_arcmin', 'radial_r500'])
            AGN_list['SPT_ID'] = spt_id
            AGN_list['SZ_RA'] = 0
            AGN_list['SZ_DEC'] = 0
            AGN_list['M500'] = m500_cl
            AGN_list['REDSHIFT'] = z_cl
            AGN_list['r500'] = r500_cl
            AGN_list['MASK_NAME'] = mask_name

            # Reorder the columns in the cluster for ascetic reasons.
            AGN_cat = AGN_list['SPT_ID', 'SZ_RA', 'SZ_DEC', 'REDSHIFT', 'M500', 'r500',
                               'radial_arcmin', 'radial_r500', 'MASK_NAME']
            AGN_cats.append(AGN_cat)

            # Create a histogram of the objects in the cluster using evenly spaced bins on radius
            hist, bin_edges = np.histogram(AGN_cat['radial_r500'], bins=np.linspace(0, max_radius, num=num_bins+1))

            # Compute area in terms of r500^2
            area_edges = np.pi * bin_edges ** 2
            area = area_edges[1:len(area_edges)] - area_edges[0:len(area_edges) - 1]

            # Calculate the good pixel fraction for each annulus area (We can do this here for now as all mock clusters use
            # the same mask. If we source from real masks we'll need to move this up into the loop.)
            # For our center set a dummy center at (0,0)
            SZ_center = AGN_cat['SZ_RA', 'SZ_DEC'][0]
            gpf, pixel_area = good_pixel_fraction(bin_edges, z_cl, r500_cl, mask_file, SZ_center)

            # Scale our area by the good pixel fraction
            scaled_area = area * gpf

            # Use small-N Poisson error of counts in each bin normalized by the area of the bin
            count_err = small_poisson(hist)
            err = [count_err_ul / scaled_area for count_err_ul in count_err]
            np.nan_to_num(err, copy=False)

            # Calculate the model for this cluster
            rall = np.linspace(0, 2, num=100)
            model_cl = model_rate(z_cl, m500_cl, r500_cl, rall, max_radius, params_true)

            # Remove any radial bin that doesn't have a gpf of 1.
            # hist = hist.astype(float)
            # hist[np.where(np.array(gpf) < 1.0)] = np.nan
            # scaled_area[np.where(np.array(gpf) < 1.0)] = np.nan
            # area[np.where(np.array(gpf) < 1.0)] = np.nan
            #
            # bins = (bin_edges[1:len(bin_edges)] - bin_edges[0:len(bin_edges) - 1]) / 2. + bin_edges[0:len(bin_edges) - 1]
            # bins[np.where(np.array(gpf) < 1.0)] = np.nan
            # max_bin_radius = np.nanmax(bins)

            # model_cl[np.where(rall > max_bin_radius)] = np.nan

            # Store the binned data into the dictionaries
            if Dx == 5:
                Dx_5_hist_heights.update({spt_id: hist})
                Dx_5_hist_scaled_areas.update({spt_id: scaled_area})
                Dx_5_hist_errors.update({spt_id: err})
                Dx_5_hist_models.update({spt_id: model_cl})
                Dx_5_hist_raw_areas.update({spt_id: area})
                Dx_5_hist_gpf.update({spt_id: gpf})
                Dx_5_hist_pix_area.update({spt_id: pixel_area})
            else:
                Dx_25_hist_heights.update({spt_id: hist})
                Dx_25_hist_scaled_areas.update({spt_id: scaled_area})
                Dx_25_hist_errors.update({spt_id: err})
                Dx_25_hist_models.update({spt_id: model_cl})
                Dx_25_hist_raw_areas.update({spt_id: area})
                Dx_25_hist_gpf.update({spt_id: gpf})
                Dx_25_hist_pix_area.update({spt_id: pixel_area})

    outAGN = vstack(AGN_cats)

    print('\n------\nparameters: {param}\nTotal number of clusters: {cl} \t Total number of objects: {agn}'
          .format(param=params_true, cl=len(outAGN.group_by('SPT_ID').groups.keys), agn=len(outAGN)))

# Stack the number counts
Dx_5_stacked_heights = np.nansum(np.array(list(Dx_5_hist_heights.values())), axis=0)
Dx_25_stacked_heights = np.nansum(np.array(list(Dx_25_hist_heights.values())), axis=0)

# Stack the scaled areas
Dx_5_stacked_scaled_areas = np.nansum(np.array(list(Dx_5_hist_scaled_areas.values())), axis=0)
Dx_25_stacked_scaled_areas = np.nansum(np.array(list(Dx_25_hist_scaled_areas.values())), axis=0)

# Calculate the surface densities
Dx_5_surface_density = Dx_5_stacked_heights / Dx_5_stacked_scaled_areas
Dx_25_surface_density = Dx_25_stacked_heights / Dx_25_stacked_scaled_areas

# Calculate the fractional errors
Dx_5_frac_err = np.sqrt(Dx_5_stacked_heights) / Dx_5_stacked_heights
Dx_25_frac_err = np.sqrt(Dx_25_stacked_heights) / Dx_25_stacked_heights

# Apply the fractional error to the number counts
Dx_5_stacked_heights_err = Dx_5_frac_err * Dx_5_stacked_heights
Dx_25_stacked_heights_err = Dx_25_frac_err * Dx_25_stacked_heights

# Apply the fractional error to the surface density
Dx_5_surface_density_err = Dx_5_frac_err * Dx_5_surface_density
Dx_25_surface_density_err = Dx_25_frac_err * Dx_25_surface_density

# Average the cluster models
Dx_5_stacked_model = np.nanmean(list(Dx_5_hist_models.values()), axis=0)
Dx_25_stacked_model = np.nanmean(list(Dx_25_hist_models.values()), axis=0)

# Find the scatter on the models
Dx_5_stacked_model_err = np.nanstd(list(Dx_5_hist_models.values()), axis=0)
Dx_25_stacked_model_err = np.nanstd(list(Dx_25_hist_models.values()), axis=0)

# Radial bins
bin_edges = np.linspace(0, max_radius, num=num_bins+1)
bins = (bin_edges[1:len(bin_edges)] - bin_edges[0:len(bin_edges) - 1]) / 2. + bin_edges[0:len(bin_edges) - 1]

# A grid of radii for the model to be plotted on
rall = np.linspace(0, 2.0, 100)

print('Making plots.')
# Make the plots
fig, axarr = plt.subplots(nrows=3, ncols=2, sharex='col', figsize=[12.8, 9.6])
# 5 arcmin surface density
axarr[0, 0].errorbar(bins, Dx_5_surface_density, yerr=Dx_5_surface_density_err, fmt='o', color='C1',
                     label='Mock AGN Candidate Surface Density')
axarr[0, 0].plot(rall, Dx_5_stacked_model, color='C0', label='Model Rate')
axarr[0, 0].fill_between(rall,
                         y1=Dx_5_stacked_model+Dx_5_stacked_model_err,
                         y2=Dx_5_stacked_model-Dx_5_stacked_model_err, color='C0', alpha=0.2)
axarr[0, 0].set(title='Surface Density (5 arcmin)',
                ylabel=r'Rate per cluster [$r_{{500}}^{-2}$]')
print('Plot 1 done.')

# # 5 arcmin number count
axarr[1, 0].bar(bins, Dx_5_stacked_heights, yerr=Dx_5_stacked_heights_err, width=0.095)
axarr[1, 0].set(title='Number of objects on image', ylabel='Number of Objects')
print('Plot 2 done.')

# 5 arcmin scaled areas
axarr[2, 0].bar(bins, Dx_5_stacked_scaled_areas, width=0.095)
axarr[2, 0].set(title='Scaled Area on image', ylabel=r'Area [$r_{{500}}^{{-2}}$]')
print('Plot 3 done.')

# 25 arcmin surface density
axarr[0, 1].errorbar(bins, Dx_25_surface_density, yerr=Dx_25_surface_density_err, fmt='o', color='C1',
                     label='Mock AGN Candidate Surface Density')
axarr[0, 1].plot(rall, Dx_25_stacked_model, color='C0', label='Model Rate')
axarr[0, 1].fill_between(rall,
                         y1=Dx_25_stacked_model+Dx_5_stacked_model_err,
                         y2=Dx_5_stacked_model-Dx_5_stacked_model_err, color='C0', alpha=0.2)
axarr[0, 1].set(title='Surface Density (25 arcmin)')
print('Plot 4 done.')

# 25 arcmin number count
axarr[1, 1].bar(bins, Dx_25_stacked_heights, yerr=Dx_25_stacked_heights_err, width=0.095)
axarr[1, 1].set(title='Number of objects on image')
print('Plot 5 done.')

# 25 arcmin scaled areas
axarr[2, 1].bar(bins, Dx_25_stacked_scaled_areas, width=0.095)
axarr[2, 1].set(title='Scaled Area on image')
print('Plot 6 done.')

for ax in axarr.flat:
    ax.set(xlabel=r'$r/r_{{500}}$')

fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/Image_Size/Comparison_Plots_5_25_arcmin_gpf_1r500_ang_size_maxr1.pdf',
            format='pdf')

fig, ax = plt.subplots()
ax.errorbar(bins, Dx_5_surface_density, yerr=Dx_5_surface_density_err, fmt='o', color='C1',
            label='Mock AGN Candidate Surface Density')
ax.plot(rall, Dx_5_stacked_model, color='C0', label='Model Rate')
ax.fill_between(rall,
                y1=Dx_5_stacked_model+Dx_5_stacked_model_err,
                y2=Dx_5_stacked_model-Dx_5_stacked_model_err, color='C0', alpha=0.2)
ax.set(title='Surface Density (5 arcmin)', ylabel=r'Rate per cluster [$r_{{500}}^{-2}$]', xlabel=r'$r/r_{{500}}$')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/Image_Size/Stacked_Surface_Density_5_arcmin_gpf_1r500_ang_size_maxr1.pdf',
            format='pdf')

fig, ax = plt.subplots()
ax.errorbar(bins, Dx_25_surface_density, yerr=Dx_25_surface_density_err, fmt='o', color='C1',
            label='Mock AGN Candidate Surface Density')
ax.plot(rall, Dx_25_stacked_model, color='C0', label='Model Rate')
ax.fill_between(rall,
                y1=Dx_25_stacked_model+Dx_25_stacked_model_err,
                y2=Dx_25_stacked_model-Dx_25_stacked_model_err, color='C0', alpha=0.2)
ax.set(title='Surface Density (25 arcmin)', ylabel=r'Rate per cluster [$r_{{500}}^{-2}$]', xlabel=r'$r/r_{{500}}$')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/Image_Size/Stacked_Surface_Density_25_arcmin_gpf_1r500_ang_size_maxr1.pdf',
            format='pdf')

# Make combined plots
fig, ax = plt.subplots()
ax.bar(bins, Dx_25_stacked_heights, yerr=Dx_25_stacked_heights_err, alpha=0.4, width=0.095, color='C1',
       ecolor='darkorange', label='25 arcmin')
ax.bar(bins, Dx_5_stacked_heights, yerr=Dx_5_stacked_heights_err, alpha=0.4, width=0.095, color='C0',
       ecolor='darkblue', label='5 arcmin')
ax.set(title='Number of objects on image', ylabel='Number of Objects', xlabel=r'$r/r_{{500}}$')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/Image_Size/Stacked_Number_Comparison_5_25_arcmin_gpf_1r500_ang_size_maxr1.pdf',
            format='pdf')

fig, ax = plt.subplots()
ax.bar(bins, Dx_25_stacked_scaled_areas, width=0.095, alpha=0.4, color='C1', label='25 arcmin')
ax.bar(bins, Dx_5_stacked_scaled_areas, width=0.095, alpha=0.4, color='C0', label='5 arcmin')
ax.set(title='Scaled Area on image', ylabel=r'Area [$r_{{500}}^{{-2}}$]', xlabel=r'$r/r_{{500}}$')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/Image_Size/'
            'Stacked_Scaled_Area_Comparison_5_25_arcmin_gpf_1r500_ang_size_maxr1.pdf', format='pdf')
