"""
SPTcl-IRAGN-close_pairs.py
Author: Benjamin Floyd

Using only the existing IRAGN catalog, find the fraction of close pairs.
"""
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import QTable
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def radial_areas(r, z, r500, center, mask_name) -> u.Quantity:
    # Read in the mask file and the mask file's WCS
    image, header = fits.getdata(mask_name, header=True)
    image_wcs = WCS(header)

    # From the WCS get the pixel scale
    pix_scale = image_wcs.proj_plane_pixel_scales()[0]
    pix_area = image_wcs.proj_plane_pixel_area()

    # Convert our center into pixel units
    center_pix = center.to_pixel(image_wcs)

    # Convert our radius to pixels
    r_pix = r * r500 * cosmo.arcsec_per_kpc_proper(z).to(pix_scale.unit / u.Mpc) / pix_scale
    r_pix = r_pix.value

    # Because we potentially integrate to larger radii than can be fit on the image we will need to increase the size of
    # our mask. To do this, we will pad the mask with a zeros out to the radius we need.
    # Find the width needed to pad the image to include the largest radius inside the image.
    width = ((int(round(np.max(r_pix) - center_pix[1])),
              int(round(np.max(r_pix) - (image.shape[0] - center_pix[1])))),
             (int(round(np.max(r_pix) - center_pix[0])),
              int(round(np.max(r_pix) - (image.shape[1] - center_pix[0])))))

    # Ensure that we are adding a non-negative padding width.
    width = tuple(tuple([i if i >= 0 else 0 for i in axis]) for axis in width)

    large_image = np.pad(image, pad_width=width, mode='constant', constant_values=0)

    # Generate a list of all pixel coordinates in the padded image
    image_coords = np.dstack(np.mgrid[0:large_image.shape[0], 0:large_image.shape[1]]).reshape(-1, 2)

    # The center pixel's coordinate needs to be transformed into the large image system
    center_coord = np.array(center_pix) + np.array([width[1][0], width[0][0]])
    center_coord = center_coord.reshape((1, 2))

    # Compute the distance matrix. The entries are a_ij = sqrt((x_j - cent_x)^2 + (y_i - cent_y)^2)
    image_dists = cdist(image_coords, np.flip(center_coord)).reshape(large_image.shape)

    # select all pixels that are within the annulus
    annulus_area = []
    for j in np.arange(len(r_pix) - 1):
        pix_ring = large_image[np.where((r_pix[j] <= image_dists) & (image_dists < r_pix[j + 1]))]

        # Calculate the area
        annulus_area.append(np.sum(pix_ring) * pix_area)

    return u.Quantity(annulus_area)


sptcl_iragn = QTable.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN_no-stars.fits')
sptcl_iragn = sptcl_iragn.group_by('SPT_ID')

radial_bins = np.arange(0, 2, 0.1)
radial_bin_centers = radial_bins[:-1] + np.diff(radial_bins) / 2

all_agn_pair_counts, all_non_agn_pair_counts, all_galaxy_counts = [], [], []
all_bin_areas, all_bin_areas_r500 = [], []
for cluster in sptcl_iragn.groups:
    # Cluster level information
    cluster_z = cluster['REDSHIFT'][0]
    cluster_r500 = cluster['R500'][0]
    cluster_center = SkyCoord(cluster['SZ_RA'][0], cluster['SZ_DEC'][0], unit=u.deg)
    cluster_mask_name = cluster['MASK_NAME'][0]

    # Separate the AGN from the non-AGN in the cluster
    agn = cluster[cluster['SELECTION_MEMBERSHIP'] >= 0.5]
    non_agn = cluster[cluster['SELECTION_MEMBERSHIP'] < 0.5]

    # We will apply a distance to all galaxies based on the cluster redshift.
    cluster_dist = cosmo.angular_diameter_distance(cluster_z)


    all_coords = SkyCoord(cluster['ALPHA_J2000'], cluster['DELTA_J2000'], distance=cluster_dist)
    agn_coords = SkyCoord(agn['ALPHA_J2000'], agn['DELTA_J2000'], distance=cluster_dist)
    non_agn_coords = SkyCoord(non_agn['ALPHA_J2000'], non_agn['DELTA_J2000'], distance=cluster_dist)

    agn_idx, agn_all_idx, agn_d2d, agn_d3d = all_coords.search_around_3d(agn_coords, 50 * u.kpc)
    non_agn_idx, non_agn_all_idx, non_agn_d2d, non_agn_d3d = non_agn_coords.search_around_3d(non_agn_coords, 50 * u.kpc)

    # Identify pairs
    agn_idx_unique, agn_idx_counts = np.unique(agn_idx, return_counts=True)
    non_agn_idx_unique, non_agn_counts = np.unique(non_agn_idx, return_counts=True)

    agn_idx_pairs = agn_idx_unique[agn_idx_counts > 1]
    non_agn_idx_pairs = non_agn_idx_unique[non_agn_counts > 1]

    agn_pairs = agn[agn_idx_pairs]
    non_agn_pairs = non_agn[non_agn_idx_pairs]

    bin_areas = radial_areas(radial_bins, cluster_z, cluster_r500, cluster_center, cluster_mask_name)
    bin_areas_r500 = bin_areas * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc/u.arcmin)**2 / cluster_r500**2

    agn_pair_counts, _ = np.histogram(agn_pairs['RADIAL_SEP_R500'], bins=radial_bins)
    non_agn_pair_counts, _ = np.histogram(non_agn_pairs['RADIAL_SEP_R500'], bins=radial_bins)
    galaxy_counts, _ = np.histogram(cluster['RADIAL_SEP_R500'], bins=radial_bins)

    all_agn_pair_counts.append(agn_pair_counts)
    all_non_agn_pair_counts.append(non_agn_pair_counts)
    all_galaxy_counts.append(galaxy_counts)
    all_bin_areas.append(bin_areas.to(u.arcmin**2))
    all_bin_areas_r500.append(bin_areas_r500)

all_agn_pair_counts = np.nansum(all_agn_pair_counts, axis=0)
all_non_agn_pair_counts = np.nansum(all_non_agn_pair_counts, axis=0)
all_galaxy_counts = np.nansum(all_galaxy_counts, axis=0)
all_bin_areas = np.nansum(all_bin_areas, axis=0)
all_bin_areas_r500 = np.nansum(all_bin_areas_r500, axis=0)

#%%
all_agn_pair_den = all_agn_pair_counts / all_bin_areas
all_non_agn_pair_den = all_non_agn_pair_counts / all_bin_areas
all_galaxy_den = all_galaxy_counts / all_bin_areas
all_galaxy_pair_den = (all_agn_pair_counts + all_non_agn_pair_counts) / all_bin_areas

all_frac_agn_pair_den = all_agn_pair_den / all_galaxy_counts
all_frac_non_agn_pair_den = all_non_agn_pair_den / all_galaxy_counts

all_agn_pair_den_r500 = all_agn_pair_counts / all_bin_areas_r500
all_non_agn_pair_den_r500 = all_non_agn_pair_counts / all_bin_areas_r500
all_galaxy_den_r500 = all_galaxy_counts / all_bin_areas_r500
all_galaxy_pair_den_r500 = (all_agn_pair_counts + all_non_agn_pair_counts) / all_bin_areas_r500

all_frac_agn_pair_den_r500 = all_agn_pair_den_r500 / all_galaxy_counts
all_frac_non_agn_pair_den_r500 = all_non_agn_pair_den_r500 / all_galaxy_counts

all_agn_pair_count_err = np.sqrt(np.nansum(all_agn_pair_counts, axis=0))
all_non_agn_pair_counts_err = np.sqrt(np.nansum(all_non_agn_pair_counts, axis=0))
all_galaxy_pair_counts_err = np.sqrt(all_agn_pair_count_err**2 + all_non_agn_pair_counts_err**2)

all_agn_pair_count_log_err = np.abs(all_agn_pair_count_err / (all_agn_pair_counts * np.log(10)))
all_non_agn_pair_count_log_err = np.abs(all_non_agn_pair_counts_err / (all_non_agn_pair_counts * np.log(10)))
all_galaxy_pair_count_log_err = np.abs(all_galaxy_pair_counts_err / ((all_agn_pair_counts + all_non_agn_pair_counts) * np.log(10)))

all_agn_pair_den_log_err = all_agn_pair_count_log_err / all_bin_areas
all_non_agn_pair_den_log_err = all_non_agn_pair_count_log_err / all_bin_areas
all_agn_pair_den_r500_log_err = all_agn_pair_count_log_err / all_bin_areas_r500
all_non_agn_pair_den_r500_log_err = all_non_agn_pair_count_log_err / all_bin_areas_r500

all_galaxy_pair_den_r500_log_err = all_galaxy_pair_count_log_err / all_bin_areas_r500


#%%
def power_law(x, a, b):
    return a + b * x

all_agn_pair_den_popt, all_agn_pair_den_pcov = curve_fit(power_law, radial_bin_centers, np.log10(all_agn_pair_den), sigma=all_agn_pair_den_log_err)
all_non_agn_pair_den_popt, all_non_agn_pair_den_pcov = curve_fit(power_law, radial_bin_centers, np.log10(all_non_agn_pair_den), sigma=all_non_agn_pair_den_log_err)
all_agn_pair_den_r500_popt, all_agn_pair_den_r500_pcov = curve_fit(power_law, radial_bin_centers, np.log10(all_agn_pair_den_r500), sigma=all_agn_pair_den_r500_log_err)
all_non_agn_pair_den_r500_popt, all_non_agn_pair_den_r500_pcov = curve_fit(power_law, radial_bin_centers, np.log10(all_non_agn_pair_den_r500), sigma=all_non_agn_pair_den_r500_log_err)

all_galaxy_pair_den_r500_popt, all_galaxy_pair_den_r500_pcov = curve_fit(power_law, radial_bin_centers, np.log10(all_galaxy_pair_den_r500), sigma=all_galaxy_pair_den_r500_log_err)

all_agn_pair_den_perr = np.sqrt(np.diag(all_agn_pair_den_pcov))
all_non_agn_pair_den_perr = np.sqrt(np.diag(all_non_agn_pair_den_pcov))
all_agn_pair_den_r500_perr = np.sqrt(np.diag(all_agn_pair_den_r500_pcov))
all_non_agn_pair_den_r500_perr = np.sqrt(np.diag(all_non_agn_pair_den_r500_pcov))

all_galaxy_pair_den_r500_perr = np.sqrt(np.diag(all_galaxy_pair_den_r500_pcov))


#%%
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(2*6.4, 2*4.8), constrained_layout=True)
axes[0, 0].scatter(radial_bin_centers, np.log10(all_agn_pair_den),
                   # yerr=all_agn_pair_den_log_err, fmt='o',
                   label='AGN pairs')
axes[0, 0].scatter(radial_bin_centers, np.log10(all_non_agn_pair_den),
                   # yerr=all_non_agn_pair_den_log_err, fmt='o',
                   label='Non-AGN pairs')
axes[0, 0].scatter(radial_bin_centers, np.log10(all_galaxy_den), label='All galaxies')
axes[0, 0].plot(radial_bin_centers, power_law(radial_bin_centers, *all_agn_pair_den_popt),
                label=fr'$a = {all_agn_pair_den_popt[0]:.2f}\pm {all_agn_pair_den_perr[0]:.2f}, '
                      fr'b = {all_agn_pair_den_popt[1]:.2f}\pm {all_agn_pair_den_perr[1]:.2f}$')
axes[0, 0].plot(radial_bin_centers, power_law(radial_bin_centers, *all_non_agn_pair_den_popt),
                label=fr'$a = {all_non_agn_pair_den_popt[0]:.2f}\pm {all_non_agn_pair_den_perr[0]:.2f}, '
                      fr'b = {all_non_agn_pair_den_popt[1]:.2f}\pm {all_non_agn_pair_den_perr[1]:.2f}$')
axes[0, 0].legend()
axes[0, 0].set(ylabel=r'Pair Density [log(arcmin$^{-2}$)]', title='Angular Densities', ylim=(-1.5, 1.5))

axes[0, 1].scatter(radial_bin_centers, np.log10(all_agn_pair_den_r500),) # yerr=all_agn_pair_den_r500_log_err, fmt='o')
axes[0, 1].scatter(radial_bin_centers, np.log10(all_non_agn_pair_den_r500),) # yerr=all_non_agn_pair_den_r500_log_err, fmt='o')
axes[0, 1].scatter(radial_bin_centers, np.log10(all_galaxy_den_r500), label='All galaxies')
axes[0, 1].plot(radial_bin_centers, power_law(radial_bin_centers, *all_agn_pair_den_r500_popt),
                label=fr'$a = {all_agn_pair_den_r500_popt[0]:.2f}\pm {all_agn_pair_den_r500_perr[0]:.2f}, '
                      fr'b = {all_agn_pair_den_r500_popt[1]:.2f}\pm {all_agn_pair_den_r500_perr[1]:.2f}$')
axes[0, 1].plot(radial_bin_centers, power_law(radial_bin_centers, *all_non_agn_pair_den_r500_popt),
                label=fr'$a = {all_non_agn_pair_den_r500_popt[0]:.2f}\pm {all_non_agn_pair_den_r500_perr[0]:.2f}, '
                      fr'b = {all_non_agn_pair_den_r500_popt[1]:.2f}\pm {all_non_agn_pair_den_r500_perr[1]:.2f}$')
axes[0, 1].legend()
axes[0, 1].set(ylabel=r'Pair Density [log($r_{500}^{-2}$)]', title=r'$r_{500}$ Densities', ylim=(2.5, 6.0))

axes[1, 0].scatter(radial_bin_centers, all_agn_pair_den / all_agn_pair_den[0])
axes[1, 0].scatter(radial_bin_centers, all_non_agn_pair_den / all_non_agn_pair_den[0])
axes[1, 0].set(xlabel=r'$r/r_{500}$', ylabel=r'Relative Pair Density', yscale='log')

axes[1, 1].scatter(radial_bin_centers, all_agn_pair_den_r500 / all_agn_pair_den_r500[0])
axes[1, 1].scatter(radial_bin_centers, all_non_agn_pair_den_r500 / all_non_agn_pair_den_r500[0])
axes[1, 1].set(xlabel=r'$r/r_{500}$', ylabel=r'Relative Pair Density', yscale='log')

axes[0, 0].legend()
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/close_pairs/SPTcl-IRAGN_close_pairs_densities.pdf')
plt.show()

#%%
fig, ax = plt.subplots()
ax.scatter(radial_bin_centers, np.log10(all_agn_pair_den_r500),) # yerr=all_agn_pair_den_r500_log_err, fmt='o')
ax.scatter(radial_bin_centers, np.log10(all_non_agn_pair_den_r500),) # yerr=all_non_agn_pair_den_r500_log_err, fmt='o')
ax.scatter(radial_bin_centers, np.log10(all_galaxy_pair_den_r500))
ax.plot(radial_bin_centers, power_law(radial_bin_centers, *all_agn_pair_den_r500_popt),
        label=fr'AGN: $a = {all_agn_pair_den_r500_popt[0]:.2f}\pm {all_agn_pair_den_r500_perr[0]:.2f}, '
              fr'b = {all_agn_pair_den_r500_popt[1]:.2f}\pm {all_agn_pair_den_r500_perr[1]:.2f}$')
ax.plot(radial_bin_centers, power_law(radial_bin_centers, *all_non_agn_pair_den_r500_popt),
        label=fr'Non-AGN: $a = {all_non_agn_pair_den_r500_popt[0]:.2f}\pm {all_non_agn_pair_den_r500_perr[0]:.2f}, '
              fr'b = {all_non_agn_pair_den_r500_popt[1]:.2f}\pm {all_non_agn_pair_den_r500_perr[1]:.2f}$')
ax.plot(radial_bin_centers, power_law(radial_bin_centers, *all_galaxy_pair_den_r500_popt),
        label=fr'All: $a = {all_galaxy_pair_den_r500_popt[0]:.2f}\pm {all_galaxy_pair_den_r500_perr[0]:.2f}, '
              fr'b = {all_galaxy_pair_den_r500_popt[1]:.2f}\pm {all_galaxy_pair_den_r500_perr[1]:.2f}$')
ax.legend()
ax.set(ylabel=r'Pair Density [log($r_{500}^{-2}$)]', xlabel=r'$r/r_{500}$', title=r'$r_{500}$ Densities')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/close_pairs/SPTcl-IRAGN_close_pairs_densities_r500.pdf')
plt.show()

#%%
fig, ax = plt.subplots()
ax.scatter(radial_bin_centers, all_agn_pair_den_r500/all_galaxy_pair_den_r500,) # yerr=all_agn_pair_den_r500_log_err, fmt='o')
ax.scatter(radial_bin_centers, all_non_agn_pair_den_r500/all_galaxy_pair_den_r500,) # yerr=all_non_agn_pair_den_r500_log_err, fmt='o')
ax.scatter(radial_bin_centers, all_galaxy_pair_den_r500/all_galaxy_pair_den_r500)
# ax.plot(radial_bin_centers, power_law(radial_bin_centers, *all_agn_pair_den_r500_popt)/np.log10(all_galaxy_pair_den_r500),
#         label=fr'AGN: $a = {all_agn_pair_den_r500_popt[0]:.2f}\pm {all_agn_pair_den_r500_perr[0]:.2f}, '
#               fr'b = {all_agn_pair_den_r500_popt[1]:.2f}\pm {all_agn_pair_den_r500_perr[1]:.2f}$')
# ax.plot(radial_bin_centers, power_law(radial_bin_centers, *all_non_agn_pair_den_r500_popt)/np.log10(all_galaxy_pair_den_r500),
#         label=fr'Non-AGN: $a = {all_non_agn_pair_den_r500_popt[0]:.2f}\pm {all_non_agn_pair_den_r500_perr[0]:.2f}, '
#               fr'b = {all_non_agn_pair_den_r500_popt[1]:.2f}\pm {all_non_agn_pair_den_r500_perr[1]:.2f}$')
# ax.plot(radial_bin_centers, power_law(radial_bin_centers, *all_galaxy_pair_den_r500_popt)/np.log10(all_galaxy_pair_den_r500),
#         label=fr'All: $a = {all_galaxy_pair_den_r500_popt[0]:.2f}\pm {all_galaxy_pair_den_r500_perr[0]:.2f}, '
#               fr'b = {all_galaxy_pair_den_r500_popt[1]:.2f}\pm {all_galaxy_pair_den_r500_perr[1]:.2f}$')
# ax.legend()
ax.axhline(0.2, ls='--', color='k')
ax.set(ylabel=r'Fractional Pair Density [$r_{500}^{-2}$]', xlabel=r'$r/r_{500}$', title=r'$r_{500}$ Densities')
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/close_pairs/SPTcl-IRAGN_close_pairs_densities_r500_fractions.pdf')
plt.show()

#%%
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(2*6.4, 2*4.8), constrained_layout=True)
axes[0, 0].scatter(radial_bin_centers, all_frac_agn_pair_den, label='AGN pairs')
axes[0, 0].scatter(radial_bin_centers, all_frac_non_agn_pair_den, label='Non-AGN pairs')
axes[0, 0].set(ylabel=r'Fractional Pair Density [arcmin$^{-2}$]', yscale='log', title='Angular Densities')

axes[0, 1].scatter(radial_bin_centers, all_frac_agn_pair_den_r500)
axes[0, 1].scatter(radial_bin_centers, all_frac_non_agn_pair_den_r500)
axes[0, 1].set(ylabel=r'Fractional Pair Density [$r_{500}^{-2}$]', yscale='log', title=r'$r_{500}$ Densities')

axes[1, 0].scatter(radial_bin_centers, all_frac_agn_pair_den / all_frac_agn_pair_den[0])
axes[1, 0].scatter(radial_bin_centers, all_frac_non_agn_pair_den / all_frac_non_agn_pair_den[0])
axes[1, 0].set(xlabel=r'$r/r_{500}$', ylabel=r'Fractional Relative Pair Density', yscale='log')

axes[1, 1].scatter(radial_bin_centers, all_frac_agn_pair_den_r500 / all_frac_agn_pair_den_r500[0])
axes[1, 1].scatter(radial_bin_centers, all_frac_non_agn_pair_den_r500 / all_frac_non_agn_pair_den_r500[0])
axes[1, 1].set(xlabel=r'$r/r_{500}$', ylabel=r'Fractional Relative Pair Density', yscale='log')

axes[0, 0].legend()
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/close_pairs/SPTcl-IRAGN_close_pairs_fractional_densities.pdf')
plt.show()