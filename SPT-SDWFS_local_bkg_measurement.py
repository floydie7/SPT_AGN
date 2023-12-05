"""
SPT-SDWFS_local_bkg_measurement.py
Author: Benjamin Floyd

A full-stack processing of SPT local background AGN surface densities natively measured with WISE galaxy counts and
scaled to SDWFS IRAC AGN counts. The number count distributions are then integrated to find the final local background
AGN surface density measurements.
"""
import json

import astropy.units as u
import numpy as np
from astro_compendium.utils.small_poisson import small_poisson
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def log_linear_model(x, alpha, beta):
    return alpha + beta * x


def power_model(x, alpha, beta):
    return alpha * x ** beta


def sdwfs_wise_gal_irac_agn(sdwfs_wise_gal, sdwfs_irac_agn_bright):
    """Compute scaling factors per mag bin for SDWFS IRAC AGN number counts to SDWFS WISE galaxy number counts."""
    # Bin into 0.25 magnitude bins
    mag_bin_width = 0.25
    num_counts_mag_bins = np.arange(ch2_bright_mag, ch2_faint_mag, mag_bin_width)
    mag_bin_centers = num_counts_mag_bins[:-1] + np.diff(num_counts_mag_bins) / 2

    # Create histogram for the WISE galaxies
    wise_gal_dn_dm, _ = np.histogram(sdwfs_wise_gal['w2mpro'], bins=num_counts_mag_bins)
    wise_gal_dn_dm_weighted = wise_gal_dn_dm / (sdwfs_area.value * mag_bin_width)

    # Compute the WISE galaxy errors
    wise_gal_dn_dm_err = tuple(err / (sdwfs_area.value * mag_bin_width) for err in small_poisson(wise_gal_dn_dm))[::-1]

    # Iterate through the AGN selection thresholds and build the IRAC AGN histograms
    sdwfs_irac_agn_dn_dm = {}
    selection_membership_columns = [colname for colname in sdwfs_irac_agn_bright.colnames
                                    if 'SELECTION_MEMBERSHIP' in colname]
    for selection_membership in selection_membership_columns:
        # Make the AGN selection for the color threshold
        irac_agn = sdwfs_irac_agn_bright[sdwfs_irac_agn_bright[selection_membership] >= 0.5]

        # Create the IRAC AGN histogram
        irac_agn_dn_dm, _ = np.histogram(irac_agn['I2_MAG_APER4'], bins=num_counts_mag_bins,
                                         weights=irac_agn['COMPLETENESS_CORRECTION'])
        irac_agn_dn_dm_weighted = irac_agn_dn_dm / (sdwfs_area.value * mag_bin_width)

        # Compute the IRAC AGN errors
        irac_agn_dn_dm_err = tuple(err / (sdwfs_area.value * mag_bin_width)
                                   for err in small_poisson(irac_agn_dn_dm))[::-1]

        # Compute the scaling fractions from the WISE galaxy dN/dm to the IRAC AGN dN/dm
        scaling_fract = irac_agn_dn_dm_weighted / wise_gal_dn_dm_weighted

        sdwfs_irac_agn_dn_dm[selection_membership] = {'hist': irac_agn_dn_dm_weighted,
                                                      'err': irac_agn_dn_dm_err,
                                                      'scaling_frac': scaling_fract}

    return sdwfs_irac_agn_dn_dm


def spt_wise_gal_irac_agn(spt_wise_gal, wise_irac_scaling):
    # Compute the area of the annulus
    spt_bkg_area = np.pi * (outer_radius_deg ** 2 - inner_radius_deg ** 2)

    # Bin into 0.25 magnitude bins
    mag_bin_width = 0.25
    num_counts_mag_bins = np.arange(ch2_bright_mag, ch2_faint_mag, mag_bin_width)

    # Create histogram
    spt_wise_dn_dm, _ = np.histogram(spt_wise_gal['w2mpro'], bins=num_counts_mag_bins)
    spt_wise_dn_dm_weighted = spt_wise_dn_dm / (spt_bkg_area.value * mag_bin_width)

    # Compute the errors
    spt_wise_dn_dm_err = tuple(err / (spt_bkg_area.value * mag_bin_width) for err in small_poisson(spt_wise_dn_dm))[
                         ::-1]

    # Determine the fractional error
    spt_wise_frac_err = spt_wise_dn_dm_err / spt_wise_dn_dm_weighted

    # Scale the data to IRAC AGN levels
    spt_irac_dn_dm = spt_wise_dn_dm_weighted * wise_irac_scaling

    # Fix the errors using a constant fractional error
    spt_irac_dn_dm_err = spt_irac_dn_dm * spt_wise_frac_err

    return spt_irac_dn_dm, spt_irac_dn_dm_err


def spt_irac_sdwfs_irac(spt_irac_dn_dm, spt_irac_dn_dm_err, sdwfs_full_dn_dm, sdwfs_full_dn_dm_err, mag_bin_centers):
    """Fits a simple model to the data and scales the SDWFS data to have the same amplitude as the WISE "AGN"."""

    # First we need to only select a narrow range where data is reliable
    spt_irac_dn_dm_narrow = spt_irac_dn_dm[(15. < mag_bin_centers) & (mag_bin_centers <= 16.5)]
    sdwfs_full_dn_dm_narrow = sdwfs_full_dn_dm[(15. < mag_bin_centers) & (mag_bin_centers <= 16.5)]
    mag_bin_centers_narrow = mag_bin_centers[(15. < mag_bin_centers) & (mag_bin_centers <= 16.5)]

    # Do the same with the errors
    spt_irac_dn_dm_err_narrow = np.array(spt_irac_dn_dm_err)[:, (15. < mag_bin_centers) & (mag_bin_centers <= 16.5)]
    sdwfs_full_dn_dm_err_narrow = np.array(sdwfs_full_dn_dm_err)[:, (15. < mag_bin_centers) & (mag_bin_centers <= 16.5)]

    # Symmetrize the errors
    spt_irac_dn_dm_symerr_narrow = np.sqrt(spt_irac_dn_dm_err_narrow[0] * spt_irac_dn_dm_err_narrow[1])
    sdwfs_full_dn_dm_symerr_narrow = np.sqrt(sdwfs_full_dn_dm_err_narrow[0] * sdwfs_full_dn_dm_err_narrow[1])

    # Take logarithms of the data for better fitting performance
    spt_irac_dn_dm_narrow_log = np.log10(spt_irac_dn_dm_narrow)
    spt_irac_dn_dm_symerr_narrow_log = spt_irac_dn_dm_symerr_narrow / (spt_irac_dn_dm_narrow * np.log(10))

    sdwfs_full_dn_dm_narrow_log = np.log10(sdwfs_full_dn_dm_narrow)
    sdwfs_full_dn_dm_symerr_narrow_log = sdwfs_full_dn_dm_symerr_narrow / (sdwfs_full_dn_dm_narrow * np.log(10))

    # Fit a log-power model to the data
    spt_popt, spt_pcov = curve_fit(log_linear_model, mag_bin_centers_narrow, spt_irac_dn_dm_narrow_log,
                                   sigma=spt_irac_dn_dm_symerr_narrow_log)
    sdwfs_popt, sdwfs_pcov = curve_fit(log_linear_model, mag_bin_centers_narrow, sdwfs_full_dn_dm_narrow_log,
                                       sigma=sdwfs_full_dn_dm_symerr_narrow_log)

    # Find the difference in the amplitude (intercept) parameter
    amplitude_shift = sdwfs_popt[0] - spt_popt[0]

    return spt_popt, spt_pcov, sdwfs_popt, sdwfs_pcov


def renorm_sdwfs_agn(sdwfs_full_dn_dm, sdwfs_full_dn_dm_err, spt_dn_dm, spt_dn_dm_err):
    # First we need to only select a narrow range where data is reliable
    spt_dn_dm_narrow = spt_dn_dm[(15. < mag_bin_centers) & (mag_bin_centers <= ch2_faint_mag)]
    sdwfs_full_dn_dm_narrow = sdwfs_full_dn_dm[(15. < mag_bin_centers) & (mag_bin_centers <= ch2_faint_mag)]

    # Do the same with the errors
    spt_irac_dn_dm_err_narrow = np.array(spt_dn_dm_err)[:, (15. < mag_bin_centers) & (mag_bin_centers <= ch2_faint_mag)]
    sdwfs_full_dn_dm_err_narrow = np.array(sdwfs_full_dn_dm_err)[:,
                                  (15. < mag_bin_centers) & (mag_bin_centers <= ch2_faint_mag)]

    # Symmetrize the errors
    spt_irac_dn_dm_symerr_narrow = np.sqrt(spt_irac_dn_dm_err_narrow[0] * spt_irac_dn_dm_err_narrow[1])
    sdwfs_full_dn_dm_symerr_narrow = np.sqrt(sdwfs_full_dn_dm_err_narrow[0] * sdwfs_full_dn_dm_err_narrow[1])

    def model(data, a):
        return a * data

    sdwfs_popt, sdwfs_pcov = curve_fit(model, spt_dn_dm_narrow, sdwfs_full_dn_dm_narrow,
                                       sigma=sdwfs_full_dn_dm_symerr_narrow)
    sdwfs_perr = np.sqrt(np.diag(sdwfs_pcov))

    return sdwfs_popt, sdwfs_perr


# %%
# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Read in the SPT WISE galaxy background catalog
spt_wise_gal = Table.read('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/'
                          'large_test_catalog_SPT-CLJ0334-4645.ecsv')

# Read in the SDWFS IRAC AGN catalog
sdwfs_irac_agn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_full-field_IRAGN.fits')

# Read in the SDWFS WISE galaxy catalog
sdwfs_wise_gal = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/SDWFS_catWISE.ecsv')

# Read in the SDWFS mask image and WCS
sdwfs_mask_img, sdwfs_mask_hdr = fits.getdata('Data_Repository/Project_Data/SPT-IRAGN/Masks/SDWFS/'
                                              'SDWFS_full-field_cov_mask11_11.fits', header=True)
sdwfs_wcs = WCS(sdwfs_mask_hdr)

# Determine the area of the mask
sdwfs_area = np.count_nonzero(sdwfs_mask_img) * sdwfs_wcs.proj_plane_pixel_area()

# Convert the mask image into a boolean mask
sdwfs_mask_img = sdwfs_mask_img.astype(bool)

xy_coords = np.array(sdwfs_wcs.world_to_array_index(SkyCoord(sdwfs_wise_gal['ra'], sdwfs_wise_gal['dec'],
                                                             unit=u.deg)))

# Filter the WISE galaxies using the mask
sdwfs_wise_gal = sdwfs_wise_gal[sdwfs_mask_img[*xy_coords]]

# Select objects within our magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

sdwfs_irac_agn = sdwfs_irac_agn[(ch1_bright_mag < sdwfs_irac_agn['I1_MAG_APER4']) &
                                (sdwfs_irac_agn['I1_MAG_APER4'] <= ch1_faint_mag) &
                                (ch2_bright_mag < sdwfs_irac_agn['I2_MAG_APER4']) &
                                (sdwfs_irac_agn['I2_MAG_APER4'] <= ch2_faint_mag)]
sdwfs_wise_gal = sdwfs_wise_gal[(ch1_bright_mag < sdwfs_wise_gal['w1mpro']) &
                                (sdwfs_wise_gal['w1mpro'] <= ch1_faint_mag) &
                                (ch2_bright_mag < sdwfs_wise_gal['w2mpro']) &
                                (sdwfs_wise_gal['w2mpro'] <= ch2_faint_mag)]
spt_wise_gal = spt_wise_gal[(ch1_bright_mag < spt_wise_gal['w1mpro']) &
                            (spt_wise_gal['w1mpro'] <= ch1_faint_mag) &
                            (ch2_bright_mag < spt_wise_gal['w2mpro']) &
                            (spt_wise_gal['w2mpro'] <= ch2_faint_mag)]

# Further restrict the catalogs within the bright range
# wise_bright_mag = 14.
# wise_faint_mag = 16.25

# sdwfs_irac_agn_bright = sdwfs_irac_agn[(wise_bright_mag < sdwfs_irac_agn['I2_MAG_APER4']) &
#                                        (sdwfs_irac_agn['I2_MAG_APER4'] <= wise_faint_mag)]
# sdwfs_wise_gal = sdwfs_wise_gal[(wise_bright_mag < sdwfs_wise_gal['w2mpro']) &
#                                 (sdwfs_wise_gal['w2mpro'] <= wise_faint_mag)]
# spt_wise_gal = spt_wise_gal[(wise_bright_mag < spt_wise_gal['w2mpro']) & (spt_wise_gal['w2mpro'] <= wise_faint_mag)]

# For the SPT WISE galaxy catalog we need to excise the center and select objects in the annulus
spt_wise_gal_cluster_coord = SkyCoord(spt_wise_gal['SZ_RA'][0], spt_wise_gal['SZ_DEC'][0], unit=u.deg)
spt_wise_gal_coords = SkyCoord(spt_wise_gal['ra'], spt_wise_gal['dec'], unit=u.deg)
spt_wise_gal_sep_deg = spt_wise_gal_cluster_coord.separation(spt_wise_gal_coords)
spt_wise_gal_sep_mpc = (spt_wise_gal_sep_deg * cosmo.kpc_proper_per_arcmin(spt_wise_gal['REDSHIFT'][0])
                        .to(u.Mpc / spt_wise_gal_sep_deg.unit))

inner_radius_factor = 3
outer_radius_factor = 7
inner_radius_mpc = inner_radius_factor * spt_wise_gal['R200'][0] * u.Mpc
outer_radius_mpc = outer_radius_factor * spt_wise_gal['R200'][0] * u.Mpc
inner_radius_deg = inner_radius_mpc * cosmo.arcsec_per_kpc_proper(spt_wise_gal['REDSHIFT'][0]).to(u.deg / u.Mpc)
outer_radius_deg = outer_radius_mpc * cosmo.arcsec_per_kpc_proper(spt_wise_gal['REDSHIFT'][0]).to(u.deg / u.Mpc)

spt_wise_gal = spt_wise_gal[(inner_radius_mpc < spt_wise_gal_sep_mpc) & (spt_wise_gal_sep_mpc <= outer_radius_mpc)]

# %% Test Plots
mag_bin_width = 0.25
num_counts_mag_bins = np.arange(ch2_bright_mag, ch2_faint_mag, mag_bin_width)
mag_bin_centers = num_counts_mag_bins[:-1] + np.diff(num_counts_mag_bins) / 2

# Create histogram for the WISE galaxies
wise_gal_dn_dm, _ = np.histogram(sdwfs_wise_gal['w2mpro'], bins=num_counts_mag_bins)
wise_gal_dn_dm_weighted = wise_gal_dn_dm / (sdwfs_area.value * mag_bin_width)

# Compute the WISE galaxy errors
wise_gal_dn_dm_err = tuple(err / (sdwfs_area.value * mag_bin_width) for err in small_poisson(wise_gal_dn_dm))[::-1]

# Compute the area of the annulus
spt_bkg_area = np.pi * (outer_radius_deg ** 2 - inner_radius_deg ** 2)

# Create histogram
spt_wise_dn_dm, _ = np.histogram(spt_wise_gal['w2mpro'], bins=num_counts_mag_bins)
spt_wise_dn_dm_weighted = spt_wise_dn_dm / (spt_bkg_area.value * mag_bin_width)

# Compute the errors
spt_wise_dn_dm_err = tuple(err / (spt_bkg_area.value * mag_bin_width) for err in small_poisson(spt_wise_dn_dm))[::-1]

# Calculate the SDWFS WISE galaxy--IRAC AGN scaling factors
sdwfs_scalings = sdwfs_wise_gal_irac_agn(sdwfs_wise_gal, sdwfs_irac_agn)
spt0334_scaling = sdwfs_scalings[f'SELECTION_MEMBERSHIP_{agn_purity_color(spt_wise_gal["REDSHIFT"][0]):.2f}']

# Apply the scaling factors to our cluster
spt_irac_dndm, spt_irac_dndm_err = spt_wise_gal_irac_agn(spt_wise_gal, spt0334_scaling['scaling_frac'])

# Scale the SDWFS IRAC AGN data to the level of the SPT "AGN"
# spt_param, spt_cov, sdwfs_param, sdwfs_cov = spt_irac_sdwfs_irac(spt_irac_dndm, spt_irac_dndm_err, spt0334_scaling['hist'], spt0334_scaling['err'], mag_bin_centers)
scaling_param, scaling_err = renorm_sdwfs_agn(sdwfs_full_dn_dm=spt0334_scaling['hist'],
                                              sdwfs_full_dn_dm_err=spt0334_scaling['err'], spt_dn_dm=spt_irac_dndm,
                                              spt_dn_dm_err=spt_irac_dndm_err)

sdwfs_irac_dndm_scaled = 1 / scaling_param * spt0334_scaling['hist']

# Integrate the SDWFS dN/dm distribution over the magnitude range to determine the total AGN surface density
bkg_agn_surf_den = simpson(sdwfs_irac_dndm_scaled, mag_bin_centers) * u.deg ** -2

print(f'Background AGN Surface density: {bkg_agn_surf_den:.2f} ({bkg_agn_surf_den.to(u.arcmin ** -2):.3f})')
print(f'Number of AGN in background annulus: {bkg_agn_surf_den * spt_bkg_area:.2f}')
print(f'Number of AGN in approximate LoS region: {(bkg_agn_surf_den * 25 * u.arcmin ** 2).decompose():.2f}')

# %% Make plot of the SDWFS input dN/dm distributions for the scaling factors
fig, ax = plt.subplots()
ax.errorbar(mag_bin_centers, wise_gal_dn_dm_weighted, yerr=wise_gal_dn_dm_err, fmt='o', label='WISE Galaxies')
ax.errorbar(mag_bin_centers, spt0334_scaling['hist'], yerr=spt0334_scaling['err'], fmt='o',
            label=r'IRAC AGN ($[3.6] - [4.5] \geq 0.86$)')
ax.legend()
ax.set(title='SDWFS', xlabel='[4.5] or W2 (Vega)', ylabel=r'$dN/dm$ [deg$^{-2}$ mag$^{-1}$]', yscale='log')
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/cluster_gal-agn_scaling_tests/'
#             'SDWFS_WISEgal-IRACagn_dNdm.pdf')
plt.show()

# %% Make plot of applying the scaling factors on our test cluster
fig, ax = plt.subplots()
ax.errorbar(mag_bin_centers, spt_wise_dn_dm_weighted, yerr=spt_wise_dn_dm_err, fmt='o', label='Original WISE Galaxies')
ax.errorbar(mag_bin_centers, spt_irac_dndm, yerr=spt_irac_dndm_err, fmt='o', label='Scaled IRAC "AGN"')
ax.errorbar(mag_bin_centers, spt0334_scaling['hist'], yerr=spt0334_scaling['err'], fmt='o', label='SDWFS IRAC AGN')
ax.legend()
ax.set(title=r'SPT-CLJ0334-4645   $3 < r/r_{200} \leq 7$', xlabel='[4.5] or W2 (Vega)',
       ylabel=r'$dN/dm$ [deg$^{-2}$ mag$^{-1}$', yscale='log')
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/cluster_gal-agn_scaling_tests/'
#             'wide_mag_range_test_SPT-CLJ0334-4645_3-7r200.pdf')
plt.show()

# %%
xx = np.linspace(15., 16.5, num=100)
xxx = np.linspace(0., 15., num=100)
fig, ax = plt.subplots()
ax.plot(xx, log_linear_model(xx, *spt_param), label='SPT')
ax.plot(xx, log_linear_model(xx, *sdwfs_param), label='SDWFS')
ax.plot(xxx, log_linear_model(xxx, *spt_param), ls='--', c='tab:blue')
ax.plot(xxx, log_linear_model(xxx, *sdwfs_param), ls='--', c='tab:orange')
ax.legend()
ax.set(xlabel='[4.5]', ylabel='log(dn/dm)')
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/cluster_gal-agn_scaling_tests/'
#             'log-linear_fit_comparisons_wide.pdf')
plt.show()

# %%
fig, ax = plt.subplots()
ax.errorbar(mag_bin_centers, spt_wise_dn_dm_weighted, yerr=spt_wise_dn_dm_err, fmt='o', label='Original WISE Galaxies')
ax.errorbar(mag_bin_centers, spt_irac_dndm, yerr=spt_irac_dndm_err, fmt='o', label='Scaled IRAC "AGN"')
ax.errorbar(mag_bin_centers, 1 / scaling_param * spt0334_scaling['hist'],
            yerr=1 / scaling_param * spt0334_scaling['err'],
            fmt='o', label='SDWFS IRAC AGN (scaled)')
ax.axvline(x=15., ls='--', c='k', alpha=0.45)
ax.axvline(x=ch2_faint_mag, ls='--', c='k', alpha=0.45)
ax.legend()
ax.set(title=r'SPT-CLJ0334-4645    $3 < r/r_{200} \leq 7$', xlabel='[4.5] or W2 (Vega)',
       ylabel=r'$dN/dm$ [deg$^{-2}$ mag$^{-1}$]', yscale='log')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/cluster_gal-agn_scaling_tests/'
            'scaled_dndm_distributions.pdf')
plt.show()

# %%
fig, ax = plt.subplots()
ax.errorbar(mag_bin_centers, spt_irac_dndm, yerr=spt_irac_dndm_err, fmt='o', label='Scaled IRAC "AGN"')
ax.errorbar(mag_bin_centers, spt0334_scaling['hist'], yerr=spt0334_scaling['err'],
            fmt='o', label='SDWFS IRAC AGN (original)')
ax.errorbar(mag_bin_centers, 1 / scaling_param * spt0334_scaling['hist'],
            yerr=1 / scaling_param * spt0334_scaling['err'],
            fmt='o', label='SDWFS IRAC AGN (scaled)')
ax.text(0.3, 0.12, s=r'$\left(\frac{dN}{dm}\right)_{\rm SDWFS} \pm \delta\left(\frac{dN}{dm}\right)_{\rm SDWFS}'
                     r' = A \left(\frac{dN}{dm}\right)_{\rm SPT}$',
        transform=ax.transAxes, fontsize='x-large')
ax.text(0.54, 0.03, s=fr'$A = {scaling_param[0]:.2f}$', transform=ax.transAxes, fontsize='x-large')
ax.legend()
ax.set(title=r'SPT-CLJ0334-4645    $3 < r/r_{200} \leq 7$', xlabel='[4.5] or W2 (Vega)',
       ylabel=r'$dN/dm$ [deg$^{-2}$ mag$^{-1}$]', yscale='log', xlim=[15., ch2_faint_mag], ylim=[20, 210])
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/cluster_gal-agn_scaling_tests/'
            'SDWFS_IRAC_AGN_scaling_range.pdf')
plt.show()

# %%
int_agn_surf_den = [simpson(sdwfs_irac_dndm_scaled[i:], mag_bin_centers[i:]) for i, _ in enumerate(mag_bin_centers)]

fig, ax = plt.subplots()
integral_label = ax.scatter(mag_bin_centers, int_agn_surf_den, marker='o', color='tab:blue',
                            label=r'$\Sigma_{\rm AGN}(a > [4.5]) = \int_{a}^{17.48} \frac{dN}{dm}dm$')
bx = ax.twinx()
dndm_label = bx.errorbar(mag_bin_centers, sdwfs_irac_dndm_scaled, yerr=1 / scaling_param * spt0334_scaling['err'],
                         fmt='o',
                         color='tab:green', label='SDWFS IRAC AGN (scaled)')
ax.legend(handles=[integral_label, dndm_label], bbox_to_anchor=(14.15, 180), bbox_transform=ax.transData)
ax.set(title=r'SPT-CLJ0334-4645    $3 < r/r_{200} \leq 7$', xlabel='[4.5] (Vega)',
       ylabel=r'$\Sigma_{\rm AGN}(>[4.5])$ [deg$^{-2}$]')
bx.set(ylabel=r'$dN/dm$ [deg$^{-2}$ mag$^{-1}$]', yscale='log')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/cluster_gal-agn_scaling_tests/'
            'integrated_agn_surface_density.pdf')
plt.show()
