"""
SDWFS_background_WISE_num_counts.py
Author: Benjamin Floyd

Computes the number count distribution for the IRAC AGN sources and fits a log-linear model.
"""

import json
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astro_compendium.utils.small_poisson import small_poisson
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


def log_power_law(x, alpha, beta):
    return alpha + beta * x


def red_chi_sq(ydata, ymodel, n_free, sigma=None):
    if sigma is not None:
        chisq = np.sum(((ydata - ymodel) / sigma) ** 2)
    else:
        chisq = np.sum((ydata - ymodel) ** 2)

    nu = ydata.size - 1 - n_free
    return chisq / nu


# Read in the WISE catalog
wise_catalog = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/SDWFS_catWISE.ecsv')

# We first need to select for the IR-bright AGN in the field
# Select objects within our magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

wise_catalog = wise_catalog[(ch1_bright_mag < wise_catalog['w1mpro']) & (wise_catalog['w1mpro'] <= ch1_faint_mag) &
                            (ch2_bright_mag < wise_catalog['w2mpro']) & (wise_catalog['w2mpro'] <= ch2_faint_mag)]

# Filter the objects to the SDWFS footprint
mask_img, mask_hdr = fits.getdata('Data_Repository/Project_Data/SPT-IRAGN/Masks/SDWFS/'
                                  'SDWFS_full-field_cov_mask11_11.fits', header=True)
mask_wcs = WCS(mask_hdr)

# Determine the area of the mask
sdwfs_area = np.count_nonzero(mask_img) * mask_wcs.proj_plane_pixel_area()

# Convert the mask image into a boolean mask
mask_img = mask_img.astype(bool)

xy_coords = np.array(mask_wcs.world_to_array_index(SkyCoord(wise_catalog['ra'], wise_catalog['dec'], unit=u.deg)))

wise_catalog = wise_catalog[mask_img[*xy_coords]]

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
color_thresholds = sdwfs_purity_data['purity_90_colors']
agn_purity_color = interp1d(z_bins, color_thresholds, kind='previous')

# Further filter the catalog to objects within a magnitude range for fitting the number count distributions
# Magnitude cuts
bright_end_cut = 14.00
faint_end_cut = 16.25

# Filter the catalog within the magnitude range
wise_catalog = wise_catalog[(bright_end_cut < wise_catalog['w2mpro']) & (wise_catalog['w2mpro'] <= faint_end_cut)]

# Bin into 0.25 magnitude bins
mag_bin_width = 0.25
num_counts_mag_bins = np.arange(bright_end_cut, faint_end_cut, mag_bin_width)
mag_bin_centers = num_counts_mag_bins[:-1] + np.diff(num_counts_mag_bins) / 2

num_count_dists = {}
num_count_errs = {}
num_count_symm_errs = {}
model_params = {}
for color_threshold in color_thresholds:
    wise_agn = wise_catalog[wise_catalog['w1mpro'] - wise_catalog['w2mpro'] >= color_threshold]

    # Create histogram
    dn_dm, _ = np.histogram(wise_agn['w2mpro'], bins=num_counts_mag_bins)
    dn_dm_weighted = dn_dm / (sdwfs_area.value * mag_bin_width)
    num_count_dists[f'{color_threshold:.2f}'] = dn_dm_weighted

    # Compute the errors
    dn_dm_err = tuple(err / (sdwfs_area.value * mag_bin_width) for err in small_poisson(dn_dm))[::-1]
    num_count_errs[f'{color_threshold:.2f}'] = dn_dm_err
    dn_dm_symm_err = np.sqrt(dn_dm_err[0] * dn_dm_err[1])
    num_count_symm_errs[f'{color_threshold:.2f}'] = dn_dm_symm_err

    # Fit model to data
    param_opt, param_cov = curve_fit(log_power_law, mag_bin_centers, np.log10(dn_dm_weighted),
                                     sigma=dn_dm_symm_err / dn_dm_weighted, maxfev=1000)
    param_err = np.sqrt(np.diag(param_cov))
    model_params[f'{color_threshold:.2f}'] = {'params': param_opt, 'param_errors': param_err}

# Write out the model fits to a file
with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/model_fits/SDWFS/'
          'SDWFS-WISE_dNdm_loglin_model_fits.json', 'w') as f:
    json.dump(model_params, f, cls=NumpyArrayEncoder)

# %% Create plots
mag_range = np.linspace(faint_end_cut, bright_end_cut, num=200)
for color_threshold in num_count_dists:
    alpha_opt, beta_opt = model_params[color_threshold]['params']
    alpha_opt_err, beta_opt_err = model_params[color_threshold]['param_errors']
    gof = red_chi_sq(np.log10(num_count_dists[color_threshold]), log_power_law(mag_bin_centers, alpha_opt, beta_opt),
                     n_free=2, sigma=num_count_errs[color_threshold] / num_count_dists[color_threshold])
    fig, ax = plt.subplots()
    ax.errorbar(mag_bin_centers, np.log10(num_count_dists[color_threshold]), xerr=mag_bin_width / 2,
                yerr=num_count_errs[color_threshold] / num_count_dists[color_threshold], fmt='.')
    ax.plot(mag_range, log_power_law(mag_range, alpha_opt, beta_opt))
    ax.text(0.05, 0.77,
            s=fr'''Model: $\log\left(\frac{{dN}}{{dm}}\right) = \alpha + \beta m$
    $\alpha = {alpha_opt:.2f} \pm {alpha_opt_err:.2f}$
    $\beta = {beta_opt:.2f} \pm {beta_opt_err:.2f}$
    $\chi^2_{{\nu}} = {gof:.2f}$''', fontsize='large', transform=ax.transAxes)
    ax.set(title=fr'$[3.6] - [4.5] \geq {color_threshold}$', xlabel='[4.5] (Vega)',
           ylabel=r'$\log(dN/dm)$ [deg$^{-2}$ mag$^{-1}$]', ylim=np.log10([8, 400]))
    plt.tight_layout()
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/SDWFS/'
                f'SDWFS-WISE_{color_threshold}_dNdm_loglin_model_fits.pdf')
    plt.show()
    plt.close()
