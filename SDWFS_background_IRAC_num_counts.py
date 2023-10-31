"""
SDWFS_background_IRAC_num_counts.py
Author: Benjamin Floyd

Computes the number count distribution for the IRAC AGN sources and fits a power-law model.
"""
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astro_compendium.utils.small_poisson import small_poisson
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


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


# Read in the IR-AGN catalog
sdwfs_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_full-field_IRAGN.fits')

# Read in the mask image and WCS
sdwfs_mask_img, sdwfs_mask_hdr = fits.getdata('Data_Repository/Project_Data/SPT-IRAGN/Masks/SDWFS/'
                                              'SDWFS_full-field_cov_mask11_11.fits', header=True)
sdwfs_wcs = WCS(sdwfs_mask_hdr)

# Compute the total field area from the mask
sdwfs_area = np.count_nonzero(sdwfs_mask_img) * sdwfs_wcs.proj_plane_pixel_area()

# Magnitude cuts
bright_end_cut = 10.45
faint_end_cut = 17.48

# Filter the catalog within the magnitude range
# sdwfs_iragn = sdwfs_iragn[(bright_end_cut < sdwfs_iragn['I2_MAG_APER4']) &
#                           (sdwfs_iragn['I2_MAG_APER4'] <= faint_end_cut)]

# Bin into 0.25 magnitude bins
mag_bin_width = 0.25
num_counts_mag_bins = np.arange(10.25, 17.48, mag_bin_width)
mag_bin_centers = num_counts_mag_bins[:-1] + np.diff(num_counts_mag_bins) / 2

num_count_dists = {}
num_count_errs = {}
model_params = {}
for selection_membership_key in [colname for colname in sdwfs_iragn.colnames if 'SELECTION_MEMBERSHIP' in colname]:
    # Select the galaxies that are AGN for this redshift bin/color selection.
    catalog = sdwfs_iragn[sdwfs_iragn[selection_membership_key] >= 0.5]

    # Create histogram
    dn_dm, _ = np.histogram(catalog['I2_MAG_APER4'], bins=num_counts_mag_bins)
    dn_dm_weighted = dn_dm / (sdwfs_area.value * mag_bin_width)
    num_count_dists[selection_membership_key] = dn_dm_weighted

    # Compute the errors
    dn_dm_err = tuple(err / (sdwfs_area.value * mag_bin_width) for err in small_poisson(dn_dm))[::-1]
    num_count_errs[selection_membership_key] = dn_dm_err
    dn_dm_symm_err = np.sqrt(dn_dm_err[0] * dn_dm_err[1])

#     # Fit model to data
#     param_opt, param_cov = curve_fit(log_power_law, mag_bin_centers, np.log10(dn_dm_weighted),
#                                      sigma=dn_dm_symm_err/dn_dm_weighted, maxfev=1000)
#     param_err = np.sqrt(np.diag(param_cov))
#     model_params[selection_membership_key] = {'params': param_opt, 'param_errors': param_err}
#
# # Write out the model fits to a file
# model_fits = {key[-4:]: vals for key, vals in model_params.items()}
# with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/model_fits/SDWFS/'
#           'SDWFS-IRAC_dNdm_loglin_model_fits.json', 'w') as f:
#     json.dump(model_fits, f, cls=NumpyArrayEncoder)

# %% Create plots
mag_range = np.linspace(faint_end_cut, bright_end_cut, num=200)
for color_threshold in num_count_dists:
    # alpha_opt, beta_opt = model_params[color_threshold]['params']
    # alpha_opt_err, beta_opt_err = model_params[color_threshold]['param_errors']
    # gof = red_chi_sq(np.log10(num_count_dists[color_threshold]), log_power_law(mag_bin_centers, alpha_opt, beta_opt),
    #                  n_free=2, sigma=num_count_errs[color_threshold]/num_count_dists[color_threshold])

    fig, ax = plt.subplots()
    ax.errorbar(mag_bin_centers, np.log10(num_count_dists[color_threshold]), xerr=mag_bin_width / 2,
                yerr=num_count_errs[color_threshold]/(num_count_dists[color_threshold] * np.log(10)), fmt='.')
    # ax.plot(mag_range, log_power_law(mag_range, alpha_opt, beta_opt))
    # ax.text(0.05, 0.77,
    #         s=fr'''Model: $\log\left(\frac{{dN}}{{dm}}\right) = \alpha + \beta m$
    # $\alpha = {alpha_opt:.2f} \pm {alpha_opt_err:.2f}$
    # $\beta = {beta_opt:.2f} \pm {beta_opt_err:.2f}$
    # $\chi^2_{{\nu}} = {gof:.2f}$''', fontsize='large', transform=ax.transAxes)
    ax.set(title=fr'$[3.6] - [4.5] \geq {color_threshold[-4:]}$', xlabel='[4.5] (Vega)',
           ylabel=r'$\log(dN/dm)$ [deg$^{-2}$ mag$^{-1}$]', ylim=[-0.75, 3.75])
    plt.tight_layout()
    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/SDWFS/'
    #             f'SDWFS-IRAC_{color_threshold[-4:]}_dNdm_loglin_model_fits.pdf')
    plt.show()
    plt.close()
