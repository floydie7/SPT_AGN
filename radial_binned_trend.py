"""
radial_binned_trend.py
Author: Benjamin Floyd

Fits a beta model to the stacked SPTcl IRAGN sample in order to establish prior values for the radial parameters.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.optimize import curve_fit

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def small_poisson(n, s=1):
    """
    Calculates the upper and lower Poisson confidence limits for extremely low counts i.e., n << 50. These equations are
    outlined in [Gehrels1986]_.

    .._[Gehrels1986] http://adsabs.harvard.edu/abs/1986ApJ...303..336G

    :param n: The number of Poisson counts.
    :type n: float, array-like
    :param s: The s-sigma Gaussian levels. Defaults to `s=1` sigma.
    :type s: int
    :return: The upper and lower errors corresponding to the confidence levels.
    :rtype: tuple
    """

    # Recast the input as a numpy array
    n = np.asanyarray(n)

    # Parameters for the lower limit equation. These are for the 1, 2, and 3-sigma levels.
    beta = [0.0, 0.06, 0.222]
    gamma = [0.0, -2.19, -1.88]

    # Upper confidence level using equation 9 in Gehrels 1986.
    lambda_u = (n + 1.) * (1. - 1. / (9. * (n + 1.)) + s / (3. * np.sqrt(n + 1.))) ** 3

    # Lower confidence level using equation 14 in Gehrels 1986.
    lambda_l = n * (1. - 1. / (9. * n) - s / (3. * np.sqrt(n)) + beta[s - 1] * n ** gamma[s - 1]) ** 3

    # To clear the lower limit array of any possible NaNs from n = 0 incidences.
    np.nan_to_num(lambda_l, copy=False)

    # Calculate the upper and lower errors from the confidence values.
    upper_err = lambda_u - n
    lower_err = n - lambda_l

    return upper_err, lower_err


def beta_model(r, a, beta, rc):
    C = 0.371 * (med_r500 * cosmo.arcsec_per_kpc_proper(med_z).to(u.arcmin / u.Mpc)).value ** 2
    return a * (1 + (r / rc) ** 2) ** (-1.5 * beta + 0.5) + C

def r5002_to_arcmin2(r):
    return r / (med_r500 * cosmo.arcsec_per_kpc_proper(med_z).to(u.arcmin/u.Mpc)).value**2

def convert_axes(ax):
    min_lim, max_lim = ax.get_ylim()
    ax_arcmin.set_ylim(r5002_to_arcmin2(min_lim), r5002_to_arcmin2(max_lim))
    ax.figure.canvas.draw()


# Read in the IRAGN table
SPTcl_IRAGN = Table.read('Data/Output/SPTcl_IRAGN.fits')

# Number of clusters
n_clusters = len(SPTcl_IRAGN.group_by('SPT_ID').groups.keys)

# Median r500 and redshift for unit conversions
med_r500 = np.median(SPTcl_IRAGN['R500']) * u.Mpc
med_z = np.median(SPTcl_IRAGN['REDSHIFT'])

# Filter the clusters to only be near the median redshift
# SPTcl_IRAGN = SPTcl_IRAGN[np.abs(SPTcl_IRAGN['REDSHIFT'] - med_z) <= 0.05]

# Filter radial separations by their angular separations and by their r500 separations
radial_r500 = SPTcl_IRAGN['RADIAL_SEP_R500'][(SPTcl_IRAGN['RADIAL_SEP_R500'] <= 0.6)]

# Generate the number count histogram
bin_width = 0.05
bins = np.arange(0, radial_r500.max(), bin_width)
hist, bin_edges = np.histogram(radial_r500, bins=bins)
bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

# Calculate the surface areas for each bin
surface_area_r500 = np.diff(np.pi * bin_edges ** 2)
surface_area_arcmin = surface_area_r500 * (
            med_r500 * cosmo.arcsec_per_kpc_proper(med_z).to(u.arcmin / u.Mpc)).value ** 2

surface_den_r500 = hist / surface_area_r500 / n_clusters
surface_den_r500_uerr, surface_den_r500_lerr = small_poisson(hist) / surface_area_r500 / n_clusters
surface_den_err_r500 = np.sqrt(surface_den_r500_uerr * surface_den_r500_lerr)

surface_den_arcmin = hist / surface_area_arcmin / n_clusters
surface_den_arcmin_uerr, surface_den_arcmin_lerr = small_poisson(hist) / surface_area_arcmin / n_clusters
surface_den_err_arcmin = np.sqrt(surface_den_arcmin_uerr * surface_den_arcmin_lerr)

# Fit the model
# param_bounds = ([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 2.0])
param_bounds = ([-np.inf, -np.inf, 0.0], [np.inf, 30, np.inf])
# param_bounds = ([0, -5, 0.05], [12, 5, 1.0])
popt_r500, pcov_r500 = curve_fit(beta_model, bin_centers, surface_den_r500, sigma=surface_den_err_r500,
                                 bounds=param_bounds)
perr_r500 = np.sqrt(np.diag(pcov_r500))
print(f"""Parameter fits (r500)
a = {popt_r500[0]:.3f} +- {perr_r500[0]:.4f}
beta = {popt_r500[1]:.3f} +- {perr_r500[1]:.4f}
rc = {popt_r500[2]:.3f} +- {perr_r500[2]:.4f}""")
# C = {popt_r500[3]:.3f} +- {perr_r500[3]:.4f}

# popt_arcmin, pcov_arcmin = curve_fit(beta_model, bin_centers, surface_den_arcmin, sigma=surface_den_err_arcmin,
#                                      bounds=param_bounds)
# perr_arcmin = np.sqrt(np.diag(pcov_arcmin))
# print(f"""Parameter fits (arcmin)
# a = {popt_arcmin[0]:.3f} +- {perr_arcmin[0]:.4f}
# beta = {popt_arcmin[1]:.3f} +- {perr_arcmin[1]:.4f}
# rc = {popt_arcmin[2]:.3f} +- {perr_arcmin[2]:.4f}""")
# # C = {popt_arcmin[3]:.3f} +- {perr_arcmin[3]:.4f}

# fig, ax = plt.subplots()
# ax.errorbar(bin_centers, surface_den_r500, yerr=[surface_den_r500_lerr, surface_den_r500_uerr],
#             xerr=np.diff(bin_edges) / 2, fmt='.')
# # ax.plot(bin_centers, beta_model(bin_centers, *popt_r500))
# ax.set(xlabel=r'$R_{500}$', ylabel=r'$\Sigma_{\rm AGN}$ [$R_{500}^{-2}$ per cluster]')
# # fig.savefig('Data/Plots/SPTcl_IRAGN_radial_binned_trend.pdf', format='pdf')
# plt.show()
#
# fig, ax = plt.subplots()
# ax.errorbar(bin_centers, surface_den_arcmin, yerr=[surface_den_arcmin_lerr, surface_den_arcmin_uerr],
#             xerr=np.diff(bin_edges) / 2, fmt='.')
# # ax.plot(bin_centers, beta_model(bin_centers, *popt_arcmin))
# ax.set(xlabel=r'$R_{500}$', ylabel=r'$\Sigma_{\rm AGN}$ [arcmin$^{-2}$ per cluster]')
# # fig.savefig('Data/Plots/SPTcl_IRAGN_radial_binned_trend.pdf', format='pdf')
# plt.show()

fig, ax = plt.subplots()
ax_arcmin = ax.twinx()
ax.callbacks.connect('ylim_changed', convert_axes)
ax.errorbar(bin_centers, surface_den_r500, yerr=[surface_den_r500_lerr, surface_den_r500_uerr],
            xerr=np.diff(bin_edges) / 2, fmt='.')
ax.plot(bin_centers, beta_model(bin_centers, *popt_r500))
ax.set(xlabel=r'$R_{500}$', ylabel=r'$\Sigma_{\rm AGN}$ [$R_{500}^{-2}$ per cluster]')
ax_arcmin.set(ylabel=r'$\Sigma_{\rm AGN}$ [arcmin$^{-2}$ per cluster]')
plt.show()
fig.savefig('Data/Plots/SPTcl_IRAGN_radial_binned_0.6r500_with_model.pdf')

