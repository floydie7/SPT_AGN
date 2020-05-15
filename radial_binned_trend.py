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
    return a * (1 + (r/rc)**2)**(-1.5 * beta + 0.5)


# Read in the IRAGN table
SPTcl_IRAGN = Table.read('Data/Output/SPTcl_IRAGN.fits')

# Number of clusters
n_clusters = len(SPTcl_IRAGN.group_by('SPT_ID').groups.keys)

# Median r500 and redshift for unit conversions
med_r500 = np.median(SPTcl_IRAGN['R500']) * u.Mpc
med_z = np.median(SPTcl_IRAGN['REDSHIFT'])

# Generate the number count histogram
hist, bin_edges = np.histogram(SPTcl_IRAGN['RADIAL_SEP_R500'], bins='auto')
bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

# Calculate the surface areas for each bin
surface_area_r500 = np.diff(np.pi * bin_edges**2)
surface_area_arcmin = surface_area_r500 * (med_r500 * cosmo.arcsec_per_kpc_proper(med_z).to(u.arcmin/u.Mpc)).value ** 2

surface_den = hist / surface_area_r500 / n_clusters
surface_den_uerr, surface_den_lerr = small_poisson(hist) / surface_area_r500 / n_clusters
surface_den_err = np.sqrt(surface_den_uerr * surface_den_lerr)

# Fit the model
# param_bounds = ([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 2.0])
param_bounds = ([-np.inf, -np.inf, 0.0], [np.inf, 30, np.inf])
# param_bounds = ([0, -5, 0.05], [12, 5, 1.0])
popt, pcov = curve_fit(beta_model, bin_centers, surface_den, sigma=surface_den_err, bounds=param_bounds)
perr = np.sqrt(np.diag(pcov))
print(f"""Parameter fits
a = {popt[0]:.3f} +- {perr[0]:.4f}
beta = {popt[1]:.3f} +- {perr[1]:.4f}
rc = {popt[2]:.3f} +- {perr[2]:.4f}""")

fig, ax = plt.subplots()
ax.errorbar(bin_centers, surface_den, yerr=[surface_den_lerr, surface_den_uerr], xerr=np.diff(bin_edges)/2, fmt='.')
ax.plot(bin_centers, beta_model(bin_centers, *popt))
ax.set(xlabel=r'$R_{500}$', ylabel=r'$\Sigma_{\rm AGN}$ [$R_{500}^{-2}$ per cluster]')
fig.savefig('Data/Plots/SPTcl_IRAGN_radial_binned_trend.pdf', format='pdf')
plt.show()
