"""
Absolute_Mag_test_suite.py
Author: Benjamin Floyd

Computes absolute magnitudes of the SPTcl-IRAGN against a suite of SED template options used for the K-corrections.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from astro_compendium.utils.custom_math import rchisq
from astro_compendium.utils.k_correction import k_correction
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack
from synphot import SourceSpectrum, SpectralElement
from synphot.models import ConstFlux1D

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Read in the SPTcl-IRAGN catalog
spt_iragn = Table.read('Data/Output/SPTcl_IRAGN.fits')
spt_iragn_grp = spt_iragn.group_by('SPT_ID')

# Read in a selection of SED templates
sed_dict = {
    'CWW_E': SourceSpectrum.from_file('../../Data/Data_Repository/SEDs/CWW/CWW_E_ext.sed'),
    'S0': SourceSpectrum.from_file('../../Data/Data_Repository/SEDs/Polletta-SWIRE/S0_template_norm.sed'),
    'Sb': SourceSpectrum.from_file('../../Data/Data_Repository/SEDs/Polletta-SWIRE/Sb_template_norm.sed'),
    'M82': SourceSpectrum.from_file('../../Data/Data_Repository/SEDs/Polletta-SWIRE/M82_template_norm.sed'),
    'QSO1': SourceSpectrum.from_file('../../Data/Data_Repository/SEDs/Polletta-SWIRE/QSO1_template_norm.sed'),
    'QSO2': SourceSpectrum.from_file('../../Data/Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed'),
    'Sey2': SourceSpectrum.from_file('../../Data/Data_Repository/SEDs/Polletta-SWIRE/Sey2_template_norm.sed'),
    'Mrk231': SourceSpectrum.from_file('../../Data/Data_Repository/SEDs/Polletta-SWIRE/Mrk231_template_norm.sed')
}

# Read in the IRAC 4.5 um filter
irac_45 = SpectralElement.from_file('../../Data/Data_Repository/filter_curves/Spitzer_IRAC/080924ch2trans_full.txt',
                                    wave_unit=u.um)

# IRAC 4.5 um official zero-point
irac_45_zp = SourceSpectrum(ConstFlux1D, amplitude=179.7 * u.Jy)

# We'll send the K-corrections to the Johnson B-band
johnson_b = SpectralElement.from_filter('johnson_b')

# Pre-load the AB zero-point reference spectrum
ab_zp = SourceSpectrum(ConstFlux1D, amplitude=1 * u.ABflux)

abs_mag_dict = {}
for sed_name, template in sed_dict.items():
    sub_tables = []
    for cluster in spt_iragn_grp.groups:
        cluster_z = cluster['REDSHIFT'][0]

        # Compute K-corrections
        k_corr = k_correction(cluster_z, f_lambda=template, g_lambda_R=irac_45_zp, g_lambda_Q=ab_zp,
                              R=irac_45, Q=johnson_b)

        # Compute the distance modulus
        dist_mod = cosmo.distmod(cluster_z).value

        # Calculate the absolute magnitude
        abs_mag = cluster['I2_MAG_APER4'].data - dist_mod - k_corr
        sub_tables.append(abs_mag)

    abs_mag_dict[sed_name] = vstack(sub_tables)

# %% Plotting and fitting
bin_width = 0.15
gaussian = lambda x, a, mu, sigma: a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

fig, axarr = plt.subplots(nrows=2, ncols=4, figsize=(3.5 * 6.4, 2 * 4.8), tight_layout=dict(rect=(0, 0, 1, 0.96)))
for ax, (sed_name, abs_mag) in zip(axarr.flatten(), abs_mag_dict.items()):
    abs_mag = abs_mag['I2_MAG_APER4'].data
    bins = np.arange(np.min(abs_mag), np.max(abs_mag) + bin_width, bin_width)
    # bins = np.histogram_bin_edges(abs_mag, bins='auto')
    bin_centers = bins[:-1] + np.diff(bins) / 2

    # Plot the histogram and store the histogram heights so we can fit on them
    abs_mag_hist, _, _ = ax.hist(abs_mag, bins=bins)

    # Set the fitting cutoff to be just over the peak
    cutoff = bin_centers[np.argmax(abs_mag_hist)] + 2 * np.mean(np.diff(bins))
    fitted_points = (abs_mag_hist >= 25) & (bin_centers <= cutoff)

    # Fit the histogram to the Gaussian
    popt, pcov = op.curve_fit(gaussian, bin_centers[fitted_points], abs_mag_hist[fitted_points],
                              sigma=np.sqrt(abs_mag_hist[fitted_points]),
                              p0=(np.max(abs_mag_hist), np.mean(abs_mag), 1))
    perr = np.sqrt(np.diag(pcov))
    red_chi2 = rchisq(abs_mag_hist[fitted_points], gaussian(bin_centers[fitted_points], *popt), n_parameters=3,
                      stdev=np.sqrt(abs_mag_hist[fitted_points]))
    print(f'SED: {sed_name}\n'
          f'a = {popt[0]:.2f}+-{perr[0]:.3f}\n'
          f'mu = {popt[1]:.2f}+-{perr[1]:.3f}\n'
          f'sigma = {popt[2]:.2f}+-{perr[2]:.3f}\n---')

    # Plot the curve
    ax.plot(bin_centers[fitted_points], gaussian(bin_centers[fitted_points], *popt), 'C1-')
    ax.plot(bin_centers, gaussian(bin_centers, *popt), 'C1--')

    ax.text(0.1, 0.8, r'$\mu = {mu:.2f}\pm{mu_err:.3f}$'
                      '\n'
                      r'$\sigma={sig:.2f}\pm{sig_err:.3f}$'
                      '\n'
                      r'$\chi_\nu^2 = {red_chi2:.2f}$'.format(mu=popt[1], mu_err=perr[1], sig=popt[2], sig_err=perr[2],
                                                              red_chi2=red_chi2),
            va='center', transform=ax.transAxes)
    ax.set(title=f'{sed_name}')
for ax in axarr[-1]:
    ax.set(xlabel=r'$M_B$ [AB mag]')
for ax in axarr[:, 0]:
    ax.set(ylabel=r'$N_{AGN}$')
fig.suptitle('SPTcl-IRAGN Absolute Magnitudes in Johnson B')
# plt.tight_layout()
fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/Absolute_Mags/Plots/AGN_abs_mag_suite.pdf')
plt.show()
