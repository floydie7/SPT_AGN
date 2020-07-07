"""
SPT_AGN_Luminosity.py
Author: Benjamin Floyd

Computes absolute magnitudes (and converts to luminosities) of the SPTcl-IRAGN sample in order to inform a luminosity
cut to be to the selection criteria.
"""

import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from astro_compendium.utils.k_corection import k_correction
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack
from synphot import SourceSpectrum, SpectralElement, units
from synphot.models import Box1D

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Read in the catalog
sptcl_agn = Table.read('Data/Output/SPTcl_IRAGN.fits')

# Read in the QSO2 SED
qso2_sed = SourceSpectrum.from_file('Data/Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                    wave_unit=u.angstrom, flux_unit=units.FLAM)

# Read in the IRAC 4.5 um filter response curve
irac_45 = SpectralElement.from_file('Data/Data_Repository/filter_curves/Spitzer_IRAC/080924ch2trans_full.txt',
                                    wave_unit=u.um)

# Create an artificial filter centered at 2.8 um with a resolution of R = 5.
f280 = SpectralElement(Box1D, amplitude=1, x_0=2.8 * u.um, width=2.8 * u.um / 5)

start_time = time.process_time()
# Compute the F280 absolute magnitudes
sub_tables = []
for cluster in sptcl_agn.group_by('SPT_ID').groups:
    # As the K-correction only depends on the redshift, we only need to compute it once per cluster
    cluster_z = cluster['REDSHIFT'][0]
    k_corr = k_correction(cluster_z, f_lambda=qso2_sed, g_lambda_R=179.7 * u.Jy, g_lambda_Q='vega', R=irac_45, Q=f280)

    # Also compute the distance modulus
    dist_mod = cosmo.distmod(cluster_z).value

    # Compute the absolute magnitudes
    cluster['F280_ABS_MAG'] = cluster['I2_MAG_APER4'].data - dist_mod - k_corr.value
    sub_tables.append(cluster)
print(f'Absolute Magnitudes Computed, run time: {time.process_time() - start_time:.2f} s')

# Recombine the catalog
sptcl_agn = vstack(sub_tables)

# From the absolute magnitude, convert to luminosity
sptcl_agn['Luminosity'] = 10 ** (-(sptcl_agn['F280_ABS_MAG'] - 4.74) / 2.5) * u.Lsun.to(u.erg / u.s)

#%% Fit a Gaussian to the histogram
bin_width = 0.15
mag_bins = np.arange(sptcl_agn['F280_ABS_MAG'].min(), sptcl_agn['F280_ABS_MAG'].max() + bin_width, bin_width)
abs_mag_hist, mag_bins = np.histogram(sptcl_agn['F280_ABS_MAG'], bins=mag_bins)
bin_centers = mag_bins[:-1] + np.diff(mag_bins) / 2
cutoff = bin_centers[np.argmax(abs_mag_hist)] + 2 * bin_width
fitted_points = (-28.5 < bin_centers) & (bin_centers <= cutoff)

gaussian = lambda x, a, mu, sigma: a * np.exp(-(x - mu)**2 / (2 * sigma**2))

popt, pcov = op.curve_fit(gaussian, bin_centers[fitted_points], abs_mag_hist[fitted_points],
                          sigma=np.sqrt(abs_mag_hist[fitted_points]), p0=(200, -24.5, 1.25))
perr = np.sqrt(np.diag(pcov))

print(f'Fit parameters:\na={popt[0]:.2f}+-{perr[0]:.3f}, '
      f'mu={popt[1]:.2f}+-{perr[1]:.3f}, '
      f'sigma={popt[2]:.2f}+-{perr[2]:.3f}')

mean_luminosity = u.Lsun * 10**(-(popt[1] - 4.74) / 2.5)
print(f'Mean Luminosity: {mean_luminosity:.2e} = {mean_luminosity.to(u.erg/u.s):.2e}')

fig, ax = plt.subplots()
ax.hist(sptcl_agn['F280_ABS_MAG'], bins=mag_bins)
ax.plot(bin_centers[fitted_points], gaussian(bin_centers[fitted_points], *popt), 'C1-')
ax.plot(bin_centers, gaussian(bin_centers, *popt), 'C1--')
# ax.plot(bin_centers[bin_centers <= -24], gaussian(bin_centers[bin_centers <= -24], *popt), 'C1--')
ax.set(xlabel='[F280] Absolute Vega Magnitude', ylabel=r'$N_{\rm AGN}$',
       title=rf'Gaussian: $\mu={popt[1]:.2f}\pm{perr[1]:.3f}, \sigma={popt[2]:.2f}\pm{perr[2]:.3f}$')
fig.savefig('Data/Plots/SPTcl-IRAGN_Abs_Mag.pdf')
plt.show()
