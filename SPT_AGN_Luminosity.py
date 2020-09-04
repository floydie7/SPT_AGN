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
from astro_compendium.utils.k_correction import k_correction
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

# Read in the FLAMINGOS J-band filter for comparison with Assef et al. (2011)
flamingos_j = SpectralElement.from_file('Data/Data_Repository/filter_curves/Gemini/South/FLAMINGOS/'
                                        'FLAMINGOS.BARR.J.ColdWitness.txt',
                                        wave_unit=u.nm)

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
    cluster['F280_ABS_MAG'] = cluster['I2_MAG_APER4'].data - dist_mod - k_corr
    # cluster['J_ABS_MAG'] = cluster['I2_MAG_APER4'].data - dist_mod - k_corr
    sub_tables.append(cluster)
print(f'Absolute Magnitudes Computed, run time: {time.process_time() - start_time:.2f} s')

# Recombine the catalog
sptcl_agn = vstack(sub_tables)

# From the absolute magnitude, convert to luminosity
sptcl_agn['Luminosity'] = 10 ** (-(sptcl_agn['F280_ABS_MAG'] - 4.74) / 2.5) * u.Lsun.to(u.erg / u.s)
# sptcl_agn['Luminosity'] = 10 ** (-(sptcl_agn['J_ABS_MAG'] - 4.74) / 2.5) * u.Lsun.to(u.erg / u.s)

#%% Absolute magnitude - redshift
fig, ax = plt.subplots()
ax.scatter(sptcl_agn['REDSHIFT'], sptcl_agn['F280_ABS_MAG'], marker='.')
ax.set(xlabel='Cluster Redshift', ylabel='[F280] Absolute Vega Magnitude')
ax.invert_yaxis()
fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/Absolute_Mags/Plots/SPTcl-IRAGN_f280_Abs_Mag_redshift.pdf')
# plt.show()

#%% Absolute magnitude - redshift with histograms
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2, width_ratios=(7,2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0])
ax.scatter(sptcl_agn['REDSHIFT'], sptcl_agn['F280_ABS_MAG'], marker='.')
ax.set(xlabel='Cluster Redshift', ylabel='[F280] Absolute Vega Magnitude')
ax.invert_yaxis()

ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
ax_histx.tick_params(axis='x', labelbottom=False)
ax_histy.tick_params(axis='y', labelleft=False)

x_bin_width = 0.1
y_bin_width = 0.25
x_bins = np.arange(sptcl_agn['REDSHIFT'].min(), sptcl_agn['REDSHIFT'].max() + x_bin_width, x_bin_width)
y_bins = np.arange(sptcl_agn['F280_ABS_MAG'].min(), sptcl_agn['F280_ABS_MAG'].max() + y_bin_width, y_bin_width)
ax_histx.hist(sptcl_agn['REDSHIFT'], bins=x_bins)
ax_histy.hist(sptcl_agn['F280_ABS_MAG'], bins=y_bins, orientation='horizontal')

ax_histx.set(ylabel=r'$N_{\rm AGN}$')
ax_histy.set(xlabel=r'$N_{\rm AGN}$')

fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/Absolute_Mags/Plots/SPTcl-IRAGN_f280_Abs_Mag_redshift_hist.pdf')
plt.show()

#%% Bolometric luminosity - redshift
fig, ax = plt.subplots()
ax.scatter(sptcl_agn['REDSHIFT'], sptcl_agn['Luminosity'], marker='.')
ax.set(xlabel='Cluster Redshift', ylabel=r'$L_{\rm AGN}$ [erg s$^{-1}$]', yscale='log')
min_y, max_y = ax.get_ylim()
ax1 = ax.twinx()
ax1.set(ylabel=r'$L_{\rm AGN}\, /\, L_\odot$', ylim=[min_y / u.Lsun.to(u.erg / u.s), max_y / u.Lsun.to(u.erg / u.s)],
        yscale='log')
fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/Absolute_Mags/Plots/SPTcl-IRAGN_f280_Lum_redshift.pdf')
# plt.show()

