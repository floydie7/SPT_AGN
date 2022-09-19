"""
abs_mag_test.py
Author: Benjamin Floyd

Testing the absolute magnitude functions.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astro_compendium.utils.k_correction import k_corr_abs_mag, k_correction
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d
from synphot import SourceSpectrum, SpectralElement, units, Observation
from synphot.models import Box1D

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Vectorize the k_correction function
k_correction_vect = np.vectorize(k_correction, otypes=[float],
                                 excluded=['f_lambda', 'g_lambda_R', 'g_lambda_Q', 'R', 'Q'])

#%% Read in the SEDs (Using the Polletta SWIRE SEDs)
qso1_sed = SourceSpectrum.from_file('../../Data_Repository/SEDs/Polletta-SWIRE/QSO1_template_norm.sed',
                                    wave_unit=u.angstrom, flux_unit=units.FLAM)
qso2_sed = SourceSpectrum.from_file('../../Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                    wave_unit=u.angstrom, flux_unit=units.FLAM)

# Read in the IRAC 3.6 um and 4.5 um filters
ch1_filter = SpectralElement.from_file('../../Data_Repository/filter_curves/Spitzer_IRAC/080924ch1trans_full.txt',
                                       wave_unit=u.um)
ch2_filter = SpectralElement.from_file('../../Data_Repository/filter_curves/Spitzer_IRAC/080924ch2trans_full.txt',
                                       wave_unit=u.um)

# Also create an artificial filter centered at 2.8 um
f280_filter = SpectralElement(Box1D, amplitude=1, x_0=2.8 * u.um, width=0.56 * u.um)

# Add the FLAMINGOS J-band filter
flamingos_j_filter = SpectralElement.from_file('../../Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                                               'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt', wave_unit=u.nm)

app_mag = 15.0
redshift = 0.6
qso1_abs_mag_vega_zp = k_corr_abs_mag(app_mag, z=redshift, f_lambda_sed=qso1_sed, zero_pt_obs_band=179.7 * u.Jy,
                                 zero_pt_em_band='vega', obs_filter=ch2_filter, em_filter=ch1_filter, cosmo=cosmo)
qso1_abs_mag_official_zp = k_corr_abs_mag(app_mag, z=redshift, f_lambda_sed=qso1_sed, zero_pt_obs_band=179.7 * u.Jy,
                                     zero_pt_em_band=280.9 * u.Jy, obs_filter=ch2_filter, em_filter=ch1_filter,
                                     cosmo=cosmo)
qso1_abs_mag_f280 = k_corr_abs_mag(app_mag, z=redshift, f_lambda_sed=qso1_sed, zero_pt_obs_band=179.7 * u.Jy,
                              zero_pt_em_band='vega', obs_filter=ch2_filter, em_filter=f280_filter, cosmo=cosmo)
qso1_abs_mag_j = k_corr_abs_mag(app_mag, z=redshift, f_lambda_sed=qso1_sed, zero_pt_obs_band=179.7 * u.Jy,
                              zero_pt_em_band='vega', obs_filter=ch2_filter, em_filter=flamingos_j_filter, cosmo=cosmo)

print(f"""Apparent magnitude: {app_mag:.2f} mag
Source Redshift: {redshift}
SED: QSO1
Absolute Magnitude (Ch1, Official IRAC): {qso1_abs_mag_official_zp:.2f}
Absolute Magnitude (Ch1, Vega Source): {qso1_abs_mag_vega_zp:.2f}
Absolute Magnitude (F280 Filter, Vega Source): {qso1_abs_mag_f280:.2f}
Absolute Magnitude (J-band, Vega Source): {qso1_abs_mag_j:.2f}""")

#%%
app_mag = 15.0
redshift = 0.6
qso2_abs_mag_vega_zp = k_corr_abs_mag(app_mag, z=redshift, f_lambda_sed=qso2_sed, zero_pt_obs_band=179.7 * u.Jy,
                                 zero_pt_em_band='vega', obs_filter=ch2_filter, em_filter=ch1_filter, cosmo=cosmo)
qso2_abs_mag_official_zp = k_corr_abs_mag(app_mag, z=redshift, f_lambda_sed=qso2_sed, zero_pt_obs_band=179.7 * u.Jy,
                                     zero_pt_em_band=280.9 * u.Jy, obs_filter=ch2_filter, em_filter=ch1_filter,
                                     cosmo=cosmo)
qso2_abs_mag_f280 = k_corr_abs_mag(app_mag, z=redshift, f_lambda_sed=qso2_sed, zero_pt_obs_band=179.7 * u.Jy,
                              zero_pt_em_band='vega', obs_filter=ch2_filter, em_filter=f280_filter, cosmo=cosmo)
qso2_abs_mag_j = k_corr_abs_mag(app_mag, z=redshift, f_lambda_sed=qso2_sed, zero_pt_obs_band=179.7 * u.Jy,
                              zero_pt_em_band='vega', obs_filter=ch2_filter, em_filter=flamingos_j_filter, cosmo=cosmo)

print(f"""Apparent magnitude: {app_mag:.2f} mag
Source Redshift: {redshift}
SED: QSO2
Absolute Magnitude (Ch1, Official IRAC): {qso2_abs_mag_official_zp:.2f}
Absolute Magnitude (Ch1, Vega Source): {qso2_abs_mag_vega_zp:.2f}
Absolute Magnitude (F280 Filter, Vega Source): {qso2_abs_mag_f280:.2f}
Absolute Magnitude (J-band, Vega Source): {qso2_abs_mag_j:.2f}""")

# exit()

#%%
# Load SDSS filters in
sdss_u_filter = SpectralElement.from_file('../../Data_Repository/filter_curves/SDSS/filter_curves.fits', ext=1,
                                          wave_col='wavelength', flux_col='respt')
sdss_g_filter = SpectralElement.from_file('../../Data_Repository/filter_curves/SDSS/filter_curves.fits', ext=2,
                                          wave_col='wavelength', flux_col='respt')
sdss_r_filter = SpectralElement.from_file('../../Data_Repository/filter_curves/SDSS/filter_curves.fits', ext=3,
                                          wave_col='wavelength', flux_col='respt')
sdss_i_filter = SpectralElement.from_file('../../Data_Repository/filter_curves/SDSS/filter_curves.fits', ext=4,
                                          wave_col='wavelength', flux_col='respt')
sdss_z_filter = SpectralElement.from_file('../../Data_Repository/filter_curves/SDSS/filter_curves.fits', ext=5,
                                          wave_col='wavelength', flux_col='respt')

# Load in the Coleman, Wu and Weedman (1980, CWW) Elliptical SED
CWW_E_sed = SourceSpectrum.from_file('../../Data_Repository/SEDs/CWW/CWW_E_ext.sed',
                                     wave_unit=u.angstrom, flux_unit=units.FLAM)

# Compute the K-correction of the CWW E SED for the SSDS u' filter over a range of redshifts
redshift_range = np.linspace(0, 1, num=100)
cww_e_sdss_g_k_corr = k_correction_vect(redshift_range, f_lambda=CWW_E_sed, g_lambda_R='ab', g_lambda_Q='ab',
                                        R=sdss_u_filter, Q=sdss_u_filter)

# Read in the extracted data from Figure 20, Fukugita et al. (1995)
fukugita_fig20 = Table.read('../../Data_Repository/Project_Data/SPT-IRAGN/Absolute_Mags/fukugita_fig20_sdss_ug_kcorr.csv', data_start=2,
                            names=['redshift_u', 'k_correction_u', 'redshift_g', 'k_correction_g'])

fig, ax = plt.subplots()
ax.plot(redshift_range, cww_e_sdss_g_k_corr, label='Mine')
ax.plot(fukugita_fig20['redshift_u'], fukugita_fig20['k_correction_u'], label='Fukugita+95')
ax.legend()
ax.set(title="CWW E SED K-corrections SDSS g'", xlabel='Redshift', ylabel='K-correction')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
# fig.savefig('Data/Plots/K-correction_SDSS_g.pdf')
plt.show()

#%% For reference
cww_e_redshift = 0.2
fukugita_g = interp1d(fukugita_fig20['redshift_g'], fukugita_fig20['k_correction_g'])
cww_e_g_k_corr_z02 = k_correction(cww_e_redshift, f_lambda=CWW_E_sed, g_lambda_R='vega', g_lambda_Q='vega',
                                  R=sdss_g_filter, Q=sdss_g_filter)
fukugita_g_k_corr_z02 = fukugita_g(cww_e_redshift)
print(f'\n\nMy K-correction:{cww_e_g_k_corr_z02:.2f}\nFukugita+95 K-correction: {fukugita_g_k_corr_z02:.2f}')

#%%
sdss_u_obs = Observation(CWW_E_sed, sdss_u_filter)
sdss_g_obs = Observation(CWW_E_sed, sdss_g_filter)
sdss_r_obs = Observation(CWW_E_sed, sdss_r_filter)
sdss_i_obs = Observation(CWW_E_sed, sdss_i_filter)
sdss_z_obs = Observation(CWW_E_sed, sdss_z_filter)

u_flux = sdss_u_obs.integrate(flux_unit=units.FLAM) / sdss_u_filter.integrate()
g_flux = sdss_g_obs.integrate(flux_unit=units.FLAM) / sdss_g_filter.integrate()
r_flux = sdss_r_obs.integrate(flux_unit=units.FLAM) / sdss_r_filter.integrate()
i_flux = sdss_i_obs.integrate(flux_unit=units.FLAM) / sdss_i_filter.integrate()
z_flux = sdss_z_obs.integrate(flux_unit=units.FLAM) / sdss_z_filter.integrate()

photometry = u.Quantity([u_flux, g_flux, r_flux, i_flux, z_flux])
waves = u.Quantity([sdss_u_filter.avgwave(), sdss_g_filter.avgwave(), sdss_r_filter.avgwave(), sdss_i_filter.avgwave(),
                    sdss_z_filter.avgwave()])

fig, ax = plt.subplots()
ax.plot(CWW_E_sed.waveset, CWW_E_sed(CWW_E_sed.waveset, flux_unit=units.FLAM))
ax.scatter(waves, photometry, color='r')
plt.show()
