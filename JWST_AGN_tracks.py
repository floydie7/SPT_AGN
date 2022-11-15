"""
JWST_AGN_tracks.py
Author: Benjamin Floyd

Explores color planes and draws tracks to select for AGN using the proposed NIRCam + MIRI observations
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from synphot import SourceSpectrum, SpectralElement, Observation, units
from synphot.models import Empirical1D

# Load in the Vega reference spectrum
vega = SourceSpectrum.from_vega()

# Load in a couple of SEDs
m82 = SourceSpectrum.from_file('Data_Repository/SEDs/Polletta-SWIRE/M82_template_norm.sed',
                               wave_unit=u.angstrom, flux_unit=units.FLAM)
qso2 = SourceSpectrum.from_file('Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                wave_unit=u.angstrom, flux_unit=units.FLAM)

# Renormalize the SEDs to 20 Vega mag in Spitzer 4.5 um
irac45 = SpectralElement.from_file('Data_Repository/filter_curves/Spitzer_IRAC/080924ch2trans_full.txt',
                                   wave_unit=u.um)
# m82.normalize(20 * units.VEGAMAG, irac45, vegaspec=vega)
# qso2.normalize(20 * units.VEGAMAG, irac45, vegaspec=vega)

# Load the NIRCam filters
f356w = SpectralElement.from_file('Data_Repository/filter_curves/JWST/nircam_throughputs/'
                                  'modAB_mean/nrc_plus_ote/F356W_NRC_and_OTE_ModAB_mean.txt',
                                  wave_unit=u.um)
f444w = SpectralElement.from_file('Data_Repository/filter_curves/JWST/nircam_throughputs/'
                                  'modAB_mean/nrc_plus_ote/F444W_NRC_and_OTE_ModAB_mean.txt',
                                  wave_unit=u.um)

# Load the MIRI filters
miri_filter_responses = Table.read('Data_Repository/filter_curves/JWST/MIRI/MIRI_Filter_PCE_all_bands.csv',
                                   data_start=2)
miri_filter_responses.rename_column('\ufeffWave', 'Wave')
f1000w = SpectralElement(Empirical1D,
                         points=miri_filter_responses['Wave'][miri_filter_responses['F1000W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F1000W'][miri_filter_responses['F1000W'] != 0.0])
f1280w = SpectralElement(Empirical1D,
                         points=miri_filter_responses['Wave'][miri_filter_responses['F1280W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F1280W'][miri_filter_responses['F1280W'] != 0.0])
f1500w = SpectralElement(Empirical1D,
                         points=miri_filter_responses['Wave'][miri_filter_responses['F1500W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F1500W'][miri_filter_responses['F1500W'] != 0.0])
f1800w = SpectralElement(Empirical1D,
                         points=miri_filter_responses['Wave'][miri_filter_responses['F1800W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F1800W'][miri_filter_responses['F1800W'] != 0.0])
f2100w = SpectralElement(Empirical1D,
                         points=miri_filter_responses['Wave'][miri_filter_responses['F2100W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F2100W'][miri_filter_responses['F2100W'] != 0.0])
# %%
for z in np.arange(1.0, 1.3, 0.05):
    # Redshift the sources
    m82_z = SourceSpectrum(m82.model, z=z)
    qso2_z = SourceSpectrum(qso2.model, z=z)

    # plot the spectra with the observation filters
    fig, ax = plt.subplots()
    ax.plot(m82_z.waveset.to(u.um), m82_z(m82_z.waveset.to(u.um)) / m82_z(5500 * u.Angstrom * (1 + z)), label='M82')
    ax.plot(qso2_z.waveset.to(u.um), qso2_z(qso2_z.waveset.to(u.um)) / qso2_z(5500 * u.Angstrom * (1 + z)),
            label='QSO2')

    ax.plot(f356w.waveset.to(u.um), f356w(f356w.waveset.to(u.um)) * 2)
    ax.plot(f444w.waveset.to(u.um), f444w(f444w.waveset.to(u.um)) * 2)
    ax.plot(f1000w.waveset.to(u.um), f1000w(f1000w.waveset.to(u.um)) * 2)
    ax.plot(f1280w.waveset.to(u.um), f1280w(f1280w.waveset.to(u.um)) * 2)
    ax.plot(f1500w.waveset.to(u.um), f1500w(f1500w.waveset.to(u.um)) * 2)
    ax.plot(f1800w.waveset.to(u.um), f1800w(f1800w.waveset.to(u.um)) * 2)
    ax.plot(f2100w.waveset.to(u.um), f2100w(f2100w.waveset.to(u.um)) * 2)

    ax.legend()
    ax.set(title=f'z = {z:.2f}', xlabel='Wavelength (um)', ylabel='Flux', xlim=[2, 25], ylim=[0, 2.5])
    plt.show()

results = []
for z in [1.03, 1.15, 1.16, 1.20, 1.22, 1.30]:
    m82_z = SourceSpectrum(m82.model, z=z)
    qso2_z = SourceSpectrum(qso2.model, z=z)

    # Observe in both 3.5 and 4.4 um bands
    m82_f356w = Observation(m82_z, f356w)
    m82_f444w = Observation(m82_z, f444w)
    qso2_f356w = Observation(qso2_z, f356w)
    qso2_f444w = Observation(qso2_z, f444w)

    if z == 1.03:
        # 10 um band
        m82_miri_blue = Observation(m82_z, f1000w)
        qso2_miri_blue = Observation(qso2_z, f1000w)

        # 15 um band
        m82_miri_pah = Observation(m82_z, f1500w)
        qso2_miri_pah = Observation(qso2_z, f1500w)

        # 18 um band (as red filter)
        m82_miri_red = Observation(m82_z, f1800w)
        qso2_miri_red = Observation(qso2_z, f1800w)

    elif 1.03 < z <= 1.16:
        # 10 um band
        m82_miri_blue = Observation(m82_z, f1000w)
        qso2_miri_blue = Observation(qso2_z, f1000w)

        # 15 um band
        m82_miri_pah = Observation(m82_z, f1500w)
        qso2_miri_pah = Observation(qso2_z, f1500w)

        # 21 um band
        m82_miri_red = Observation(m82_z, f2100w)
        qso2_miri_red = Observation(qso2_z, f2100w)

    else:
        # 12.8 um band
        m82_miri_blue = Observation(m82_z, f1280w)
        qso2_miri_blue = Observation(qso2_z, f1280w)

        # 18 um band (as on-PAH filter)
        m82_miri_pah = Observation(m82_z, f1800w)
        qso2_miri_pah = Observation(qso2_z, f1800w)

        # 21 um band
        m82_miri_red = Observation(m82_z, f2100w)
        qso2_miri_red = Observation(qso2_z, f2100w)

    results.append({'m82': {'f356': m82_f356w, 'f444': m82_f444w,
                            'blue': m82_miri_blue, 'pah': m82_miri_pah, 'red': m82_miri_red},
                    'qso2': {'f356': qso2_f356w, 'f444': qso2_f444w,
                             'blue': qso2_miri_blue, 'pah': qso2_miri_pah, 'red': qso2_miri_red}})

m82_results = {band: u.Quantity([d['m82'][band].integrate(flux_unit=units.FLAM)
                                .to(u.Jy, equivalencies=u.spectral_density(d['m82'][band].bandpass.pivot()))
                                 for d in results]) for band in results[0]['m82']}
qso2_results = {band: u.Quantity([d['qso2'][band].integrate(flux_unit=units.FLAM)
                                 .to(u.Jy, equivalencies=u.spectral_density(d['qso2'][band].bandpass.pivot()))
                                  for d in results]).to(u.Jy) for band in results[0]['qso2']}
#%%
fig, ax = plt.subplots()
ax.plot(np.log10(m82_results['pah'] / m82_results['blue']), np.log10(m82_results['red'] / m82_results['pah']),
        color='C0', label='M82')
ax.plot(np.log10(qso2_results['pah'] / qso2_results['blue']), np.log10(qso2_results['red'] / qso2_results['pah']),
        color='C1', label='QSO2')
ax.scatter(np.log10(m82_results['pah'][0] / m82_results['blue'][0]), np.log10(m82_results['red'][0] / m82_results['pah'][0]),
           marker='o', edgecolor='C0', facecolor='none')
ax.scatter(np.log10(m82_results['pah'][1:] / m82_results['blue'][1:]), np.log10(m82_results['red'][1:] / m82_results['pah'][1:]),
           marker='o', color='C0')
ax.scatter(np.log10(qso2_results['pah'][0] / qso2_results['blue'][0]), np.log10(qso2_results['red'][0] / qso2_results['pah'][0]),
           marker='o', edgecolor='C1', facecolor='none')
ax.scatter(np.log10(qso2_results['pah'][1:] / qso2_results['blue'][1:]), np.log10(qso2_results['red'][1:] / qso2_results['pah'][1:]),
           marker='o', color='C1')
ax.set(xlabel=r'$\log(S_{\rm PAH} / S_{\rm blue}$)', ylabel=r'$\log(S_{\rm red} / S_{\rm PAH})$')
ax.legend()
fig.savefig('Data_Repository/Project_Data/Observations/JWST_100d/MIRI_color-color_redshift_tracks.pdf')
plt.show()
