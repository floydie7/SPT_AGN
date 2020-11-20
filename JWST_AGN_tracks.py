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
m82 = SourceSpectrum.from_file('Data/Data_Repository/SEDs/Polletta-SWIRE/M82_template_norm.sed',
                               wave_unit=u.angstrom, flux_unit=units.FLAM)
qso2 = SourceSpectrum.from_file('Data/Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                wave_unit=u.angstrom, flux_unit=units.FLAM)

# Renormalize the SEDs to 20 Vega mag in Spitzer 4.5 um
irac45 = SpectralElement.from_file('Data/Data_Repository/filter_curves/Spitzer_IRAC/080924ch2trans_full.txt',
                                   wave_unit=u.um)
# m82.normalize(20 * units.VEGAMAG, irac45, vegaspec=vega)
# qso2.normalize(20 * units.VEGAMAG, irac45, vegaspec=vega)

# Load the NIRCam filters
f356w = SpectralElement.from_file('Data/Data_Repository/filter_curves/JWST/nircam_throughputs/'
                                  'modAB_mean/nrc_plus_ote/F356W_NRC_and_OTE_ModAB_mean.txt',
                                  wave_unit=u.um)
f444w = SpectralElement.from_file('Data/Data_Repository/filter_curves/JWST/nircam_throughputs/'
                                  'modAB_mean/nrc_plus_ote/F444W_NRC_and_OTE_ModAB_mean.txt',
                                  wave_unit=u.um)

# Load the MIRI filters
miri_filter_responses = Table.read('Data/Data_Repository/filter_curves/JWST/MIRI/MIRI_Filter_PCE_all_bands.csv',
                                   data_start=2)
miri_filter_responses.rename_column('\ufeffWave', 'Wave')
f1000w = SpectralElement(Empirical1D, points=miri_filter_responses['Wave'][miri_filter_responses['F1000W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F1000W'][miri_filter_responses['F1000W'] != 0.0])
f1280w = SpectralElement(Empirical1D, points=miri_filter_responses['Wave'][miri_filter_responses['F1280W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F1280W'][miri_filter_responses['F1280W'] != 0.0])
f1500w = SpectralElement(Empirical1D, points=miri_filter_responses['Wave'][miri_filter_responses['F1500W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F1500W'][miri_filter_responses['F1500W'] != 0.0])
f1800w = SpectralElement(Empirical1D, points=miri_filter_responses['Wave'][miri_filter_responses['F1800W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F1800W'][miri_filter_responses['F1800W'] != 0.0])
f2100w = SpectralElement(Empirical1D, points=miri_filter_responses['Wave'][miri_filter_responses['F2100W'] != 0.0] * u.um,
                         lookup_table=miri_filter_responses['F2100W'][miri_filter_responses['F2100W'] != 0.0])
#%%
for z in np.arange(1.0, 1.3, 0.05):
    # Redshift the sources
    m82_z = SourceSpectrum(m82.model, z=z)
    qso2_z = SourceSpectrum(qso2.model, z=z)

    # plot the spectra with the observation filters
    fig, ax = plt.subplots()
    ax.plot(m82_z.waveset.to(u.um), m82_z(m82_z.waveset.to(u.um))/m82_z(5500 * u.Angstrom * (1 + z)), label='M82')
    ax.plot(qso2_z.waveset.to(u.um), qso2_z(qso2_z.waveset.to(u.um))/qso2_z(5500 * u.Angstrom * (1 + z)), label='QSO2')

    ax.plot(f356w.waveset.to(u.um), f356w(f356w.waveset.to(u.um))*2)
    ax.plot(f444w.waveset.to(u.um), f444w(f444w.waveset.to(u.um))*2)
    ax.plot(f1000w.waveset.to(u.um), f1000w(f1000w.waveset.to(u.um))*2)
    ax.plot(f1280w.waveset.to(u.um), f1280w(f1280w.waveset.to(u.um))*2)
    ax.plot(f1500w.waveset.to(u.um), f1500w(f1500w.waveset.to(u.um))*2)
    ax.plot(f1800w.waveset.to(u.um), f1800w(f1800w.waveset.to(u.um))*2)
    ax.plot(f2100w.waveset.to(u.um), f2100w(f2100w.waveset.to(u.um))*2)

    ax.legend()
    ax.set(title=f'z = {z:.2f}', xlabel='Wavelength (um)', ylabel='Flux', xlim=[2, 25], ylim=[0, 5])
    plt.show()

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

    elif z > 1.16:
        # 12.8 um band
        m82_miri_blue = Observation(m82_z, f1280w)
        qso2_miri_blue = Observation(qso2_z, f1280w)

        # 18 um band (as on-PAH filter)
        m82_miri_pah = Observation(m82_z, f1800w)
        qso2_miri_pah = Observation(qso2_z, f1800w)

        # 21 um band
        m82_miri_red = Observation(m82_z, f2100w)
        qso2_miri_red = Observation(qso2_z, f2100w)
