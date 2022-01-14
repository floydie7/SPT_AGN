"""
Example_SED_plot.py
Author: Benjamin Floyd

Creates a figure for presentation purposes of example SED templates.
"""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from synphot import SourceSpectrum, units

# Read in the SEDs
ell5 = SourceSpectrum.from_file('Data_Repository/SEDs/Polletta-SWIRE/Ell5_template_norm.sed',
                                wave_unit=u.Angstrom, flux_unit=units.FLAM)
sa = SourceSpectrum.from_file('Data_Repository/SEDs/Polletta-SWIRE/Sa_template_norm.sed',
                              wave_unit=u.Angstrom, flux_unit=units.FLAM)
qso1 = SourceSpectrum.from_file('Data_Repository/SEDs/Polletta-SWIRE/QSO1_template_norm.sed',
                                wave_unit=u.Angstrom, flux_unit=units.FLAM)
qso2 = SourceSpectrum.from_file('Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                wave_unit=u.Angstrom, flux_unit=units.FLAM)

# Set wavelength range to plot over
wave = np.linspace(0.1, 1000, num=int(1e5)) * u.um
nu = wave.to(u.Hz, u.spectral())

# Make plot, normalizing the SEDs at 5 um

fig, ax = plt.subplots()
for sed, label in zip([ell5, sa, qso1, qso2], ['Ell5', 'Sa', 'QSO1', 'QSO2']):
    sed_norm = units.convert_flux(5 * u.um, sed(5 * u.um), units.FNU)
    ax.plot(wave, nu * units.convert_flux(wave, sed(wave), units.FNU) / sed_norm, label=label)
ax.legend(frameon=False)

ax.set(xlabel=r'$\lambda$ [$\mu$m]', ylabel=r'Normalized $\nu F_{\nu}$', xscale='log', yscale='log', xlim=[0.1, 1000])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/Example_Polletta_SEDs.pdf')
