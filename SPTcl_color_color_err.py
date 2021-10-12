"""
SPTcl_color_color_err.py
Author: Benjamin Floyd

Generates a plot of color--color_err for both SPT-SZ and SPTpol 100d samples. Additionally, make a histogram and store
it in a file for use in the mock catalog generator script.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# Read in the AGN catalogs
from matplotlib.lines import Line2D

sptsz = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPT-SZ_2500d.fits')
sptpol = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTpol_100d.fits')

# Extact the colors
sptsz_color = sptsz['I1_MAG_APER4'] - sptsz['I2_MAG_APER4']
sptpol_color = sptpol['I1_MAG_APER4'] - sptpol['I2_MAG_APER4']

# Compute the color errors
sptsz_color_err = np.sqrt((2.5 * sptsz['I1_FLUXERR_APER4'] / (sptsz['I1_FLUX_APER4'] * np.log(10))) ** 2 +
                          (2.5 * sptsz['I2_FLUXERR_APER4'] / (sptsz['I2_FLUX_APER4'] * np.log(10))) ** 2)
sptpol_color_err = np.sqrt((2.5 * sptpol['I1_FLUXERR_APER4'] / (sptpol['I1_FLUX_APER4'] * np.log(10))) ** 2 +
                           (2.5 * sptpol['I2_FLUXERR_APER4'] / (sptpol['I2_FLUX_APER4'] * np.log(10))) ** 2)

fig, ax = plt.subplots()
ax.hexbin(sptsz_color, sptsz_color_err, gridsize=50, bins='log', cmap='Blues', mincnt=1, extent=[0.695, 1.4, 0., 0.25])
ax.hexbin(sptpol_color, sptpol_color_err, gridsize=50, bins='log', cmap='Oranges', mincnt=1, extent=[0.695, 1.4, 0., 0.25], alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
handles.extend([Line2D([0], [0], marker='h', color='none', markerfacecolor='lightblue', markeredgecolor='steelblue',
                       markersize=10),
                Line2D([0], [0], marker='h', color='none', markerfacecolor='orange', markeredgecolor='darkorange',
                       markersize=10)])
labels.extend(['SPT-SZ', 'SPTpol 100d'])
plt.legend(handles, labels, frameon=False)
ax.set(xlabel=r'$[3.6] - [4.5]$ (Vega)', ylabel=r'$\delta([3.6] - [4.5])$')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SPT-SZ_SPTpol_color-color_err.pdf')

# Make the histograms
color_bins = np.arange()
