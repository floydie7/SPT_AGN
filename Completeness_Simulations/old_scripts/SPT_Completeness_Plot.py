"""
SPT_Completeness_Plot.py
Author: Benjamin Floyd

This script reads in the completeness dictionaries and generates the plots of the curves.
"""

from __future__ import print_function, division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Read in the dictionaries.
ch1_rates = np.load('Data/Comp_Sim/Results/SPT_I1_results_gaussian_fwhm195_corr01_mag02.npy').item()
ch2_rates = np.load('Data/Comp_Sim/Results/SPT_I2_results_gaussian_fwhm202_corr011_mag02.npy').item()
sdwfs_ch1_rates = np.load('Data/Comp_Sim/Results/SDWFS_I1_results_gaussian_fwhm166_corr005_mag02.npy').item()
sdwfs_ch2_rates = np.load('Data/Comp_Sim/Results/SDWFS_I2_results_gaussian_fwhm172_corr005_mag02.npy').item()

# Extract the magnitude bins
ch1_mag_bins = ch1_rates.pop('magnitude_bins')[:-1]
ch2_mag_bins = ch2_rates.pop('magnitude_bins')[:-1]
sdwfs_ch1_rates.pop('magnitude_bins')
sdwfs_ch2_rates.pop('magnitude_bins')

# Calculate the median curves.
ch1_med_rates = [np.median(e) for e in zip(*ch1_rates.values())]
ch2_med_rates = [np.median(e) for e in zip(*ch2_rates.values())]
sdwfs_ch1_med_rates = [np.median(e) for e in zip(*sdwfs_ch1_rates.values())]
sdwfs_ch2_med_rates = [np.median(e) for e in zip(*sdwfs_ch2_rates.values())]

# Make the plot
channel = 0
model = 'gaussian'
mag_diff = 0.2
# Add 0.25 to bins so that the data point is centered on the bin
ch1_mag_bins += 0.25
ch2_mag_bins += 0.25

# for channel in range(2):
#     if channel == 0:
#         psf_fwhm = 1.95
#         aper_corr = 0.10
#         rates = ch1_rates
#         med_rates = ch1_med_rates
#         bins = ch1_mag_bins
#     else:
#         psf_fwhm = 2.02
#         aper_corr = 0.11
#         rates = ch2_rates
#         med_rates = ch2_med_rates
#         bins = ch2_mag_bins
#
#     # Make the composite plot
#     fig, ax = plt.subplots()
#     ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#     ax.yaxis.set_minor_locator(AutoMinorLocator(2))
#     for curve in rates:
#         ax.plot(bins, rates[curve], 'k-', alpha=0.4)
#     ax.plot(bins, med_rates, 'r-', alpha=1.0, linewidth=2)
#
#     ax.set(xlim=[10.0, 23.0], ylim=[0.0, 1.0], xlabel='Vega Magnitude', ylabel='Recovery Rate',
#            title='Completeness Simulation for Channel {ch} SPT Clusters'.format(ch=channel + 1))
#     fig.savefig('Data/Comp_Sim/Plots/SPT_Comp_Sim_I{ch}_{model}_fwhm{fwhm}_corr{corr}_mag{mag_diff}.pdf'
#                 .format(ch=channel + 1, model=model, fwhm=str(psf_fwhm).replace('.', ''),
#                         corr=str(np.abs(aper_corr)).replace('.', ''), mag_diff=str(mag_diff).replace('.', '')),
#                 format='pdf')
#
# # Make a plot for a specific cluster.
# # Pick a random cluster
# cluster_key = ch1_rates.keys()[np.random.randint(len(ch1_rates.keys()))]
#
# fig, ax = plt.subplots()
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax.plot(ch1_mag_bins, ch1_rates[cluster_key], 'b-', label='Channel 1')
# ax.plot(ch2_mag_bins, ch2_rates[cluster_key], 'r-', label='Channel 2')
# ax.set(xlim=[10.0, 23.0], ylim=[0.0, 1.0], xlabel='Vega Magnitude', ylabel='Recovery Rate',
#        title='Completeness Simulations for {spt_id}'.format(spt_id=cluster_key))
# ax.legend(loc='best')
# fig.savefig('Data/Comp_Sim/Plots/{spt_id}_Comp_Sim_ch1ch2_fwhm195202_corr010011_mag02.pdf'.format(spt_id=cluster_key),
#             format='pdf')
#
# # Make the SPT/SDWFS comparison plot.
# fig, ax = plt.subplots()
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax.plot(ch1_mag_bins, ch1_med_rates, 'b-', label='SPT Channel 1')
# ax.plot(ch1_mag_bins, sdwfs_ch1_med_rates, 'b--', label='SDWFS Channel 1')
# ax.plot(ch2_mag_bins, ch2_med_rates, 'r-', label='SPT Channel 2')
# ax.plot(ch2_mag_bins, sdwfs_ch2_med_rates, 'r--', label='SDWFS Channel 2')
# ax.set(xlim=[10.0, 23.0], ylim=[0.0, 1.0], xlabel='Vega Magnitude', ylabel='Recovery Rate',
#        title='Median Completeness Simulations for SPT and SDWFS')
# ax.legend(loc='best')
# fig.savefig('Data/Comp_Sim/Plots/SPT_SDWFS_Comp_Sim_Comparison.pdf', format='pdf')


# Create a two pane figure with the ch2 full SPT completeness plot and the SPT/SDWFS comparison plot.
fig, (ax, bx) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 4))

# Channel 2 curve
ax.tick_params(direction='in', which='both', right=True)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
for curve in ch2_rates:
    ax.plot(ch2_mag_bins, ch2_rates[curve], 'k-', alpha=0.4)
ax.plot(ch2_mag_bins, ch2_med_rates, 'r-', alpha=1.0, linewidth=2)
ax.set(xlim=[10.0, 23.0], ylim=[0.0, 1.0], xlabel='Vega Magnitude', ylabel='Recovery Rate')

x_top_tick_labels = [15, 20, 25]
x_top_tick_pos = [round(ab + -3.255, 2) for ab in x_top_tick_labels]
ax1 = ax.twiny()
ax1.tick_params(direction='in', which='both')
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.set(xlim=[10.0, 23.0], xlabel='AB magnitude')
ax1.set_xticks(x_top_tick_pos)
ax1.set_xticklabels(['{:g}'.format(ab) for ab in x_top_tick_labels])

# SPT/SDWFS comparison plot
bx.tick_params(direction='in', which='both', right=True, labelright=True)
bx.xaxis.set_minor_locator(AutoMinorLocator(5))
bx.yaxis.set_minor_locator(AutoMinorLocator(2))
bx.plot(ch1_mag_bins, ch1_med_rates, 'b-', label='SPT Ch. 1')
bx.plot(ch1_mag_bins, sdwfs_ch1_med_rates, 'b--', label='SDWFS Ch. 1')
bx.plot(ch2_mag_bins, ch2_med_rates, 'r-', label='SPT Ch. 2')
bx.plot(ch2_mag_bins, sdwfs_ch2_med_rates, 'r--', label='SDWFS Ch. 2')
bx.set(xlim=[10.0, 23.0], ylim=[0.0, 1.0], xlabel='Vega Magnitude')
bx.legend(loc='best', frameon=False)

plt.subplots_adjust(wspace=0)

fig.savefig('Data/Comp_Sim/Plots/SPT_Comp_Sim_Paper_Plot.pdf')
