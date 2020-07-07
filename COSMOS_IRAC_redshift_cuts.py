"""
COSMOS_IRAC_redshift_cuts.py
Author: Benjamin Floyd

Uses the COSMOS catalog to examine our IR-bright AGN selection cuts as a function of redshift to determine if we have
significant contamination in our sample.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.lines import Line2D

# Read in th COSMOS catalog
cosmos = Table.read('Data/COSMOS2015_Laigle+_v1.1.fits.gz')

#%%
# Remove non-detections
cosmos = cosmos[(cosmos['SPLASH_1_MAG'] > -99) &
                (cosmos['SPLASH_2_MAG'] > -99) &
                (cosmos['SPLASH_3_MAG'] > -99) &
                (cosmos['SPLASH_4_MAG'] > -99)]
cosmos = cosmos[(cosmos['PHOTOZ'] >= 0) | (cosmos['ZQ'] >= 0)]

# Convert the IRAC magnitudes from AB to Vega (because I don't want to have to convert the selections to AB).
cosmos['SPLASH_1_MAG'] += -2.788
cosmos['SPLASH_2_MAG'] += -3.255
cosmos['SPLASH_3_MAG'] += -3.743
cosmos['SPLASH_4_MAG'] += -4.372

# Apply our 4.5 um magnitude cuts as we would in SDWFS/SPT
cosmos = cosmos[(10.45 < cosmos['SPLASH_2_MAG']) & (cosmos['SPLASH_2_MAG'] < 17.46)]

# Apply Stern Wedge selection
stern_agn = cosmos[(cosmos['SPLASH_3_MAG'] - cosmos['SPLASH_4_MAG'] > 0.6) &
                   (cosmos['SPLASH_1_MAG'] - cosmos['SPLASH_2_MAG'] >
                    0.2 * (cosmos['SPLASH_3_MAG'] - cosmos['SPLASH_4_MAG']) + 0.18) &
                   (cosmos['SPLASH_1_MAG'] - cosmos['SPLASH_2_MAG'] >
                    2.5 * (cosmos['SPLASH_3_MAG'] - cosmos['SPLASH_4_MAG']) - 3.5)]

# Separate the non-AGN galaxies from the AGN
stern_complement_ids = list(set(cosmos['NUMBER']) - set(stern_agn['NUMBER']))
non_agn = cosmos[np.in1d(cosmos['NUMBER'], stern_complement_ids)]

#%%
non_agn_0_2 = non_agn[non_agn['PHOTOZ'] <= 4]
stern_agn_0_2 = stern_agn[stern_agn['ZQ'] <= 4]

# [3.6] - [4.5] color
agn_color = stern_agn_0_2['SPLASH_1_MAG'] - stern_agn_0_2['SPLASH_2_MAG']
non_agn_color = non_agn_0_2['SPLASH_1_MAG'] - non_agn_0_2['SPLASH_2_MAG']


#%%
fig = plt.figure()
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(8, 3), wspace=0)

# Main plot
ax = fig.add_subplot(gs[0])
ax.hexbin(non_agn_0_2['PHOTOZ'], non_agn_color, gridsize=(2500, 100), bins='log', mincnt=1, cmap='Blues')
ax.hexbin(stern_agn_0_2['ZQ'], agn_color, gridsize=(50, 75), bins='log', mincnt=1, cmap='Reds', alpha=0.5)
ax.set(title='COSMOS', xlabel='Photometric Redshift', ylabel='[3.6] - [4.5] (Vega)', xlim=[0, 4], ylim=[0., 1.5])

ax.axhline(y=0.7, color='k', linewidth=2)
# ax.arrow(0.25, 0.7, 0, 0.2, color='k', width=0.004, head_width=6 * 0.004, length_includes_head=True)

handles, labels = ax.get_legend_handles_labels()
handles.extend([Line2D([0], [0], marker='h', color='w', markerfacecolor='lightcoral', markeredgecolor='firebrick',
                       markersize=10),
                Line2D([0], [0], marker='h', color='w', markerfacecolor='lightblue', markeredgecolor='steelblue',
                       markersize=10)
                ])
labels.extend(['Stern Wedge AGN', 'Non-Active Galaxies'])
ax.legend(handles, labels, frameon=False)

# Side histogram
ax_hist = fig.add_subplot(gs[1], sharey=ax)
ax_hist.tick_params(axis='y', labelleft=False)
min_z, max_z = ax.get_xlim()
binwidth = 0.05
bins_nonAGN = np.arange(np.min(non_agn_color), np.max(non_agn_color) + binwidth, binwidth)
bins_AGN = np.arange(np.min(agn_color), np.max(agn_color) + binwidth, binwidth)
ax_hist.hist(agn_color[(stern_agn_0_2['ZQ'] > min_z) & (stern_agn_0_2['ZQ'] < max_z)], bins=bins_AGN,
             orientation='horizontal', color='lightcoral', alpha=0.6)
ax_hist.hist(non_agn_color[(non_agn_0_2['PHOTOZ'] > min_z) & (non_agn_0_2['PHOTOZ'] < max_z)], bins=bins_nonAGN,
             orientation='horizontal', color='lightblue', alpha=0.6)
ax_hist.axhline(y=0.7, color='k', linewidth=2)
ax_hist.set(xlabel=r'$N$', xscale='log')

# fig.savefig('Data/Plots/COSMOS_IRAC_Redshift_z4.pdf')
plt.show()
