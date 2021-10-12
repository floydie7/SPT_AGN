"""
SDWFS_Stern_wedge_plot.py
Author: Benjamin Floyd

Plots the IRAC color-color plane with the Stern Wedge shown to illustrate our choice in [3.6] - [4.5] color threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# Read in the SDWFS catalog
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

SDWFS_main = Table.read(
    'Data_Repository/Catalogs/Bootes/SDWFS/ch2v33_sdwfs_2009mar3_apcorr_matched_ap4_Main_v0.4.cat.gz',
    names=['ID', 'IRAC_RA', 'IRAC_DEC', 'B_APFLUX4', 'R_APFLUX4', 'I_APFLUX4', 'B_APFLUXERR4',
           'R_APFLUXERR4', 'I_APFLUXERR4', 'B_APMAG4', 'R_APMAG4', 'I_APMAG4', 'B_APMAGERR4',
           'R_APMAGERR4', 'I_APMAGERR4', 'CH1_APFLUX4', 'CH2_APFLUX4', 'CH3_APFLUX4', 'CH4_APFLUX4',
           'CH1_APFLUXERR4', 'CH2_APFLUXERR4', 'CH3_APFLUXERR4', 'CH4_APFLUXERR4',
           'CH1_APFLUXERR4_BROWN', 'CH2_APFLUXERR4_BROWN', 'CH3_APFLUXERR4_BROWN',
           'CH4_APFLUXERR4_BROWN', 'CH1_APMAG4', 'CH2_APMAG4', 'CH3_APMAG4', 'CH4_APMAG4',
           'CH1_APMAGERR4', 'CH2_APMAGERR4', 'CH3_APMAGERR4', 'CH4_APMAGERR4',
           'CH1_APMAGERR4_BROWN', 'CH2_APMAGERR4_BROWN', 'CH3_APMAGERR4_BROWN',
           'CH4_APMAGERR4_BROWN', 'STARS_COLOR', 'STARS_MORPH', 'CLASS_STAR', 'MBZ_FLAG_4_4_4'],
    format='ascii')

# Make the appropriate magnitude cuts
SDWFS_main = SDWFS_main[(10. < SDWFS_main['CH1_APMAG4']) &
                        (10.45 < SDWFS_main['CH2_APMAG4']) & (SDWFS_main['CH2_APMAG4'] <= 17.46)]

# Create colors for the full catalog
SDWFS_main_Ch12 = SDWFS_main['CH1_APMAG4'] - SDWFS_main['CH2_APMAG4']
SDWFS_main_Ch34 = SDWFS_main['CH3_APMAG4'] - SDWFS_main['CH4_APMAG4']

# Select for Stern wedge AGN following the Stern+05 criteria
Stern_AGN = SDWFS_main[(SDWFS_main_Ch34 > 0.6) &
                       (SDWFS_main_Ch12 > 0.2 * SDWFS_main_Ch34 + 0.18) &
                       (SDWFS_main_Ch12 > 2.5 * SDWFS_main_Ch34 - 3.5)]

# Create colors for the Stern AGN
Stern_AGN_Ch12 = Stern_AGN['CH1_APMAG4'] - Stern_AGN['CH2_APMAG4']
Stern_AGN_Ch34 = Stern_AGN['CH3_APMAG4'] - Stern_AGN['CH4_APMAG4']

# Vertices of the Stern Wedge
stern_verts = np.array([[0.6, 1.5], [0.6, 0.3], [1.6, 0.5], [2.0, 1.5]])

#%% Create plot
fig, ax = plt.subplots()
ax.hexbin(SDWFS_main_Ch34, SDWFS_main_Ch12, gridsize=75, extent=(-0.4, 3.4, -0.2, 1.5), cmap='Blues', bins='log', mincnt=1)
ax.hexbin(Stern_AGN_Ch34, Stern_AGN_Ch12, gridsize=75, extent=(-0.4, 3.4, -0.2, 1.5), cmap='Reds', bins='log', mincnt=1)
ax.add_artist(Polygon(stern_verts, linewidth=2, closed=False, fill=False))
ax.axhline(y=0.7, color='k', linestyle='--')
ax.set(xlabel='[5.8] - [8.0] (Vega)', ylabel='[3.6] - [4.5] (Vega)', xlim=[-0.4, 3.4], ylim=[-0.1, 1.5])
min_ch12_color, max_ch12_color = ax.get_ylim()
min_ch34_color, max_ch34_color = ax.get_xlim()
ax_ab1, ax_ab2 = ax.twinx(), ax.twiny()
ax_ab1.set(ylabel='[3.6] - [4.5] (AB)', ylim=[min_ch12_color + 2.788 - 3.255, max_ch12_color + 2.788 - 3.255])
ax_ab2.set(xlabel='[5.8] - [8.0] (AB)', xlim=[min_ch34_color + 3.743 - 4.372, max_ch34_color + 3.743 - 4.372])
handles, labels = ax.get_legend_handles_labels()
handles.extend([Line2D([0], [0], marker='h', color='none', markerfacecolor='lightcoral', markeredgecolor='firebrick',
                       markersize=10),
                Line2D([0], [0], marker='h', color='none', markerfacecolor='lightblue', markeredgecolor='steelblue',
                       markersize=10)])
labels.extend(['Stern Wedge AGN', 'Non-Active Galaxies'])
plt.legend(handles, labels, frameon=False)
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/SDWFS_Stern_Wedge.pdf')
