"""
SPT_AGN_Mass_Plot.py
Author: Benjamin Floyd

Creates the mass cluster AGN plot using the results calculated from `SPT_AGN_Prelim_Sci_Mass_Binned_Analysis.py`.
"""

from __future__ import print_function, division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Read in the results files
results05 = np.load('Data/Mass_0.5r500_bin_data.npy').item()
results10 = np.load('Data/Mass_1.0r500_bin_data.npy').item()
results15 = np.load('Data/Mass_1.5r500_bin_data.npy').item()
results20 = np.load('Data/Mass_2.0r500_bin_data.npy').item()

# Extract all values from the files
mass_bin_cent = results05['mass_bin_cent']

mid_low_surf_den_05 = results05['mid_low_z_mass_surf_den']
mid_low_surf_den_err_05 = results05['mid_low_z_mass_surf_den_err']
mid_mid_surf_den_05 = results05['mid_mid_z_mass_surf_den']
mid_mid_surf_den_err_05 = results05['mid_mid_z_mass_surf_den_err']
mid_high_surf_den_05 = results05['mid_high_z_mass_surf_den']
mid_high_surf_den_err_05 = results05['mid_high_z_mass_surf_den_err']
high_surf_den_05 = results05['high_z_mass_surf_den']
high_surf_den_err_05 = results05['high_z_mass_surf_den_err']

# Make the radial distance plot
fig, ax = plt.subplots()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.errorbar(mass_bin_cent-0.09, mid_low_surf_den_05, yerr=mid_low_surf_den_err_05[::-1], fmt='o', c='C0',
            label='$0.5 < z \leq 0.65$')
ax.errorbar(mass_bin_cent-0.03, mid_mid_surf_den_05, yerr=mid_mid_surf_den_err_05[::-1], fmt='o', c='C1',
            label='$0.65 < z \leq 0.75$')
ax.errorbar(mass_bin_cent+0.03, mid_high_surf_den_05, yerr=mid_high_surf_den_err_05[::-1], fmt='o', c='C2',
            label='$0.75 < z \leq 1.0$')
ax.errorbar(mass_bin_cent+0.09, high_surf_den_05, yerr=high_surf_den_err_05[::-1], fmt='o', c='C3', label='$z > 1.0$')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
ax.set(title='239 SPT Clusters', xlabel='$\log M_{500} (M_\odot)$', ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]')
ax.legend()
