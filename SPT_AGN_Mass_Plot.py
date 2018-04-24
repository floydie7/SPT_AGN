"""
SPT_AGN_Mass_Plot.py
Author: Benjamin Floyd

Creates the mass cluster AGN plot using the results calculated from `SPT_AGN_Prelim_Sci_Mass_Binned_Analysis.py`.
"""

from __future__ import print_function, division

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

os.environ['PATH'] += os.pathsep + '/Library/TeX/texbin/'

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{booktabs}"

# Read in the radial files
# results05 = np.load('Data/Binned_Analysis/Mass_0.5r500_bin_data.npy', encoding='latin1').item()
# results10 = np.load('Data/Binned_Analysis/Mass_1.0r500_bin_data.npy', encoding='latin1').item()
results15 = np.load('Data/Binned_Analysis/Mass_1.5r500_bin_data.npy', encoding='latin1').item()
# results20 = np.load('Data/Mass_2.0r500_bin_data.npy').item()

mass_chisq_pte_table = r'''\begin{tabular}{rcc} \multicolumn{2}{c}{Model: No Environmental Effect} \\ \toprule & $\chi_\nu^2$ & PTE \\ \midrule $0.5 < z \leq 0.65$ & 3.50 & 0.062\\ $0.65< z \leq 0.75$ & 0.64 & 0.430 \\ $0.75 < z \leq 1$ &  4.41 & 0.036 \\ $z > 1$ &  12.33 & $4.5\times 10^{-4}$\\ All redshifts & 13.98 & $1.9\times 10^{-4}$ \end{tabular}'''

# Extract all values from the files
# Mass bin centers are the same across all files.
mass_bin_cent = results15['mass_bin_cent']

# # 0.5*r500 enclosing radius.
# mid_low_surf_den_05 = results05['mid_low_z_mass_surf_den']
# mid_low_surf_den_err_05 = results05['mid_low_z_mass_surf_den_err']
# mid_mid_surf_den_05 = results05['mid_mid_z_mass_surf_den']
# mid_mid_surf_den_err_05 = results05['mid_mid_z_mass_surf_den_err']
# mid_high_surf_den_05 = results05['mid_high_z_mass_surf_den']
# mid_high_surf_den_err_05 = results05['mid_high_z_mass_surf_den_err']
# high_surf_den_05 = results05['high_z_mass_surf_den']
# high_surf_den_err_05 = results05['high_z_mass_surf_den_err']
# all_surf_den_05 = results05['all_z_mass_surf_den']
# all_surf_den_err_05 = results05['all_z_mass_surf_den_err']
#
# # 1.0*r500 enclosing radius.
# mid_low_surf_den_10 = results10['mid_low_z_mass_surf_den']
# mid_low_surf_den_err_10 = results10['mid_low_z_mass_surf_den_err']
# mid_mid_surf_den_10 = results10['mid_mid_z_mass_surf_den']
# mid_mid_surf_den_err_10 = results10['mid_mid_z_mass_surf_den_err']
# mid_high_surf_den_10 = results10['mid_high_z_mass_surf_den']
# mid_high_surf_den_err_10 = results10['mid_high_z_mass_surf_den_err']
# high_surf_den_10 = results10['high_z_mass_surf_den']
# high_surf_den_err_10 = results10['high_z_mass_surf_den_err']
# all_surf_den_10 = results10['all_z_mass_surf_den']
# all_surf_den_err_10 = results10['all_z_mass_surf_den_err']

# 1.5*r500 enclosing radius.
mid_low_surf_den_15 = results15['mid_low_z_mass_surf_den']
mid_low_surf_den_err_15 = results15['mid_low_z_mass_surf_den_err']
mid_mid_surf_den_15 = results15['mid_mid_z_mass_surf_den']
mid_mid_surf_den_err_15 = results15['mid_mid_z_mass_surf_den_err']
mid_high_surf_den_15 = results15['mid_high_z_mass_surf_den']
mid_high_surf_den_err_15 = results15['mid_high_z_mass_surf_den_err']
high_surf_den_15 = results15['high_z_mass_surf_den']
high_surf_den_err_15 = results15['high_z_mass_surf_den_err']
all_surf_den_15 = results15['all_z_mass_surf_den']
all_surf_den_err_15 = results15['all_z_mass_surf_den_err']

# 2.0*r500 enclosing radius.
# mid_low_surf_den_20 = results20['mid_low_z_mass_surf_den']
# mid_low_surf_den_err_20 = results20['mid_low_z_mass_surf_den_err']
# mid_mid_surf_den_20 = results20['mid_mid_z_mass_surf_den']
# mid_mid_surf_den_err_20 = results20['mid_mid_z_mass_surf_den_err']
# mid_high_surf_den_20 = results20['mid_high_z_mass_surf_den']
# mid_high_surf_den_err_20 = results20['mid_high_z_mass_surf_den_err']
# high_surf_den_20 = results20['high_z_mass_surf_den']
# high_surf_den_err_20 = results20['high_z_mass_surf_den_err']
# all_surf_den_20 = results20['all_z_mass_surf_den']
# all_surf_den_err_20 = results20['all_z_mass_surf_den_err']

# # Make the mass plots
# fig, ax = plt.subplots()
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(5))
# ax.errorbar(mass_bin_cent-0.02, mid_low_surf_den_05, yerr=mid_low_surf_den_err_05[::-1], fmt='o', c='C0',
#             label='$0.5 < z \leq 0.65$')
# ax.errorbar(mass_bin_cent-0.01, mid_mid_surf_den_05, yerr=mid_mid_surf_den_err_05[::-1], fmt='o', c='C1',
#             label='$0.65 < z \leq 0.75$')
# ax.errorbar(mass_bin_cent+0.01, mid_high_surf_den_05, yerr=mid_high_surf_den_err_05[::-1], fmt='o', c='C2',
#             label='$0.75 < z \leq 1.0$')
# ax.errorbar(mass_bin_cent+0.02, high_surf_den_05, yerr=high_surf_den_err_05[::-1], fmt='o', c='C3', label='$z > 1.0$')
# ax.errorbar(mass_bin_cent, all_surf_den_05, yerr=all_surf_den_err_05[::-1], fmt='o', c='C4', label='All Redshifts')
# ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.set(title='SPT Cluster AGN within $0.5 r_{500}$', xlabel='$\log M_{500} (M_\odot)$',
#        ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]', ylim=[-2, 3])
# ax.legend()
# #fig.savefig('Data/Plots/SPT_AGN_Mass_Sci_Plot_05r500.pdf', format='pdf')
#
# fig, ax = plt.subplots()
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(5))
# ax.errorbar(mass_bin_cent-0.02, mid_low_surf_den_10, yerr=mid_low_surf_den_err_10[::-1], fmt='o', c='C0',
#             label='$0.5 < z \leq 0.65$')
# ax.errorbar(mass_bin_cent-0.01, mid_mid_surf_den_10, yerr=mid_mid_surf_den_err_10[::-1], fmt='o', c='C1',
#             label='$0.65 < z \leq 0.75$')
# ax.errorbar(mass_bin_cent+0.01, mid_high_surf_den_10, yerr=mid_high_surf_den_err_10[::-1], fmt='o', c='C2',
#             label='$0.75 < z \leq 1.0$')
# ax.errorbar(mass_bin_cent+0.02, high_surf_den_10, yerr=high_surf_den_err_10[::-1], fmt='o', c='C3', label='$z > 1.0$')
# ax.errorbar(mass_bin_cent, all_surf_den_10, yerr=all_surf_den_err_10[::-1], fmt='o', c='C4', label='All Redshifts')
# ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.set(title='SPT Cluster AGN within $1.0 r_{500}$', xlabel='$\log M_{500} (M_\odot)$',
#        ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]', ylim=[-2, 3])
# ax.legend()
# #fig.savefig('Data/Plots/SPT_AGN_Mass_Sci_Plot_10r500.pdf', format='pdf')

fig, ax = plt.subplots()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.errorbar(mass_bin_cent-0.02, mid_low_surf_den_15, yerr=mid_low_surf_den_err_15[::-1], fmt='o', c='C0',
            label='$0.5 < z \leq 0.65$')
ax.errorbar(mass_bin_cent-0.01, mid_mid_surf_den_15, yerr=mid_mid_surf_den_err_15[::-1], fmt='o', c='C1',
            label='$0.65 < z \leq 0.75$')
ax.errorbar(mass_bin_cent+0.01, mid_high_surf_den_15, yerr=mid_high_surf_den_err_15[::-1], fmt='o', c='C2',
            label='$0.75 < z \leq 1.0$')
ax.errorbar(mass_bin_cent+0.02, high_surf_den_15, yerr=high_surf_den_err_15[::-1], fmt='o', c='C3', label='$z > 1.0$')
ax.errorbar(mass_bin_cent, all_surf_den_15, yerr=all_surf_den_err_15[::-1], fmt='o', c='C4', label='All Redshifts')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
ax.set(title='SPT Cluster AGN within $1.5 r_{500}$', xlabel='$\log M_{500} (M_\odot)$',
       ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]', ylim=[-3, 3])
ax.legend(loc='lower left', frameon=False)

matplotlib.rcParams['text.usetex'] = True
ax.text(14.6, -2.9, mass_chisq_pte_table)
plt.show()
# fig.savefig('Data/Binned_Analysis/Plots/SPT_AGN_Mass_Sci_Plot_15r500_with_table.pdf', format='pdf')

# fig, ax = plt.subplots()
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(5))
# ax.errorbar(mass_bin_cent-0.012, mid_low_surf_den_20, yerr=mid_low_surf_den_err_20[::-1], fmt='o', c='C0',
#             label='$0.5 < z \leq 0.65$')
# ax.errorbar(mass_bin_cent-0.004, mid_mid_surf_den_20, yerr=mid_mid_surf_den_err_20[::-1], fmt='o', c='C1',
#             label='$0.65 < z \leq 0.75$')
# ax.errorbar(mass_bin_cent+0.004, mid_high_surf_den_20, yerr=mid_high_surf_den_err_20[::-1], fmt='o', c='C2',
#             label='$0.75 < z \leq 1.0$')
# ax.errorbar(mass_bin_cent+0.012, high_surf_den_20, yerr=high_surf_den_err_20[::-1], fmt='o', c='C3', label='$z > 1.0$')
# ax.errorbar(mass_bin_cent, all_surf_den_20, yerr=all_surf_den_err_20[::-1], fmt='o', c='C4', label='All Redshifts')
# ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.set(title='SPT Cluster AGN within $2.0 r_{500}$', xlabel='$\log M_{500} (M_\odot)$',
#        ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]', ylim=[-2, 3])
# ax.legend()
# fig.savefig('Data/Plots/SPT_AGN_Mass_Sci_Plot_20r500.pdf', format='pdf')

# fig, ax = plt.subplots()
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(5))
# ax.errorbar(mass_bin_cent-0.01, all_surf_den_05, yerr=all_surf_den_err_05[::-1], fmt='o', c='C0', label='$0.5 r_{500}$')
# ax.errorbar(mass_bin_cent, all_surf_den_10, yerr=all_surf_den_err_10[::-1], fmt='o', c='C1', label='$1.0 r_{500}$')
# ax.errorbar(mass_bin_cent+0.01, all_surf_den_15, yerr=all_surf_den_err_15[::-1], fmt='o', c='C2', label='$1.5 r_{500}$')
# ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.set(title='SPT Clusters at all Redshifts', xlabel='$\log M_{500} [M_\odot]$',
#        ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]', ylim=[-2, 3])
# ax.legend()
#fig.savefig('Data/Plots/SPT_AGN_Mass_Sci_Plot_all_redshifts.pdf', format='pdf')
