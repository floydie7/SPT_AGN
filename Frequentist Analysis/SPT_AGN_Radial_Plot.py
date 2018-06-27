"""
SPT_AGN_Radial_Plot.py
Author: Benjamin Floyd

Creates the radial cluster AGN plot using the results calculated from `SPT_AGN_Prelim_Sci_Radial_Postprocessing.py`.
"""

from __future__ import print_function, division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

os.environ['PATH'] += os.pathsep + '/Library/TeX/texbin/'

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{booktabs}"

# Read in the radial file
results = np.load('Data/Binned_Analysis/Radial_bin_data_cumulative.npy', encoding='latin1').item()

# Read in the reduced chi-sq and pte file computed by `SPT_AGN_Binned_ChiSq.py`
rad_chisq_pte = np.load('Data/Binned_Analysis/Radial_bin_chisq_pte.npy')
rad_chisq_pte_txt = np.array([['{:.2f}'.format(rad_chisq_pte[i][0]), '{:.2e}'.format(rad_chisq_pte[i][1])]
                              for i in range(len(rad_chisq_pte))])
rad_chisq_pte_table = r'''\begin{tabular}{rcc} \multicolumn{2}{c}{Model: No Environmental Effect} \\ \toprule & $\chi_\nu^2$ & PTE \\ \midrule $0.5 < z \leq 0.65$ & 8.54 & $3.6\times 10^{-5}$\\ $0.65< z \leq 0.75$ & 1.76 & $6.0\times 10^{-2}$\\ $0.75 < z \leq 1$ & 10.64 & $4.0\times 10^{-6}$\\ $z > 1$ & 38.50 & $1.7\times 10^{-18}$\\ All redshifts & 53.00 & $7.4\times 10^{-25}$ \end{tabular}'''

# Extract all the values from the file.
rad_bin_cent = results['radial_bins']
mid_low_z_rad_surf_den = results['mid_low_z_rad_surf_den']
mid_low_z_rad_err = results['mid_low_z_rad_err']
mid_mid_z_rad_surf_den = results['mid_mid_z_rad_surf_den']
mid_mid_z_rad_err = results['mid_mid_z_rad_err']
mid_high_z_rad_surf_den = results['mid_high_z_rad_surf_den']
mid_high_z_rad_err = results['mid_high_z_rad_err']
high_z_rad_surf_den = results['high_z_rad_surf_den']
high_z_rad_err = results['high_z_rad_err']
all_z_rad_surf_den = results['all_z_rad_surf_den']
all_z_rad_err = results['all_z_rad_err']

# Make the radial distance plot
fig, ax = plt.subplots()
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.errorbar(rad_bin_cent-0.09, mid_low_z_rad_surf_den, yerr=mid_low_z_rad_err[::-1], fmt='o', c='C0',
            label='$0.5 < z \leq 0.65$')
ax.errorbar(rad_bin_cent-0.03, mid_mid_z_rad_surf_den, yerr=mid_mid_z_rad_err[::-1], fmt='^', c='C1',
            label='$0.65 < z \leq 0.75$')
ax.errorbar(rad_bin_cent+0.03, mid_high_z_rad_surf_den, yerr=mid_high_z_rad_err[::-1], fmt='s', c='C2',
            label='$0.75 < z \leq 1.0$')
ax.errorbar(rad_bin_cent+0.09, high_z_rad_surf_den, yerr=high_z_rad_err[::-1], fmt='v', c='C3', label='$z > 1.0$')
ax.errorbar(rad_bin_cent, all_z_rad_surf_den, yerr=all_z_rad_err[::-1], fmt='D', c='C4', label='All redshifts')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
ax.set(xlabel='r/r$_{500}$',
       ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]',
       xlim=[0, 1.5], ylim=[-3, 2])
# ax.table(cellText=rad_chisq_pte_txt, colLabels=[r'$\chi^2_\nu$', 'PTE'], loc='lower right').auto_set_column_width((0,1))
ax.legend(loc='lower left', frameon=False)

matplotlib.rcParams['text.usetex'] = True
ax.text(0.6,-2.9, rad_chisq_pte_table)
fig.savefig('Data/Binned_Analysis/Plots/SPT_AGN_Radial_Distance_Sci_Plot_with_table_paper.pdf', format='pdf')


# # Make individual radial plots for each redshift bin.
# # Calculate the mean redshifts for each bin.
# mid_low_mean_z = mid_low_z_bin['REDSHIFT'].mean()
# mid_mid_mean_z = mid_mid_z_bin['REDSHIFT'].mean()
# mid_high_mean_z = mid_high_z_bin['REDSHIFT'].mean()
# high_mean_z = mid_low_z_bin['REDSHIFT'].mean()
#
# # Scale the field error to the mean redshifts
# mid_low_field_err = field_surf_den_err.value * (cosmo.kpc_proper_per_arcmin(mid_low_mean_z).to(u.Mpc / u.arcmin))**2
# mid_mid_field_err = field_surf_den_err.value * (cosmo.kpc_proper_per_arcmin(mid_mid_mean_z).to(u.Mpc / u.arcmin))**2
# mid_high_field_err = field_surf_den_err.value * (cosmo.kpc_proper_per_arcmin(mid_high_mean_z).to(u.Mpc / u.arcmin))**2
# high_field_err = field_surf_den_err.value * (cosmo.kpc_proper_per_arcmin(high_mean_z).to(u.Mpc / u.arcmin))**2
#
#
# fig = plt.figure(figsize=(11, 8.5), tight_layout=False)
# fig.suptitle('Overdensity of SPT AGN', va='center')
# fig.text(0.5, 0.04, 'r/r$_{500}$', ha='center', va='top')
# ax = fig.add_subplot(221)
# ax.tick_params(bottom='off', labelbottom='off')
#
#
# ax = fig.add_subplot(221, sharey=ax, sharex=ax)
# ax.errorbar(rad_bin_cent, mid_low_z_rad_surf_den, yerr=mid_low_z_rad_err[::-1], fmt='o', c='C0', label='$0.5 < z \leq 0.65$')
# ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.axhspan(ymax=mid_low_field_err.value, ymin=0.0-mid_low_field_err.value, color='0.5', alpha=0.2)
# ax.set(ylim=[-3, 2], ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]')
# ax.legend()
#
# ax = fig.add_subplot(222, sharey=ax, sharex=ax)
# ax.errorbar(rad_bin_cent, mid_mid_z_rad_surf_den, yerr=mid_mid_z_rad_err[::-1], fmt='o', c='C1', label='$0.65 < z \leq 0.75$')
# ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.axhspan(ymax=mid_mid_field_err.value, ymin=0.0-mid_mid_field_err.value, color='0.5', alpha=0.2)
# ax.set(ylim=[-3, 2])
# ax.legend()
#
# ax = fig.add_subplot(223, sharey=ax, sharex=ax)
# ax.errorbar(rad_bin_cent, mid_high_z_rad_surf_den, yerr=mid_high_z_rad_err[::-1], fmt='o', c='C2', label='$0.75 < z \leq 1.0$')
# ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.axhspan(ymax=mid_high_field_err.value, ymin=0.0-mid_high_field_err.value, color='0.5', alpha=0.2)
# ax.set(ylim=[-3, 2], ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]')
# ax.legend()
#
# ax = fig.add_subplot(224, sharey=ax, sharex=ax)
# ax.errorbar(rad_bin_cent, high_z_rad_surf_den, yerr=high_z_rad_err[::-1], fmt='o', c='C3', label='$z > 1.0$')
# ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
# ax.axhspan(ymax=high_field_err.value, ymin=0.0-high_field_err.value, color='0.5', alpha=0.2)
# ax.set(ylim=[-3, 2])
# ax.legend()
#
# fig.savefig('Data/Plots/SPT_AGN_Radial_Distance_Sci_Plot_Field_Errors.pdf', format='pdf')