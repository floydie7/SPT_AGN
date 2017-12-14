"""
SPT_AGN_Radial_Plot.py
Author: Benjamin Floyd

Creates the radial cluster AGN plot using the results calculated from `SPT_AGN_Prelim_Sci_Radial_Postprocessing.py`.
"""

from __future__ import print_function, division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Read in the results file
results = np.load('Data/Radial_bin_data.npy').item()

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

# Make the radial distance plot
fig, ax = plt.subplots()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.errorbar(rad_bin_cent-0.09, mid_low_z_rad_surf_den, yerr=mid_low_z_rad_err[::-1], fmt='o', c='C0',
            label='$0.5 < z \leq 0.65$')
ax.errorbar(rad_bin_cent-0.03, mid_mid_z_rad_surf_den, yerr=mid_mid_z_rad_err[::-1], fmt='o', c='C1',
            label='$0.65 < z \leq 0.75$')
ax.errorbar(rad_bin_cent+0.03, mid_high_z_rad_surf_den, yerr=mid_high_z_rad_err[::-1], fmt='o', c='C2',
            label='$0.75 < z \leq 1.0$')
ax.errorbar(rad_bin_cent+0.09, high_z_rad_surf_den, yerr=high_z_rad_err[::-1], fmt='o', c='C3', label='$z > 1.0$')
ax.axhline(y=0.0, c='k', linestyle='--', label='SDWFS Field Density')
ax.set(title='239 SPT Clusters', xlabel='r/r$_{500}$', ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]',
       xlim=[0, 2.5], ylim=[-3, 2])
ax.legend()
fig.savefig('Data/Plots/SPT_AGN_Radial_Distance_Sci_Plot.pdf', format='pdf')


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