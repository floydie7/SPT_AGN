"""
combined_radial_mass_binned_plot.py
Author: Benjamin Floyd

This script recreates the frequentist plots from very early analysis that show that the AGN in clusters are not
environment invariant.
"""

import numpy as np
import matplotlib.pyplot as plt

top_dir = '/home/ben-work/PycharmProjects/SPT_AGN/'

# Read in the old binned datasets (NB: Even though the radial file is named "cumulative," as far as I can tell, it is
# actually differential. It's likely an old typo when I ran it last and never fixed it.)
mass_data = np.load(f'{top_dir}Data_Repository/Project_Data/SPT-IRAGN/Binned_Analysis/Mass_1.5r500_bin_data.npy',
                    allow_pickle=True, encoding='latin1').item()
radial_data = np.load(f'{top_dir}Data_Repository/Project_Data/SPT-IRAGN/Binned_Analysis/Radial_bin_data_cumulative.npy',
                      allow_pickle=True, encoding='latin1').item()

# Create the plots
fig, (mass_ax, radial_ax) = plt.subplots(ncols=2, sharey='row', figsize=(8.5, 4.8))
# Plot the mass trend for each redshift bin
mass_ax.errorbar(mass_data['mass_bin_cent']-0.03, mass_data['mid_low_z_mass_surf_den'], fmt='o',
                 yerr=mass_data['mid_low_z_mass_surf_den_err'][::-1], label=r'$0.5 < z \leq 0.65$')
mass_ax.errorbar(mass_data['mass_bin_cent']-0.015, mass_data['mid_mid_z_mass_surf_den'], fmt='o',
                 yerr=mass_data['mid_mid_z_mass_surf_den_err'][::-1], label=r'$0.65 < z \leq 0.75$')
mass_ax.errorbar(mass_data['mass_bin_cent']+0.015, mass_data['mid_high_z_mass_surf_den'], fmt='o',
                 yerr=mass_data['mid_high_z_mass_surf_den_err'][::-1], label=r'$0.75 < z \leq 1.0$')
mass_ax.errorbar(mass_data['mass_bin_cent']+0.03, mass_data['high_z_mass_surf_den'], fmt='o',
                 yerr=mass_data['high_z_mass_surf_den_err'][::-1], label=r'$z > 1.0$')
# Plot the mass trend over all redshifts
mass_ax.errorbar(mass_data['mass_bin_cent'], mass_data['all_z_mass_surf_den'], fmt='o',
                 yerr=mass_data['all_z_mass_surf_den_err'][::-1], label='All Redshifts')

# Plot the radial trend for each redshift bin
radial_ax.errorbar(radial_data['radial_bins']-0.09, radial_data['mid_low_z_rad_surf_den'], fmt='o',
                   yerr=radial_data['mid_low_z_rad_err'][::-1], label=r'$0.5 < z \leq 0.65$')
radial_ax.errorbar(radial_data['radial_bins']-0.03, radial_data['mid_mid_z_rad_surf_den'], fmt='o',
                   yerr=radial_data['mid_mid_z_rad_err'][::-1], label=r'$0.65 < z \leq 0.75$')
radial_ax.errorbar(radial_data['radial_bins']+0.03, radial_data['mid_high_z_rad_surf_den'], fmt='o',
                   yerr=radial_data['mid_high_z_rad_err'][::-1], label=r'$0.75 < z \leq 1.0$')
radial_ax.errorbar(radial_data['radial_bins']+0.09, radial_data['high_z_rad_surf_den'], fmt='o',
                   yerr=radial_data['high_z_rad_err'][::-1], label=r'$z > 1.0$')
# Plot the radial trend over all redshifts
radial_ax.errorbar(radial_data['radial_bins'], radial_data['all_z_rad_surf_den'], fmt='o',
                   yerr=radial_data['all_z_rad_err'][::-1], label='All Redshifts')

# Plot the SDWFS field surface density on both plots
mass_ax.axhline(0., c='k', linestyle='--')
radial_ax.axhline(0., c='k', linestyle='--')

# Set axes labels
mass_ax.set(xlabel=r'$\log M_{500} [M_\odot]$', ylabel=r'$\Sigma_{\rm AGN}$ per cluster [Mpc$^{-2}$]', ylim=[-1.5, 4.5])
radial_ax.set(xlabel=r'$r/r_{500}$')

# Add the legend
radial_ax.legend(frameon=False)

plt.tight_layout()
fig.savefig(f'{top_dir}Data_Repository/Project_Data/SPT-IRAGN/Binned_Analysis/Plots/SPT_AGN_Combined_Freq_Plot.pdf')
