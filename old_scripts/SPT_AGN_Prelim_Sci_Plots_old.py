"""
SPT_AGN_Prelim_Sci_Plots.py
Author: Benjamin Floyd

This script generates the preliminary science plots for the SPT AGN study.
"""

from __future__ import print_function, division

from os import listdir

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.stats import bootstrap
from astropy.table import Table, vstack
from astropy.utils import NumpyRNGContext

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Set our cosmology
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

# The rad_field AGN surface density from SDWFS was found to be 0.371 AGN / arcmin^2.
field_surf_den = 0.371 / u.arcmin**2
field_surf_den_mpc = field_surf_den * (cosmo.kpc_proper_per_arcmin(0.8).to(u.Mpc / u.arcmin))**2

# Read in all the catalogs
AGN_cats = [Table.read('Data/Output/'+f, format='ascii') for f in listdir('Data/Output/') if not f.startswith('.')]

# Convert the radial distance column in the catalogs from arcmin to Mpc.
for cat in AGN_cats:
    cat['RADIAL_DIST'].unit = u.arcmin
    cat['RADIAL_DIST_Mpc'] = (cat['RADIAL_DIST'] * cosmo.kpc_proper_per_arcmin(cat['REDSHIFT'][0])).to(u.Mpc)

    # Calculate the r500
    cat['r500'] = (3 * cat['M500'] / (4 * 500 * cosmo.critical_density(cat['REDSHIFT']).to(u.Msun / u.Mpc ** 3)))**(1/3)

    # Define a new distance as r/r500
    cat['RADIAL_DIST_r_r500'] = cat['RADIAL_DIST_Mpc'] / cat['r500']

    cat['cluster_area'] = np.pi * (2 * cat['r500'])**2


# Combine all the catalogs into a single table.
full_AGN_cat = vstack(AGN_cats)

# Set our redshift bins
low_z_bin = full_AGN_cat[np.where(full_AGN_cat['REDSHIFT'] <= 0.8)]
high_z_bin = full_AGN_cat[np.where(full_AGN_cat['REDSHIFT'] > 0.8)]

# Group by cluster id to get the number of clusters in each bin
low_z_bin_Ncl = len(low_z_bin.group_by('SPT_ID').groups.keys)
high_z_bin_Ncl = len(high_z_bin.group_by('SPT_ID').groups.keys)

# Set up radial bins
rad_bin_arcmin = np.arange(full_AGN_cat['RADIAL_DIST'].min(), full_AGN_cat['RADIAL_DIST'].max(), 0.5)

# Generate a histogram for each bin for radial distance
low_z_rad_hist, low_z_rad_edges = np.histogram(low_z_bin['RADIAL_DIST_r_r500'],
                                               weights=low_z_bin['completeness_correction'], bins=rad_bin_arcmin)
high_z_rad_hist, high_z_rad_edges = np.histogram(high_z_bin['RADIAL_DIST_r_r500'],
                                                 weights=high_z_bin['completeness_correction'], bins=rad_bin_arcmin)

# Compute the area in each bin.
low_z_rad_area = [np.pi * (low_z_rad_edges[n+1]**2 - low_z_rad_edges[n]**2) for n in range(len(low_z_rad_edges) - 1)]
high_z_rad_area = [np.pi * (high_z_rad_edges[n+1]**2 - high_z_rad_edges[n]**2) for n in range(len(high_z_rad_edges) - 1)]

# Compute the surface density in each bin.
low_z_rad_surf_den = low_z_rad_hist / low_z_rad_area / low_z_bin_Ncl
high_z_rad_surf_den = high_z_rad_hist / high_z_rad_area / high_z_bin_Ncl

# For the errors we will preform a bootstrap resampling to obtain the scatter.
# We need to keep track of the completeness corrections to properly weight the histograms.
low_z_rad_bootarray = np.array(low_z_bin['RADIAL_DIST'], low_z_bin['completeness_correction']).T
high_z_rad_bootarray = np.array(high_z_bin['RADIAL_DIST'], high_z_bin['completeness_correction']).T

# Preform the resampling
with NumpyRNGContext(1):
    low_z_rad_boot = bootstrap(low_z_rad_surf_den, 1000)
    high_z_rad_boot = bootstrap(high_z_rad_surf_den, 1000)

# Generate the histograms for the samples
low_z_rad_boot_hists = []
for sample in low_z_rad_boot:
    low_z_rad_boot, _ = np.histogram(sample.T[0], weights=sample.T[1], bins=rad_bin_arcmin)
low_z_rad_err = np.std(np.array(low_z_rad_boot_hists), axis=0)

high_z_rad_boot_hists = []
for sample in high_z_rad_boot:
    high_z_rad_boot, _ = np.histogram(sample.T[0], weights=sample.T[1], bins=rad_bin_arcmin)
high_z_rad_err = np.std(np.array(high_z_rad_boot_hists), axis=0)

# Center the bins
low_z_rad_cent = low_z_rad_edges[:-1] + np.diff(low_z_rad_edges) / 2.
high_z_rad_cent = high_z_rad_edges[:-1] + np.diff(high_z_rad_edges) / 2.

# Set up mass bins
mass_bins = np.arange(full_AGN_cat['M500'].min(), full_AGN_cat['M500'].max(), 2e14)

# Generate a histogram for each bin for mass
low_z_mass_hist, low_z_mass_edges = np.histogram(low_z_bin['M500'], weights=low_z_bin['completeness_correction'] / (low_z_bin['cluster_area'] * low_z_bin_Ncl), bins=mass_bins)
high_z_mass_hist, high_z_mass_edges = np.histogram(high_z_bin['M500'], weights=high_z_bin['completeness_correction'] / (high_z_bin['cluster_area'] * high_z_bin_Ncl), bins=mass_bins)

# # Calculate the Poisson errors in each bin
# low_z_mass_err = np.sqrt(low_z_mass_hist) #/ low_z_mass_hist
# high_z_mass_err = np.sqrt(high_z_mass_hist) #/ high_z_mass_hist

# Center the bins
low_z_mass_cent = low_z_mass_edges[:-1] + np.diff(low_z_mass_edges) / 2.
high_z_mass_cent = high_z_mass_edges[:-1] + np.diff(high_z_mass_edges) / 2.

# Make the radial distance plot
fig, ax = plt.subplots()
ax.errorbar(low_z_rad_cent, low_z_rad_surf_den, yerr=low_z_rad_err, fmt='o', c='C0', label='$z \leq 0.8$')
ax.errorbar(high_z_rad_cent, high_z_rad_surf_den, yerr=high_z_rad_err, fmt='o', c='C1', label='$z > 0.8$')
ax.axhline(y=field_surf_den.value, c='k', linestyle='--')
ax.set(title='SPT Clusters', xlabel='Radius [arcmin]', ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [arcmin$^{-2}$]',
       ylim=[-0.25, 1.50])
ax.legend()
# fig.savefig('Data/Plots/SPT_AGN_Radial_Distance_Sci_Plot.pdf', format='pdf')

# # Make the mass plot
# fig, ax = plt.subplots()
# ax.errorbar(low_z_mass_cent, low_z_mass_hist, yerr=low_z_mass_err, fmt='o', c='C0', label='$z \leq 0.8$')
# ax.plot(low_z_mass_cent, low_z_mass_hist, c='C0')
# ax.errorbar(high_z_mass_cent, high_z_mass_hist, yerr=high_z_mass_err, fmt='o', c='C1', label='$z > 0.8$')
# ax.plot(high_z_mass_cent, high_z_mass_hist, c='C1')
# ax.axhline(y=field_surf_den_mpc.value, c='k', linestyle='--')
# ax.set(title='SPT Clusters', xlabel='$M_{500} [M_\odot]$', ylabel='$\Sigma_{\mathrm{AGN}}$ per cluster [Mpc$^{-2}$]',
#        xscale='log')
# ax.legend()
# fig.savefig('Data/Plots/SPT_AGN_Mass_Sci_Plot.pdf', format='pdf')
plt.show()