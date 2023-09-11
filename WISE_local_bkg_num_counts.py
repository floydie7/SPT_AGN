"""
WISE_local_bkg_num_counts.py
Author: Benjamin Floyd

Computes the number count distributions using the catWISE local background catalogs.
"""
import glob
import json
import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astro_compendium.utils.small_poisson import small_poisson
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.interpolate import interp1d

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

cluster_id_pattern = re.compile(r'SPT-CLJ\d+-\d+')

# Read in the catalogs
catalog_files = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/*.ecsv')
# cluster_name = 'SPT-CLJ2120-4728'
# catalog_files = [f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/{cluster_name}_wise_local_bkg.ecsv']
catalogs = {cluster_id_pattern.search(filename).group(0): Table.read(filename) for filename in catalog_files}

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# For our faint-end magnitude cut we will use the SDWFS completeness curve
with open('Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SDWFS/Results/'
          'I2_results_gaussian_fwhm172_corr005_mag02_final.json', 'r') as f:
    sdwfs_comp_sim = json.load(f)
comp_sim_mag_bins = sdwfs_comp_sim.pop('magnitude_bins', None)[:-1]
mean_curve = interp1d(np.mean(list(list(curve) for curve in sdwfs_comp_sim.values()), axis=0), comp_sim_mag_bins)

# Magnitude cuts
bright_end_cut = 14.00
faint_end_cut = 16.25

# Bin into 0.25 magnitude bins
mag_bin_width = 0.25
num_counts_mag_bins = np.arange(bright_end_cut, faint_end_cut, mag_bin_width)

num_count_dists = {}
num_count_errs = {}
for cluster_id, catalog in catalogs.items():
    cluster_z = catalog['REDSHIFT'][0]
    cluster_r200 = catalog['R200'][0] * u.Mpc

    # Cut the catalog between the magnitude limits
    catalog = catalog[(bright_end_cut <= catalog['w2mpro']) & (catalog['w2mpro'] <= faint_end_cut)]

    # Select for AGN
    color_threshold = agn_purity_color(cluster_z)
    # catalog = catalog[catalog['w1mpro'] - catalog['w2mpro'] >= color_threshold]

    # Calculate the area of the background field
    outer_radius = 3 * cluster_r200 * cosmo.arcsec_per_kpc_proper(cluster_z).to(u.deg / u.Mpc)
    inner_radius = 2 * cluster_r200 * cosmo.arcsec_per_kpc_proper(cluster_z).to(u.deg / u.Mpc)
    area = np.pi * (outer_radius ** 2 - inner_radius ** 2)

    # Create histogram
    dn_dm, _ = np.histogram(catalog['w2mpro'], bins=num_counts_mag_bins)
    dn_dm_weighted = dn_dm / (area.value * mag_bin_width)
    num_count_dists[cluster_id] = dn_dm_weighted

    # Compute the errors
    # dn_dm_err = np.sqrt(dn_dm) / (area.value / mag_bin_width)
    dn_dm_err = tuple(err / (area.value * mag_bin_width) for err in small_poisson(dn_dm))[::-1]
    num_count_errs[cluster_id] = dn_dm_err

#%% Spot check plot
mag_bin_centers = num_counts_mag_bins[:-1] + np.diff(num_counts_mag_bins) / 2
for cluster_name in num_count_dists:
    fig, ax = plt.subplots()
    ax.errorbar(mag_bin_centers, num_count_dists[cluster_name], xerr=mag_bin_width / 2, yerr=num_count_errs[cluster_name],
                fmt='.')
    ax.set(title=f'{cluster_name}', xlabel='W2 (Vega)', ylabel=r'$dN/dm$ [deg$^{-2}$ mag$^{-1}$]', yscale='log')
    # plt.show()
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/number_count_distributions/'
                f'{cluster_name}_WISE_local_bkg_dN_dm.pdf')
