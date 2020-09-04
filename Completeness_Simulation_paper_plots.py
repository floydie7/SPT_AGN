"""
Completeness_Simulation_paper_plots.py
Author: Benjamin Floyd

Generates paper quality plots of the SPT-SZ + SPTpol + SDWFS completeness simulations.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

# Completeness simulation result files
sptsz_file = '/home/ben/PycharmProjects/SPT_AGN/Data/Comp_Sim/Results/SPTSZ_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'
sptpol_file = '/home/ben/PycharmProjects/SPT_AGN/Data/Comp_Sim/SPTpol/Results/SPTpol_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'
sdwfs_file = '/home/ben/PycharmProjects/SDWFS/Data/Comp_Sim/Results/I2_results_gaussian_fwhm172_corr005_mag02_final.npy'

# Read in the numpy save file
sdwfs = np.load(sdwfs_file, encoding='latin1', allow_pickle=True).item()

# Read in the json files
with open(sptsz_file, 'r') as sptsz_f, open(sptpol_file, 'r') as sptpol_f:
    sptsz = json.load(sptsz_f)
    sptpol = json.load(sptpol_f)

# Because the SPT-SZ results are stored using observed ids not the official ids, we need to load in a look-up dictionary
with open('Data/SPT-SZ_observed_to_official_ids.json', 'r') as f, \
        open('Data/SPT-SZ_official_to_observed_ids.json', 'r') as g:
    obs_to_official_ids = json.load(f)
    official_to_obs_ids = json.load(g)

# Remove any cluster we've already determined unusable
spt_sz_clusters_to_exclude = {'SPT-CLJ0045-5757', 'SPT-CLJ0201-6051', 'SPT-CLJ0230-4427', 'SPT-CLJ0456-5623',
                              'SPT-CLJ0646-6236', 'SPT-CLJ2017-5936', 'SPT-CLJ2133-5410', 'SPT-CLJ2138-6317',
                              'SPT-CLJ2232-6151', 'SPT-CLJ2332-5358', 'SPT-CLJ2341-5726'}
sptpol_clusters_to_exclude = {'SPT-CLJ0002-5214', 'SPT-CLJ2341-5640', 'SPT-CLJ2357-5953'}
for cluster_id in spt_sz_clusters_to_exclude:
    sptsz.pop(cluster_id, None)
for cluster_id in sptpol_clusters_to_exclude:
    sptpol.pop(cluster_id, None)

# Convert all the SPT-SZ keys to official IDs
sptsz = {obs_to_official_ids.get(sz_id, sz_id): data for sz_id, data in sptsz.items()}

# Remove the magnitude bins
magnitude_bins_spt = np.array(sptsz.pop('magnitude_bins', None)[:-1]) + 0.25
del sptpol['magnitude_bins']
magnitude_bins_sdwfs = sdwfs.pop('magnitude_bins', None)[:-1]


# Merge the cluster samples perfering the SPT-SZ (targeted obs) values over the SPTpol (SSDF cutout) values.
sptcl = {**sptpol, **sptsz}

# Compute the median curves for both the SDWFS and SPT samples
median_sdwfs = np.median(list(list(curve) for curve in sdwfs.values()), axis=0)
median_sptcl = np.median(list(list(curve) for curve in sptcl.values()), axis=0)

# Find the bounds of the scatter on the SPTcl completeness curves
min_curve_sptcl = np.min(list(list(curve) for curve in sptcl.values()), axis=0)
max_curve_sptcl = np.max(list(list(curve) for curve in sptcl.values()), axis=0)

fig, ax = plt.subplots()
ax.plot(magnitude_bins_sdwfs, median_sdwfs, color='C0', label='SDWFS')
ax.plot(magnitude_bins_spt, median_sptcl, color='C1', label='SPTcl')
# ax.fill_between(magnitude_bins_spt, min_curve_sptcl, max_curve_sptcl, color='C1', alpha=0.3)
ax.axvline(x=17.46, color='k', linestyle='--', alpha=0.2)
ax.set(xlabel='[4.5] (Vega)', ylabel='Completeness')
ax.legend()
fig.savefig('Data/Comp_Sim/Plots/SPTcl_SDWFS_completeness_sim_paper.pdf')
plt.show()
