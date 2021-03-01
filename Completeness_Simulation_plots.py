"""
Completeness_Simulation_plots.py
Author: Benjamin Floyd

Generates completeness curves from the results of the completeness simulations.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

sptsz_file = 'Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SPT-SZ_2500d/Results/' \
             'SPTSZ_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'
sptpol_file = 'Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SPTpol_100d/Results/' \
              'SPTpol_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'

# Load in the results
with open(sptsz_file, 'r') as f_sptsz, open(sptpol_file, 'r') as f_sptpol:
    sptsz = json.load(f_sptsz)
    sptpol = json.load(f_sptpol)

# Remove the magnitude bin entry from all dictionaries
mag_bins = sptsz.pop('magnitude_bins', None)[:-1]
del sptpol['magnitude_bins']

# Remove any cluster we've already determined unusable
spt_sz_clusters_to_exclude = {'SPT-CLJ0045-5757', 'SPT-CLJ0201-6051', 'SPT-CLJ0230-4427', 'SPT-CLJ0456-5623',
                              'SPT-CLJ0646-6236', 'SPT-CLJ2017-5936', 'SPT-CLJ2133-5410', 'SPT-CLJ2138-6317',
                              'SPT-CLJ2232-6151', 'SPT-CLJ2332-5358', 'SPT-CLJ2341-5726'}
sptpol_clusters_to_exclude = {'SPT-CLJ0002-5214', 'SPT-CLJ2341-5640', 'SPT-CLJ2357-5953'}
for cluster_id in spt_sz_clusters_to_exclude:
    sptsz.pop(cluster_id, None)
for cluster_id in sptpol_clusters_to_exclude:
    sptpol.pop(cluster_id, None)

# Because the SPT-SZ results are stored using observed ids not the official ids, we need to load in a look-up dictionary
with open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_observed_to_official_ids.json', 'r') as f, \
        open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_official_to_observed_ids.json', 'r') as g:
    obs_to_official_ids = json.load(f)
    official_to_obs_ids = json.load(g)

# Plot the SPT-SZ 2500d clusters
fig, ax = plt.subplots()
for curve in sptsz.values():
    ax.plot(mag_bins, curve, color='k', alpha=0.2)
ax.plot(mag_bins, np.median(list(list(curve) for curve in sptsz.values()), axis=0), color='r')
ax.set(title='SPT-SZ 4.5um Completeness Simulations', xlabel='Vega Magnitude', ylabel='Recovery Rate')
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SPT-SZ_2500d/Plots/SPT-SZ_I2_completeness_sim_curves.pdf')
plt.show()

# Plot the SPTpol 100d clusters
fig, ax = plt.subplots()
for curve in sptpol.values():
    ax.plot(mag_bins, curve, color='k', alpha=0.2)
ax.plot(mag_bins, np.median(list(list(curve) for curve in sptpol.values()), axis=0), color='r')
ax.set(title='SPTpol 4.5um Completeness Simulations', xlabel='Vega Magnitude', ylabel='Recovery Rate')
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SPTpol_100d/Plots/SPTpol_I2_completeness_sim_curves.pdf')
plt.show()

# For the clusters common to both surveys supersede the SPTpol/SSDF curves for the deeper, SPT-SZ/targeted obs. curves.
sptpol_exclusive = {cluster_id: curve for cluster_id, curve in sptpol.items()
                    if official_to_obs_ids.get(cluster_id, '') not in sptsz}
combined_sample = {**sptsz, **sptpol_exclusive}

# Compute the combined sample median curve and the 85% Completeness level
median_curve = np.median(list(list(curve) for curve in combined_sample.values()), axis=0)
median_interp = interp1d(mag_bins, median_curve)
faint_end_cut = root_scalar(lambda x: median_interp(x) - 0.85, bracket=[16.5, 18.0]).root
print(f'Faint-end magnitude cut for 85% completeness: {faint_end_cut:.2f}')

# Plot the combined sample
fig, ax = plt.subplots()
for curve in combined_sample.values():
    ax.plot(mag_bins, curve, color='k', alpha=0.2)
ax.plot(mag_bins, median_curve, color='r')
ax.axvline(x=faint_end_cut, ls='--', color='r', alpha=0.5)
ax.axhline(y=0.85, ls='--', color='r', alpha=0.5)
ax.plot(faint_end_cut, 0.85, marker='o', color='r')
ax.set(title='SPTcl 4.5um Completeness Simulations', xlabel='Vega Magnitude', ylabel='Recovery Rate')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SPTcl/Plots/SPTcl_I2_completeness_sim_curves.pdf')
plt.show()
