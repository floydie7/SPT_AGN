"""
Completeness_Simulation_plots.py
Author: Benjamin Floyd

Generates completeness curves from the results of the completeness simulations.
"""

import json

import matplotlib.pyplot as plt

sptsz_file = 'Data/Comp_Sim/Results/SPTSZ_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'
sptpol_file = 'Data/Comp_Sim/SPTpol/Results/SPTpol_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'
sptsz_rereduced_file = 'Data/Comp_Sim/rereduced/Results/SPTSZ_rereduced_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'

# Load in the results
with open(sptsz_file, 'r') as f_sptsz, \
        open(sptpol_file, 'r') as f_sptpol, \
        open(sptsz_rereduced_file, 'r') as f_rereduced:
    sptsz = json.load(f_sptsz)
    sptpol = json.load(f_sptpol)
    sptsz_rereduced = json.load(f_rereduced)

# Remove the magnitude bin entry from all dictionaries
mag_bins = sptsz.pop('magnitude_bins', None)[:-1]
del sptpol['magnitude_bins']
del sptsz_rereduced['magnitude_bins']

# Because the SPT-SZ results are stored using observed ids not the offical ids, we need to load in a look-up dictionary
with open('Data/SPT-SZ_observed_to_official_ids.json', 'r') as f, \
        open('Data/SPT-SZ_official_to_observed_ids.json', 'r') as g:
    obs_to_official_ids = json.load(f)
    official_to_obs_ids = json.load(g)

for cluster_id in sptsz_rereduced.keys():
    sptsz_rereduced[obs_to_official_ids.get(cluster_id, cluster_id)] = sptsz_rereduced.pop(cluster_id, None)

del sptsz_rereduced['SPT-CLJ0002-5557']

# Collate results from common clusters
comparison = {cluster_id: [sptsz[official_to_obs_ids[cluster_id]], sptpol[cluster_id], sptsz_rereduced[cluster_id]]
              for cluster_id in sptsz_rereduced}

fig, axarr = plt.subplots(ncols=1, nrows=2, sharex='col', figsize=(8, 12))
for ax, (cluster_id, curves) in zip(axarr.flatten(), comparison.items()):
    ax.plot(mag_bins, curves[0], label='Targeted (0.86"/pix)')
    ax.plot(mag_bins, curves[1], label='SSDF (0.6"/pix)')
    ax.plot(mag_bins, curves[2], label='Rereduced Targeted (0.6"/pix)')
    ax.axvline(x=17.46, color='k', linestyle='--', alpha=0.2)
    ax.axhline(y=0.8, color='k', linestyle='--', alpha=0.2)
    ax.set(title=cluster_id, ylabel='Recovery Rate')
    ax.legend()
axarr[-1].set(xlabel='Vega Magnitude')
fig.savefig('Data/Comp_Sim/rereduced/Plots/Rereduction_Comparison_Plot.pdf', format='pdf')
