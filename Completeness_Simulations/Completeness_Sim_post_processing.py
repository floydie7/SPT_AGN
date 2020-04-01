"""
Completeness_Sim_post_processing.py
Author: Benjamin Floyd

Uses the input-output catalogs to generate the normal completeness simulation results file.
"""

import glob
import re
import json
import numpy as np
from astropy.table import Table
import astropy.units as u

hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'

# RegEx for cluster ID
cluster_id = re.compile(r'SPT-CLJ\d+-\d+')

# Magnitude bins used
mag_bins = np.arange(10.0, 22.5, 0.5)

# Maximum separation
max_sep = 2.02 * u.arcsec

# Magnitude threshold
recovery_mag_thresh = 0.2

# Get the list of catalogs
inout_catalogs = glob.iglob(f'{hcc_prefix}/Data/Comp_Sim/Input_Output_catalogs/*_inout.fits')

completeness_results = {}
for catalog in inout_catalogs:
    # Get the cluster ID
    spt_id = cluster_id.search(catalog).group(0)

    # Read in the catalog
    cluster = Table.read(catalog)

    # Create binning
    bins = np.digitize(cluster['selection_band'], mag_bins)

    # Bin the catalog by the input magnitude
    cluster_binned = cluster.group_by(bins)

    recovery_rate = []
    for cluster_mag_bin in cluster_binned.groups:
        # Check to see if we placed any objects in this magnitude bin. If we did, we can continue.
        if len(cluster_mag_bin) != 0:
            continue

        # Select for objects matching our spatial and photometric thresholds
        cluster_mag_bin_recovered = cluster_mag_bin[(cluster_mag_bin['SEP'] <= max_sep.to_value(u.deg)) & (
                np.abs(cluster_mag_bin['selection_band'] - cluster_mag_bin['MAG_APER']) <= recovery_mag_thresh)]

        # Store the recovery rates
        recovery_rate.append(len(cluster_mag_bin_recovered) / len(cluster_mag_bin))

    # Store the results in the dictionary
    completeness_results[spt_id] = recovery_rate

# Add the magnitude values used to create the completeness rates.
completeness_results['magnitude_bins'] = list(mag_bins)

# Save results to disk
results_filename = f'{hcc_prefix}/Comp_Sim/Results/SPTSZ_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'
with open(results_filename, 'w') as f:
    json.dump(completeness_results, f, ensure_ascii=False, indent=4, sort_keys=True)
