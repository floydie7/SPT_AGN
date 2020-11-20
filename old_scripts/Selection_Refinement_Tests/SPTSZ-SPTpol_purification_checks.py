"""
SPTSZ-SPTpol_purification_checks.py
Author: Benjamin Floyd

Takes in the common clusters from the SPT-SZ and SPTpol 100d surveys and compares number counts after the purification
has been applied as part of the catalog making.
"""

import glob
import re

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

# Get a list of all the catalog realizations
realizations_list = glob.glob('Data/Output/common_only_realizations/*.fits')

# Regular expression to find the realization key
realization_key = re.compile(r'(\d{3}).fits')

# Read in all the catalogs
realizations_catalogs = {'real_'+realization_key.search(name).group(1): Table.read(name) for name in realizations_list}

# Convert the bytestring columns into standard unicode
for catalog in realizations_catalogs.values():
    catalog.convert_bytestring_to_unicode()

# For some reason, we accidentally collected too many clusters (i.e., clusters that aren't perfect overlaps between both
# of the surveys). Thus, we need to filter for the true common clusters.
true_common_ids = ['SPT-CLJ0000-5748', 'SPT-CLJ0001-5440', 'SPT-CLJ2259-5431', 'SPT-CLJ2300-5331', 'SPT-CLJ2301-5546',
                   'SPT-CLJ2311-5820', 'SPT-CLJ2337-5912', 'SPT-CLJ2337-5942', 'SPT-CLJ2341-5119', 'SPT-CLJ2342-5411',
                   'SPT-CLJ2351-5452', 'SPT-CLJ2355-5055', 'SPT-CLJ2358-5229']
realizations_catalogs = {real_id: catalog[np.in1d(catalog['SPT_ID'], true_common_ids)]
                         for real_id, catalog in realizations_catalogs.items()}

# Process all the realizations and compute number counts (raw and corrected) per cluster and survey
row_data = []
for real_id, catalog in realizations_catalogs.items():
    # Split the catalog by cluster and survey
    catalog_cluster_grp = catalog.group_by(['SPT_ID', 'SURVEY'])
    for cluster_survey in catalog_cluster_grp.groups:
        row_data.append({'Realization_ID': real_id,
                         'SPT_ID': cluster_survey['SPT_ID'][0],
                         'SURVEY': cluster_survey['SURVEY'][0],
                         'raw_counts': len(cluster_survey),
                         'completeness_corrected_counts': cluster_survey['COMPLETENESS_CORRECTION'].sum()})

# Collate the data into a table for easier access
realizations_number_counts = Table(rows=row_data)

#%% Group the count data by cluster and survey (over all realizations)
realizations_number_counts_grp = realizations_number_counts.group_by(['SPT_ID'])

# Make histograms of the number counts (2 for each cluster)
for cluster in realizations_number_counts_grp.groups:
    cluster_grp = cluster.group_by('SURVEY')
    fig, ax = plt.subplots()
    for survey in cluster_grp.groups:
        ax.hist(survey['completeness_corrected_counts'], bins=100, label=f'{survey["SURVEY"][0]}')
    ax.legend()
    ax.set(title=f'{cluster["SPT_ID"][0]}', xlabel='Completeness Corrected Counts', ylabel='Number of Realizations')
    fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/AGN_Number_Counts/'
                f'{cluster["SPT_ID"][0]}_comp_corr_num_counts.pdf')
    plt.show()