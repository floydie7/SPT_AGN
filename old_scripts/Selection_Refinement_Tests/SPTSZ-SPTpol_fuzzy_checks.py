"""
SPTSZ-SPTpol_fuzzy_checks.py
Author: Benjamin Floyd

Collects and displays number count differences to compare the fuzzy membership selection between the two surveys.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

# Read in the fuzzy-selected catalog
fuzzy_catalog = Table.read('Data/Output/common_only_realizations/SPTcl_common_only_IRAGN_fuzzy.fits')

# Group by cluster and survey
fuzzy_catalog_grp = fuzzy_catalog.group_by(['SPT_ID', 'SURVEY'])

row_data = []
for cluster_survey in fuzzy_catalog_grp.groups:
    row_data.append({'Name': f"{cluster_survey['SPT_ID'][0]}_{cluster_survey['SURVEY'][0]}",
                     'raw': len(cluster_survey),
                     'completeness_corrected': cluster_survey['COMPLETENESS_CORRECTION'].sum(),
                     'membership_degree': cluster_survey['selection_membership'].sum(),
                     'total_weighted': np.sum(cluster_survey['COMPLETENESS_CORRECTION'] *
                                                     cluster_survey['selection_membership'])})
number_counts = Table(rows=row_data)
number_counts_pd = number_counts.to_pandas()

#%%
number_counts_pd.plot(x='Name', y=['membership_degree', 'total_weighted'], kind='bar')
plt.show()