"""
SPT_SZ_center_uncertainty.py
Author: Benjamin Floyd

Computes the average positional uncertainty in the SPT cluster surveys.
"""

import numpy as np
from astropy.table import Table, join, vstack, unique

# Read in tables
sptcl_agn = vstack([Table.read('Data/Output/SPT_IRAGN.fits'), Table.read('Data/Output/SPTPol_IRAGN.fits')])
bocquet = Table.read('Data/2500d_cluster_sample_Bocquet18.fits')
huang = Table.read('Data/sptpol100d_catalog_huang19.fits')

# Standardize the column names in the Huang catalog to match the Bocquet catalog
huang.rename_column('Dec', 'DEC')
huang.rename_column('redshift', 'REDSHIFT')
huang.rename_column('redshift_unc', 'REDSHIFT_UNC')
huang.rename_column('xi', 'XI')
huang.rename_column('theta_core', 'THETA_CORE')

# Merge the two catalogs
full_spt_catalog = join(bocquet, huang, join_type='outer')
full_spt_catalog.sort(keys=['SPT_ID', 'field'])  # Sub-sorting by 'field' puts Huang entries first
full_spt_catalog = unique(full_spt_catalog, keys='SPT_ID', keep='first')  # Keeping Huang entries over Bocquet
full_spt_catalog.sort(keys='SPT_ID')

# Compute 1-sigma SZ center positional uncertainty
theta_beam = 1.2  # in arcmin
full_spt_catalog['SZ_center_uncert'] = np.sqrt(theta_beam ** 2 + full_spt_catalog['THETA_CORE'] ** 2) / \
                                       full_spt_catalog['XI']

# Keep only clusters that exist in our sample
agn_cluster_list = np.unique(sptcl_agn['SPT_ID'])
matched_catalog = full_spt_catalog[np.isin(full_spt_catalog['SPT_ID'], agn_cluster_list, assume_unique=True)]

# Report median offset
median_uncert = np.median(matched_catalog['SZ_center_uncert'])
print(f'{median_uncert:.3f} arcmin')
