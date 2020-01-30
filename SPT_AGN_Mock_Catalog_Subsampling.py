"""
SPT_AGN_Mock_Catalog_Subsampling.py
Author: Benjamin Floyd

Takes an existing mock catalog and creates a subsample to simulate a simplistic incomplete catalog.
"""

import numpy as np
from astropy.table import Table

# Set the ratio we wish to sample to. The true data sample is roughly 85% of its "completeness corrected" number.
completeness_ratio = 0.85

# Set the random seed
object_seed = 930
print(f'Using a seed of {object_seed}')

# Create random number generator
object_rng = np.random.default_rng(object_seed)

# Read in the catalogs
cat_filename = 'Data/MCMC/Mock_Catalog/Catalogs/Final_tests/Slope_tests/trial_4/realistic/' \
               'mock_AGN_catalog_t0.462_e4.00_z-1.00_b1.00_C0.371_rc0.100_maxr5.00_clseed890_objseed930_slope_test.cat'
catalog = Table.read(cat_filename, format='ascii')

# Calculate the number of objects we will place in our subsample
raw_number = len(catalog)
sub_number = int(round(raw_number * completeness_ratio))
print(f'Number of objects in full catalog: {raw_number}\nNumber of objects in subsampled catalog: {sub_number}')

# Perform subsampling
sub_catalog = Table(object_rng.choice(catalog, sub_number, replace=False))

# Append a "completeness correction" value to the catalog
sub_catalog['COMPLETENESS_CORRECTION'] = 1 / completeness_ratio

# Write the catalog to disk
sub_catalog.write('Data/MCMC/Mock_Catalog/Catalogs/Final_tests/Slope_tests/trial_5/'
                  'mock_AGN_catalog_t0.462_e4.00_z-1.00_b1.00_C0.371_rc0.100_maxr5.00_clseed890_objseed930_slope_test.cat',
                  format='ascii')
