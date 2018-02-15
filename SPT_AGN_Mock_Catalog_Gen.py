"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""

from __future__ import print_function, division

import numpy as np
from astropy.table import Table, Column

# Set our random seed
np.random.seed(3543)

# Read in Bleem catalog
Bleem = Table.read('Data/2500d_cluster_sample_fiducial_cosmology.fits')
Bleem = Bleem[np.where(Bleem['M500'] != 0.0)]

# Number of data points to generate
n_points = int(1e6)

# Draw radial distances in units of r500 from a uniform distribution.
rad_dist = np.random.uniform(0.1, 1.5, n_points)

# Pull random row indexes from Bleem
rand_Bleem_idx = np.random.randint(0, len(Bleem), n_points)

# Create a column from the radial distances
radial_dist = Column(rad_dist, name='r_r500_radial')

# Pull the rows from Bleem
mock_candidates = Bleem['SPT_ID', 'REDSHIFT', 'M500'][rand_Bleem_idx]

# Add the radial distances to the catalog
mock_candidates.add_column(radial_dist)

# Set parameter values
eta_true = 1.2
beta_true = -1.5
zeta_true = -1.0
C_true = 0.371

# Need the number of clusters in the sample
N_cl = len(mock_candidates.group_by('SPT_ID').groups.keys)

# Run the candidates through the model with the values above
N_model = np.array((1 + mock_candidates['REDSHIFT'])**eta_true
                   * (mock_candidates['r_r500_radial'])**beta_true
                   * (mock_candidates['M500'] / 1e15)**zeta_true)

# Normalize the model values to create probabilities
N_model_normed = N_model / np.max(N_model)

# Draw a random number between [0,1] and if the number is smaller than the probabilities from our model, we keep the
# entry from our candidates for our mock catalog.
mock_catalog = Table(names=['SPT_ID', 'REDSHIFT', 'M500', 'r_r500_radial'], dtype=['S16', 'f8', 'f8', 'f8'])
for i in range(len(mock_candidates)):
    # Draw random number
    theta = np.random.uniform(0, 1)

    # Check if the candidate should be added to the catalog
    if theta < N_model_normed[i]:
        mock_catalog.add_row(mock_candidates[i])

print('Number of objects in Mock catalog: {}'.format(len(mock_catalog)))

# Write the mock catalog to disk
mock_catalog.write('Data/MCMC/Mock_Catalog/Catalogs/mock_AGN_catalog.cat', format='ascii', overwrite=True)

# Split the catalog into sub-groups of ~1500 AGN to mimic our real data sample.
for i in np.arange(1500, len(mock_catalog), 1500):
    sub_cat = mock_catalog[i-1500:i]

    print('Number of objects in subcatalog {j}: {n}'.format(j=i // 1500 - 1, n=len(sub_cat)))

    sub_cat.write('Data/MCMC/Mock_Catalog/Catalogs/mock_AGN_subcatalog{j}.cat'.format(j=i // 1500 - 1),
                  format='ascii', overwrite=True)
