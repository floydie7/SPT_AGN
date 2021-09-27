"""
Bleem_redshift_check.py
Author: Benjamin Floyd

This script checks the SPT clusters that are matched to the Bleem catalog but have no redshift and mass information.
"""

from __future__ import print_function, division
from astropy.table import Table
from astropy.io import fits, ascii
from Pipeline_functions import file_pairing, catalog_image_match

# Load in the full Bleem catalog
Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))  # Bleem+15 Master SPT Cluster catalog

# Pair the files
cluster_list = file_pairing('Data/Catalogs/', 'Data/Images/')
print("File pairing complete, Clusters in directory: {num_clusters}".format(num_clusters=len(cluster_list)))

# Match the clusters to Bleem
print("Matching Images against Bleem Catalog.")
matched_list = catalog_image_match(cluster_list, Bleem, cat_ra_col='RA', cat_dec_col='DEC')

# Check that the centers are well matched
matched_list = [cluster for cluster in matched_list if cluster['center_sep'] <= 60.0]
print("Matched clusters (within 1 arcmin): {num_clusters}".format(num_clusters=len(matched_list)))

# List to place the bad clusters
bad_clusters = []

Bleem_problems = []

# For all the matched clusters, check the mass column in Bleem
for cluster in matched_list:
    Bleem_idx = cluster['Bleem_idx']

    if Bleem['M500'][Bleem_idx] == 0.0:
        bad_clusters.append(cluster)

        if len(Bleem_problems) == 0:
            Bleem_problems = Table(Bleem[Bleem_idx])
        else:
            Bleem_problems.add_row(Bleem[Bleem_idx])

print("Number of matched clusters without mass data: {0}".format(len(bad_clusters)))

ascii.write(Bleem_problems, 'Data/Bleem_problem_clusters.cat')

# for cluster in bad_clusters:
#     print(cluster['sex_cat_path'], cluster['Bleem_idx'])

