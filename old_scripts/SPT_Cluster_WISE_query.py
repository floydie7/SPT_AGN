"""
SPT_Cluster_WISE_query.py
Author: Benjamin Floyd

This script generates an IPAC table of coordinates to use as an IRSA query in the ALLWISE catalog.
"""

from __future__ import print_function, division
from astropy.io import ascii, fits
from astropy.table import Table
import numpy as np
from Pipeline_functions import file_pairing, catalog_image_match

# Load in the Bleem table and filter the clusters without mass data.
Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))
Bleem = Bleem[np.where(Bleem['M500'] != 0)]

# Pair the files and pass the list to the catalog/image matcher.
cluster_list = file_pairing('Data/Catalogs', 'Data/Images')
cluster_list = catalog_image_match(cluster_list, Bleem)

# Go through the cluster list and collect all the Bleem indices.
cluster_idx = []
for cluster in cluster_list:
    cluster_idx.append(cluster['Bleem_idx'])

# Filter the Bleem table for our selected clusters.
IRAC_clusters = Bleem[cluster_idx]

# Generate the query table.
query_table = IRAC_clusters['SPT_ID', 'RA', 'DEC']

# Write the table to disk using the IPAC format.
ascii.write(query_table, 'Data/ALLWISE_SPT_Clusters.tbl', format='ipac')
