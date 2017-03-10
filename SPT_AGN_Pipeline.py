"""
SPT_AGN_Pipeline.py
Author: Benjamin Floyd
This script executes the pipeline consisting of functions from Pipeline_functions.py.
"""

from time import time
from Pipeline_functions import *


# Run the pipeline.
start_time = time()
print("Beginning Pipeline")
clusters = file_pairing('Data/test/', 'Data/Images/')
print("File pairing complete, Clusters in directory: ", len(clusters))

Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))

print("Matching Images against Bleem Catalog.")
matched_list = catalog_image_match(Bleem, clusters, cat_ra_col='RA', cat_dec_col='DEC')

# fig, ax = plt.subplots()
# ax.hist([matched_list[i][6] for i in range(len(matched_list))], bins=1e4)
# ax.set(title='Separation between Bleem and center pixel', xlabel='separation (arcsec)')
# ax.set_xlim([0,120])
# plt.show()

matched_list = [matched_list[l] for l in range(len(matched_list)) if matched_list[l][6] <= 60.0]
print("Matched clusters (within 1 arcmin): ", len(matched_list))

print("Applying mask flags.")
cluster_matched_flagged = mask_flag(matched_list, 'Data/mask_notes.txt')

print("Matching catalogs.")
match_time_start = time()
cat_matched_list = catalog_match(cluster_matched_flagged, Bleem, ['REDSHIFT', 'REDSHIFT_UNC', 'M500', 'DM500'],
                                 master_ra_col='RA', master_dec_col='DEC')
match_time_end = time()
print("Time taken calculating separtations: ", match_time_end - match_time_start, " s")

print("Generating coverage level masks.")
cov_list = coverage_mask(cat_matched_list, ch1_min_cov=4, ch2_min_cov=4)

print("Creating object masks.")
mask_cat = object_mask(cov_list, 'Data/Regions/')

# Temporary Snippet
final_catalogs(mask_cat, ['SPT_ID', 'ALPHA_J2000', 'DELTA_J2000', 'rad_dist', 'REDSHIFT', 'REDSHIFT_UNC', 'M500',
                          'DM500', 'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I2_MAG_APER4', 'I2_MAGERR_APER4'])


end_time = time()
print("Pipeline finished.")
print("Total runtime: ", end_time - start_time, " s.")