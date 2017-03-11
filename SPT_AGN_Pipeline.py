"""
SPT_AGN_Pipeline.py
Author: Benjamin Floyd
This script executes the pipeline consisting of functions from Pipeline_functions.py.
"""

from __future__ import print_function
from time import time
from Pipeline_functions import *


# Run the pipeline.
start_time = time()
print("Beginning Pipeline")
cluster_list = file_pairing('Data/Catalogs/', 'Data/Images/')
print("File pairing complete, Clusters in directory: ", len(cluster_list))

Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))

print("Matching Images against Bleem Catalog.")
matched_list = catalog_image_match(cluster_list, Bleem, cat_ra_col='RA', cat_dec_col='DEC')

# fig, ax = plt.subplots()
# ax.hist([matched_list[i][6] for i in range(len(matched_list))], bins=1e4)
# ax.set(title='Separation between Bleem and center pixel', xlabel='separation (arcsec)')
# ax.set_xlim([0,120])
# plt.show()

matched_list = [matched_list[l] for l in range(len(matched_list)) if matched_list[l][6] <= 60.0]
print("Matched clusters (within 1 arcmin): ", len(matched_list))

print("Applying mask flags.")
cluster_list = mask_flag(matched_list, 'Data/mask_notes.txt')

print("Beginning cluster level operations.")
for cluster in cluster_list:
    print("Generating coverage level mask.")
    cluster = coverage_mask(cluster, ch1_min_cov=4, ch2_min_cov=4)

    print("Creating object mask.")
    cluster = object_mask(cluster, 'Data/Regions/')

    print("Preforming selection cuts")
    cluster = object_selection(cluster, 'I2_MAG_APER4', cat_ra='ALPHA_J2000', cat_dec='DELTA_J2000',
                               sex_flag_cut=4, snr_cut=5.0, mag_cut=18.0, ch1_ch2_color_cut=0.7)

    print("Objects selected:", len(cluster[9]))

    print("Matching catalogs.")
    match_time_start = time()
    cluster = catalog_match(cluster, Bleem, ['REDSHIFT', 'REDSHIFT_UNC', 'M500', 'DM500'],
                                     sex_ra_col='ALPHA_J2000', sex_dec_col='DELTA_J2000',
                                     master_ra_col='RA', master_dec_col='DEC')
    match_time_end = time()
    print("Time taken calculating separations: ", match_time_end - match_time_start, " s")

    print("Writing final catalog.")
    final_catalogs(cluster, ['SPT_ID', 'ALPHA_J2000', 'DELTA_J2000', 'rad_dist', 'REDSHIFT', 'REDSHIFT_UNC', 'M500',
                              'DM500', 'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I2_MAG_APER4', 'I2_MAGERR_APER4'])

end_time = time()
print("Pipeline finished.")
print("Total runtime: ", end_time - start_time, " s.")