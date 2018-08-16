"""
SPT_AGN_Pipeline.py
Author: Benjamin Floyd
This script executes the pipeline consisting of functions from Pipeline_functions.py.
"""

from __future__ import print_function

from time import time

from Pipeline_functions import *

# Load in external files.
Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))  # Bleem+15 Master SPT Cluster catalog

# Clean the table of the rows without mass data. These are unconfirmed cluster candidates.
Bleem = Bleem[np.where(Bleem['M500'] != 0.0)]

# Dictionary of completeness curve values for all clusters
completeness_dictionary = np.load('Data/Comp_Sim/Results/SPT_I2_results_gaussian_fwhm202_corr011_mag02.npy', encoding='latin1').item()

# Run the pipeline.
start_time = time()
print("Beginning Pipeline")
cluster_list = file_pairing('Data/Catalogs/', 'Data/Images/')
print("File pairing complete, Clusters in directory: {num_clusters}".format(num_clusters=len(cluster_list)))

print("Matching Images against Bleem Catalog.")
matched_list = catalog_image_match(cluster_list, Bleem, cat_ra_col='RA', cat_dec_col='DEC', max_sep=6.0)

print("Matched clusters (within 1 arcmin): {num_clusters}".format(num_clusters=len(matched_list)))

print("Applying mask flags.")
cluster_list = mask_flag(matched_list, 'Data/mask_notes.txt')

print("Beginning cluster level operations.")
for cluster in cluster_list:
    print("Working on {spt_id}".format(spt_id=cluster['SPT_ID']))
    print("Generating coverage level mask.")
    cluster = coverage_mask(cluster, ch1_min_cov=4, ch2_min_cov=4)

    print("Creating object mask.")
    cluster = object_mask(cluster, 'Data/Regions/')

    print("Preforming selection cuts")
    cluster = object_selection(cluster, 'I2_MAG_APER4', cat_ra='ALPHA_J2000', cat_dec='DELTA_J2000',
                               sex_flag_cut=4, mag_cut=17.45, ch1_ch2_color_cut=0.7)

    print("Objects selected: {num_objects}".format(num_objects=len(cluster['catalog'])))

    if len(cluster['catalog']) != 0:  # This is to protect against cluster catalogs that get dropped due to no survivors
        print("Matching catalogs.")
        match_time_start = time()
        cluster = catalog_match(cluster, Bleem, ['REDSHIFT', 'REDSHIFT_UNC', 'M500', 'DM500'],
                                         sex_ra_col='ALPHA_J2000', sex_dec_col='DELTA_J2000',
                                         master_ra_col='RA', master_dec_col='DEC')
        match_time_end = time()
        print("Time taken calculating separations: {match_time} s".format(match_time=match_time_end - match_time_start))

        print("Calculating the total (good) area of the cutout image.")
        cluster = image_area(cluster, units='arcmin')

        print("Computing completeness values for selected objects.")
        # cluster = completeness_value(cluster, 'I2_MAG_APER4', completeness_dictionary)

        print("Writing final catalog.")
        final_catalogs(cluster, ['SPT_ID', 'ALPHA_J2000', 'DELTA_J2000', 'RADIAL_DIST', 'REDSHIFT', 'REDSHIFT_UNC',
                                 'M500', 'DM500', 'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I2_MAG_APER4', 'I2_MAGERR_APER4',
                                 'IMAGE_AREA'])
    else:
        print('******{spt_id} has no objects satisfying our selection cuts.******'.format(spt_id=cluster['SPT_ID']))

end_time = time()
print("Pipeline finished.")
print("Total runtime: {total_time} s".format(total_time=end_time - start_time))
