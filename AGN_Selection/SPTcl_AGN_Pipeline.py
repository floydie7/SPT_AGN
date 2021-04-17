"""
SPTcl_AGN_Pipeline.py
Author: Benjamin Floyd

A unified IR-AGN selection pipeline for both the SPT-SZ + SPTpol 100d surveys.
"""

from time import time

import astropy.units as u
import numpy as np
from astropy.table import Table, join, unique, vstack
from synphot import SourceSpectrum, units

from AGN_Selection.Pipeline_functions import SelectIRAGN

# Define directories
# prefix = '/Users/btfkwd/Documents/SPT_AGN/'
prefix = '/home/ben-work/PycharmProjects/SPT_AGN/'

# SPT-SZ Directories
spt_sz_catalog_directory = f'{prefix}Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPT-SZ_2500d'
spt_sz_image_directory = f'{prefix}Data_Repository/Images/SPT/Spitzer_IRAC/SPT-SZ_2500d'
spt_sz_regions_directory = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Regions/SPT-SZ_2500d'
spt_sz_masks_directory = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Masks/SPT-SZ_2500d'

# SPTpol 100d Directories
sptpol_catalog_directory = f'{prefix}Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPTpol_100d'
sptpol_image_directory = f'{prefix}Data_Repository/Images/SPT/Spitzer_IRAC/SPTpol_100d'
sptpol_regions_directory = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Regions/SPTpol_100d'
sptpol_masks_directory = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Masks/SPTpol_100d'

# SDWFS number count distribution file (for purification)
sdwfs_number_count_dist = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/' \
                          f'SDWFS_number_count_distribution_normed.json'

# Polletta QSO2 SED used for computing the J-band absolute magnitudes
polletta_qso2 = SourceSpectrum.from_file(f'{prefix}Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                         wave_unit=u.Angstrom, flux_unit=units.FLAM)

# Completeness simulation results files
spt_sz_completeness_sim_results = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SPT-SZ_2500d/Results/' \
                                  f'SPTSZ_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'
sptpol_completeness_sim_results = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SPTpol_100d/Results/' \
                                  f'SPTpol_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'

# Clusters to manually exclude
spt_sz_clusters_to_exclude = {'SPT-CLJ0045-5757', 'SPT-CLJ0201-6051', 'SPT-CLJ0230-4427', 'SPT-CLJ0456-5623',
                              'SPT-CLJ0646-6236', 'SPT-CLJ2017-5936', 'SPT-CLJ2133-5410', 'SPT-CLJ2138-6317',
                              'SPT-CLJ2232-6151', 'SPT-CLJ2332-5358', 'SPT-CLJ2341-5726'}
sptpol_clusters_to_exclude = {'SPT-CLJ0002-5214', 'SPT-CLJ2341-5640', 'SPT-CLJ2357-5953'}

# Maximum separation allowed between the cluster images' reference pixel and the SPT SZ centers
max_separation = 1.0 * u.arcmin

# Minimum coverage in 3.6 and 4.5 um bands allowed for good pixel map.
# SPT-SZ uses a minimum of 4 exposures
spt_sz_ch1_min_coverage = 4
spt_sz_ch2_min_coverage = 4

# SPTpol 100d uses a minimum of 3 exposures as the SSDF survey was shallower than the targeted observations
sptpol_ch1_min_coverage = 3
sptpol_ch2_min_coverage = 3

# Photometric selection cuts
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.46  # Faint-end 4.5 um magnitude
ch1_ch2_color = 0.7  # Minimum [3.6] - [4.5] color

# Additional columns to include in catalog from SPT catalog
spt_column_names = ['REDSHIFT', 'REDSHIFT_UNC', 'M500', 'M500_uerr', 'M500_lerr']

# Output catalog file name
sptsz_output_catalog = f'{prefix}Data_Repository/Project_data/SPT-IRAGN/Output/SPT-SZ_2500d.fits'
sptpol_output_catalog = f'{prefix}Data_Repository/Project_data/SPT-IRAGN/Output/SPTpol_100d.fits'
sptcl_std_output_catalog = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits'
sptcl_inv_output_catalog = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN_inverse.fits'

# Requested columns for output catalog
output_column_names = ['SPT_ID', 'SZ_RA', 'SZ_DEC', 'ALPHA_J2000', 'DELTA_J2000', 'RADIAL_SEP_ARCMIN',
                       'REDSHIFT', 'REDSHIFT_UNC', 'M500', 'M500_uerr', 'M500_lerr', 'R500', 'RADIAL_SEP_R500',
                       'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I1_FLUX_APER4', 'I1_FLUXERR_APER4', 'I2_MAG_APER4',
                       'I2_MAGERR_APER4', 'I2_FLUX_APER4', 'I2_FLUXERR_APER4', 'J_ABS_MAG', 'COMPLETENESS_CORRECTION',
                       'SELECTION_MEMBERSHIP', 'MASK_NAME']

# Read in SPT-SZ cluster catalog
Bocquet = Table.read(f'{prefix}Data_Repository/Catalogs/SPT/SPT_catalogs/2500d_cluster_sample_Bocquet18.fits')

# For the 20 common clusters between SPT-SZ 2500d and SPTpol 100d surveys we want to update the cluster information from
# the more recent survey. Thus, we will merge the SPT-SZ and SPTpol catalogs together.
Huang = Table.read(f'{prefix}Data_Repository/Catalogs/SPT/SPT_catalogs/sptpol100d_catalog_huang19.fits')

# First we need to rename several columns in the SPTpol 100d catalog to match the format of the SPT-SZ catalog
Huang.rename_columns(['Dec', 'xi', 'theta_core', 'redshift', 'redshift_unc'],
                     ['DEC', 'XI', 'THETA_CORE', 'REDSHIFT', 'REDSHIFT_UNC'])

# Now, merge the two catalogs
SPTcl = join(Bocquet, Huang, join_type='outer')
SPTcl.sort(keys=['SPT_ID', 'field'])  # Sub-sorting by 'field' puts Huang entries first
SPTcl = unique(SPTcl, keys='SPT_ID', keep='first')  # Keeping Huang entries over Bocquet
SPTcl.sort(keys='SPT_ID')  # Resort by ID.

# Convert masses to [Msun] rather than [Msun/1e14]
SPTcl['M500'] *= 1e14
SPTcl['M500_uerr'] *= 1e14
SPTcl['M500_lerr'] *= 1e14

# Remove any unconfirmed clusters
SPTcl = SPTcl[SPTcl['M500'] > 0.0]

# Run the pipeline.
print('Starting Pipeline.')
pipeline_start_time = time()
spt_sz_selector_start_time = time()

# Initialize the SPT-SZ selector
spt_sz_selector = SelectIRAGN(sextractor_cat_dir=spt_sz_catalog_directory, irac_image_dir=spt_sz_image_directory,
                              region_file_dir=spt_sz_regions_directory, mask_dir=spt_sz_masks_directory,
                              spt_catalog=SPTcl,
                              completeness_file=spt_sz_completeness_sim_results,
                              field_number_dist_file=sdwfs_number_count_dist,
                              sed=polletta_qso2,
                              irac_filter=f'{prefix}Data_Repository/filter_curves/Spitzer_IRAC/080924ch1trans_full.txt',
                              j_band_filter=f'{prefix}Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                                            f'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt')

# Run the SPT-SZ pipeline and store the catalog for later
spt_sz_agn_catalog = spt_sz_selector.run_selection(included_clusters=None,
                                                   excluded_clusters=spt_sz_clusters_to_exclude,
                                                   max_image_catalog_sep=max_separation,
                                                   ch1_min_cov=spt_sz_ch1_min_coverage,
                                                   ch2_min_cov=spt_sz_ch2_min_coverage,
                                                   ch1_bright_mag=ch1_bright_mag,
                                                   ch2_bright_mag=ch2_bright_mag,
                                                   selection_band_faint_mag=ch2_faint_mag,
                                                   ch1_ch2_color=ch1_ch2_color, spt_colnames=spt_column_names,
                                                   output_name=None,
                                                   output_colnames=output_column_names)
print('SPT-SZ selection finished. Run time: {:.2f}s'.format(time() - spt_sz_selector_start_time))
sptpol_selector_start_time = time()

# Initialize the SPTpol 100d selector
sptpol_selector = SelectIRAGN(sextractor_cat_dir=sptpol_catalog_directory, irac_image_dir=sptpol_image_directory,
                              region_file_dir=sptpol_regions_directory, mask_dir=sptpol_masks_directory,
                              spt_catalog=SPTcl,
                              completeness_file=sptpol_completeness_sim_results,
                              field_number_dist_file=sdwfs_number_count_dist,
                              sed=polletta_qso2,
                              irac_filter=f'{prefix}Data_Repository/filter_curves/Spitzer_IRAC/080924ch1trans_full.txt',
                              j_band_filter=f'{prefix}Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                                            f'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt')

# Run the SPTpol pipeline and store the catalog for later
sptpol_agn_catalog = sptpol_selector.run_selection(included_clusters=None,
                                                   excluded_clusters=sptpol_clusters_to_exclude,
                                                   max_image_catalog_sep=max_separation,
                                                   ch1_min_cov=sptpol_ch1_min_coverage,
                                                   ch2_min_cov=sptpol_ch2_min_coverage,
                                                   ch1_bright_mag=ch1_bright_mag,
                                                   ch2_bright_mag=ch2_bright_mag,
                                                   selection_band_faint_mag=ch2_faint_mag,
                                                   ch1_ch2_color=ch1_ch2_color, spt_colnames=spt_column_names,
                                                   output_name=None,
                                                   output_colnames=output_column_names)
print('SPTpol 100d selection finished. Run time: {:.2f}s'.format(time() - sptpol_selector_start_time))
print('Full pipeline finished. Run time: {:.2f}s'.format(time() - pipeline_start_time))

# To merge the catalogs we want to first remove any SPTpol 100d clusters from the SPTpol catalog that either are SPT-SZ
# clusters or have targeted IRAC observations.
spt_sz_cluster_ids = set(spt_sz_agn_catalog.group_by('SPT_ID').groups.keys['SPT_ID'].data)
sptpol_cluster_ids = set(sptpol_agn_catalog.group_by('SPT_ID').groups.keys['SPT_ID'].data)

# Create a list of cluster IDs that only have SSDF data
ssdf_only_cluster_ids = list(sptpol_cluster_ids.difference(spt_sz_cluster_ids))
spt_sz_only_cluster_ids = list(spt_sz_cluster_ids.difference(sptpol_cluster_ids))

# Filter the SPTpol 100d catalog to only include SSDF clusters
sptpol_agn_catalog_ssdf_only = sptpol_agn_catalog[np.in1d(sptpol_agn_catalog['SPT_ID'], ssdf_only_cluster_ids)]
spt_sz_agn_catalog_target_only = spt_sz_agn_catalog[np.in1d(spt_sz_agn_catalog['SPT_ID'], spt_sz_only_cluster_ids)]

# Combine the two cluster catalogs
sptcl_agn_catalog_standard = vstack([spt_sz_agn_catalog, sptpol_agn_catalog_ssdf_only])
sptcl_agn_catalog_inverse = vstack([spt_sz_agn_catalog_target_only, sptpol_agn_catalog])

# Sort by cluster ID
spt_sz_agn_catalog.sort('SPT_ID')
sptpol_agn_catalog.sort('SPT_ID')
sptcl_agn_catalog_standard.sort('SPT_ID')
sptcl_agn_catalog_inverse.sort('SPT_ID')

# Write the catalog(s) to disk
spt_sz_agn_catalog.write(sptsz_output_catalog, overwrite=True)
sptpol_agn_catalog.write(sptpol_output_catalog, overwrite=True)
sptcl_agn_catalog_standard.write(sptcl_std_output_catalog, overwrite=True)
sptcl_agn_catalog_inverse.write(sptcl_inv_output_catalog, overwrite=True)

# List catalog statistics
# SPT-SZ
number_of_clusters_sz = len(spt_sz_agn_catalog.group_by('SPT_ID').groups.keys)
total_number_sz = len(spt_sz_agn_catalog)
total_number_comp_corrected_sz = spt_sz_agn_catalog['COMPLETENESS_CORRECTION'].sum()
total_number_corrected_sz = np.sum(spt_sz_agn_catalog['COMPLETENESS_CORRECTION']
                                   * spt_sz_agn_catalog['SELECTION_MEMBERSHIP'])
number_per_cluster_sz = total_number_corrected_sz / number_of_clusters_sz
median_z_sz = np.median(spt_sz_agn_catalog['REDSHIFT'])
median_m_sz = np.median(spt_sz_agn_catalog['M500'])

# SPTpol
number_of_clusters_pol = len(sptpol_agn_catalog.group_by('SPT_ID').groups.keys)
total_number_pol = len(sptpol_agn_catalog)
total_number_comp_corrected_pol = sptpol_agn_catalog['COMPLETENESS_CORRECTION'].sum()
total_number_corrected_pol = np.sum(sptpol_agn_catalog['COMPLETENESS_CORRECTION']
                                    * sptpol_agn_catalog['SELECTION_MEMBERSHIP'])
number_per_cluster_pol = total_number_corrected_pol / number_of_clusters_pol
median_z_pol = np.median(sptpol_agn_catalog['REDSHIFT'])
median_m_pol = np.median(sptpol_agn_catalog['M500'])


# SPTcl (standard)
number_of_clusters_cl_std = len(sptcl_agn_catalog_standard.group_by('SPT_ID').groups.keys)
total_number_cl_std = len(sptcl_agn_catalog_standard)
total_number_comp_corrected_cl_std = sptcl_agn_catalog_standard['COMPLETENESS_CORRECTION'].sum()
total_number_corrected_cl_std = np.sum(sptcl_agn_catalog_standard['COMPLETENESS_CORRECTION']
                                       * sptcl_agn_catalog_standard['SELECTION_MEMBERSHIP'])
number_per_cluster_cl_std = total_number_corrected_cl_std / number_of_clusters_cl_std
median_z_cl_std = np.median(sptcl_agn_catalog_standard['REDSHIFT'])
median_m_cl_std = np.median(sptcl_agn_catalog_standard['M500'])

# SPTcl (inverse)
number_of_clusters_cl_inv = len(sptcl_agn_catalog_inverse.group_by('SPT_ID').groups.keys)
total_number_cl_inv = len(sptcl_agn_catalog_inverse)
total_number_comp_corrected_cl_inv = sptcl_agn_catalog_inverse['COMPLETENESS_CORRECTION'].sum()
total_number_corrected_cl_inv = np.sum(sptcl_agn_catalog_inverse['COMPLETENESS_CORRECTION']
                                       * sptcl_agn_catalog_inverse['SELECTION_MEMBERSHIP'])
number_per_cluster_cl_inv = total_number_corrected_cl_inv / number_of_clusters_cl_inv
median_z_cl_inv = np.median(sptcl_agn_catalog_inverse['REDSHIFT'])
median_m_cl_inv = np.median(sptcl_agn_catalog_inverse['M500'])

print(f"""SPT-SZ
Number of clusters:\t{number_of_clusters_sz}
Objects selected:\t{total_number_sz}
Objects selected (completeness corrected):\t{total_number_comp_corrected_sz:.2f}
Objects selected (comp + membership corrected):\t{total_number_corrected_sz:.2f}
Objects per cluster (comp + mem corrected):\t{number_per_cluster_sz:.2f}
Median Redshift:\t{median_z_sz:.2f}
Median Mass:\t{median_m_sz:.2e}
---------------------------
SPTpol 100d
Number of clusters:\t{number_of_clusters_pol}
Objects selected:\t{total_number_pol}
Objects selected (completeness corrected):\t{total_number_comp_corrected_pol:.2f}
Objects selected (comp + membership corrected):\t{total_number_corrected_pol:.2f}
Objects per cluster (comp + mem corrected):\t{number_per_cluster_pol:.2f}
Median Redshift:\t{median_z_pol:.2f}
Median Mass:\t{median_m_pol:.2e}
---------------------------
SPTcl (standard)
Number of clusters:\t{number_of_clusters_cl_std}
Objects selected:\t{total_number_cl_std}
Objects selected (completeness corrected):\t{total_number_comp_corrected_cl_std:.2f}
Objects selected (comp + membership corrected):\t{total_number_corrected_cl_std:.2f}
Objects per cluster (corrected):\t{number_per_cluster_cl_std:.2f}
Median Redshift:\t{median_z_cl_std:.2f}
Median Mass:\t{median_m_cl_std:.2e}
---------------------------
SPTcl (inverse)
Number of clusters:\t{number_of_clusters_cl_inv}
Objects selected:\t{total_number_cl_inv}
Objects selected (completeness corrected):\t{total_number_comp_corrected_cl_inv:.2f}
Objects selected (comp + membership corrected):\t{total_number_corrected_cl_inv:.2f}
Objects per cluster (corrected):\t{number_per_cluster_cl_inv:.2f}
Median Redshift:\t{median_z_cl_inv:.2f}
Median Mass:\t{median_m_cl_inv:.2e}""")
