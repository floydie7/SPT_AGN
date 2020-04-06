"""
SPTcl_AGN_Pipeline.py
Author: Benjamin Floyd

A unified IR-AGN selection pipeline for both the SPT-SZ + SPTpol 100d surveys.
"""

from time import time

import astropy.units as u
import numpy as np
from Pipeline_functions import SelectIRAGN
from astropy.table import Table, join, unique, vstack

# Define directories
# prefix = '/Users/btfkwd/Documents/SPT_AGN/'
prefix = '/home/ben/PycharmProjects/SPT_AGN/'

# SPT-SZ Directories
spt_sz_catalog_directory = f'{prefix}Data/Catalogs'
spt_sz_image_directory = f'{prefix}Data/Images'
spt_sz_regions_directory = f'{prefix}Data/Regions'
spt_sz_masks_directory = f'{prefix}Data/Masks'

# SPTpol 100d Directories
sptpol_catalog_directory = f'{prefix}Data/SPTPol/catalogs/cluster_cutouts'
sptpol_image_directory = f'{prefix}Data/SPTPol/images/cluster_cutouts'
sptpol_regions_directory = f'{prefix}Data/SPTPol/regions'
sptpol_masks_directory = f'{prefix}Data/SPTPol/masks'

# Completeness simulation results files
spt_sz_completeness_sim_results = f'{prefix}Data/Comp_Sim/Results/' \
                                  f'SPTSZ_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'
sptpol_completeness_sim_results = f'{prefix}Data/Comp_Sim/SPTpol/Results/' \
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
output_catalog = f'{prefix}Data/Output/SPTcl_IRAGN.fits'

# Requested columns for output catalog
output_column_names = ['SPT_ID', 'SZ_RA', 'SZ_DEC', 'ALPHA_J2000', 'DELTA_J2000', 'RADIAL_SEP_ARCMIN',
                       'REDSHIFT', 'REDSHIFT_UNC', 'M500', 'M500_uerr', 'M500_lerr', 'R500', 'RADIAL_SEP_R500',
                       'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I1_FLUX_APER4', 'I1_FLUXERR_APER4', 'I2_MAG_APER4',
                       'I2_MAGERR_APER4', 'I2_FLUX_APER4', 'I2_FLUXERR_APER4', 'COMPLETENESS_CORRECTION', 'MASK_NAME']

# Read in SPT cluster catalog and convert masses to [Msun] rather than [Msun/1e14]
Bocquet = Table.read(f'{prefix}Data/2500d_cluster_sample_Bocquet18.fits')
Bocquet['M500'] *= 1e14
Bocquet['M500_uerr'] *= 1e14
Bocquet['M500_lerr'] *= 1e14

# For the 20 common clusters between SPT-SZ 2500d and SPTpol 100d surveys we want to update the cluster information from
# the more recent survey. Thus, we will merge the SPT-SZ and SPTpol catalogs together.
Huang = Table.read(f'{prefix}Data/sptpol100d_catalog_huang19.fits')

# First we need to rename several columns in the SPTpol 100d catalog to match the format of the SPT-SZ catalog
Huang.rename_columns(['Dec', 'xi', 'theta_core', 'redshift', 'redshift_unc'],
                     ['DEC', 'XI', 'THETA_CORE', 'REDSHIFT', 'REDSHIFT_UNC'])

# Now, merge the two catalogs
SPTcl = join(Bocquet, Huang, join_type='outer')
SPTcl.sort(keys=['SPT_ID', 'field'])  # Sub-sorting by 'field' puts Huang entries first
SPTcl = unique(SPTcl, keys='SPT_ID', keep='first')  # Keeping Huang entries over Bocquet
SPTcl.sort(keys='SPT_ID')  # Resort by ID.

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
                              completeness_file=spt_sz_completeness_sim_results)

# Run the SPT-SZ pipeline and store the catalog for later
spt_sz_agn_catalog = spt_sz_selector.run_selection(excluded_clusters=spt_sz_clusters_to_exclude,
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
                              completeness_file=sptpol_completeness_sim_results)

# Run the SPTpol pipeline and store the catalog for later
sptpol_agn_catalog = sptpol_selector.run_selection(excluded_clusters=sptpol_clusters_to_exclude,
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

# Filter the SPTpol 100d catalog to only include SSDF clusters
sptpol_agn_catalog = sptpol_agn_catalog[np.in1d(sptpol_agn_catalog['SPT_ID'], ssdf_only_cluster_ids)]

# Combine the two cluster catalogs
sptcl_agn_catalog = vstack([spt_sz_agn_catalog, sptpol_agn_catalog])

# Sort by cluster ID
sptcl_agn_catalog.sort('SPT_ID')

# Write the catalog to disk
sptcl_agn_catalog.write(output_catalog, overwrite=True)

# List catalog statistics
sptcl_agn_catalog_grp = sptcl_agn_catalog.group_by('SPT_ID')
number_of_clusters = len(sptcl_agn_catalog_grp.groups.keys)
total_number = len(sptcl_agn_catalog)
total_number_corrected = sptcl_agn_catalog['COMPLETENESS_CORRECTION'].sum()
number_per_cluster = total_number_corrected / number_of_clusters
median_z = np.median(sptcl_agn_catalog['REDSHIFT'])
median_m = np.median(sptcl_agn_catalog['M500'])

print(f"""Number of clusters:\t{number_of_clusters}
Objects selected:\t{total_number}
Objects selected (completeness corrected):\t{total_number_corrected:.2f}
Objects per cluster (corrected):\t{number_per_cluster:.2f}
Median Redshift:\t{median_z:.2f}
Median Mass:\t{median_m:.2e}""")
