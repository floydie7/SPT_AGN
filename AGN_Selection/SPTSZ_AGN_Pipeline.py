"""
SPT_AGN_Pipeline.py
Author: Benjamin Floyd
This script executes the pipeline consisting of functions from Pipeline_functions.py.
"""
from time import time

import astropy.units as u
import numpy as np
from Pipeline_functions import SelectIRAGN
from astropy.table import Table

# Define directories
# prefix = '/Users/btfkwd/Documents/SPT_AGN/'
prefix = '/home/ben/PycharmProjects/SPT_AGN/'
catalog_directory = f'{prefix}Data/Catalogs'
image_directory = f'{prefix}Data/Images'
regions_directory = f'{prefix}Data/Regions'
masks_directory = f'{prefix}Data/Masks'

# Completeness simulation results file
completeness_sim_results = f'{prefix}Data/Comp_Sim/Results/SPTSZ_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'

# Clusters to manually exclude
clusters_to_exclude = {'SPT-CLJ0045-5757', 'SPT-CLJ0201-6051', 'SPT-CLJ0230-4427', 'SPT-CLJ0456-5623',
                       'SPT-CLJ0646-6236', 'SPT-CLJ2017-5936', 'SPT-CLJ2133-5410', 'SPT-CLJ2138-6317',
                       'SPT-CLJ2232-6151', 'SPT-CLJ2332-5358', 'SPT-CLJ2341-5726'}

# Maximum separation allowed between the cluster images' reference pixel and the SPT SZ centers
max_separation = 1.0 * u.arcmin

# Minimum coverage in 3.6 and 4.5 um bands allowed for good pixel map
ch1_min_coverage = 4
ch2_min_coverage = 4

# Photometric selection cuts
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.46  # Faint-end 4.5 um magnitude
ch1_ch2_color = 0.7  # Minimum [3.6] - [4.5] color

# Additional columns to include in catalog from SPT catalog
spt_column_names = ['REDSHIFT', 'REDSHIFT_UNC', 'M500', 'M500_uerr', 'M500_lerr']

# Output catalog file name
output_catalog = f'{prefix}Data/Output/SPTSZ_IRAGN.fits'

# Requested columns for output catalog
output_column_names = ['SPT_ID', 'SZ_RA', 'SZ_DEC', 'ALPHA_J2000', 'DELTA_J2000', 'RADIAL_SEP_ARCMIN',
                       'REDSHIFT', 'REDSHIFT_UNC', 'M500', 'M500_uerr', 'M500_lerr', 'R500', 'RADIAL_SEP_R500',
                       'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I1_FLUX_APER4', 'I1_FLUXERR_APER4', 'I2_MAG_APER4',
                       'I2_MAGERR_APER4', 'I2_FLUX_APER4', 'I2_FLUXERR_APER4', 'COMPLETENESS_CORRECTION']

# Read in SPT cluster catalog and convert masses to [Msun] rather than [Msun/1e14]
Bocquet = Table.read(f'{prefix}Data/2500d_cluster_sample_Bocquet18.fits')
# Bocquet = Bocquet[Bocquet['M500'] != 0.0]  # Remove unconfirmed clusters
Bocquet['M500'] *= 1e14
Bocquet['M500_uerr'] *= 1e14
Bocquet['M500_lerr'] *= 1e14

# # For the 20 common clusters between SPT-SZ 2500d and SPTpol 100d surveys we want to update the cluster information from
# # the more recent survey. Thus, we will merge the SPT-SZ and SPTpol catalogs together.
# Huang = Table.read(f'{prefix}Data/sptpol100d_catalog_huang19.fits')
#
# # First we need to rename several columns in the SPTpol 100d catalog to match the format of the SPT-SZ catalog
# Huang.rename_columns(['Dec', 'xi', 'theta_core', 'redshift', 'redshift_unc'],
#                      ['DEC', 'XI', 'THETA_CORE', 'REDSHIFT', 'REDSHIFT_UNC'])
#
# # Now, merge the two catalogs
# SPTcl = join(Bocquet, Huang, join_type='outer')
# SPTcl.sort(keys=['SPT_ID', 'field'])  # Sub-sorting by 'field' puts Huang entries first
# SPTcl = unique(SPTcl, keys='SPT_ID', keep='first')  # Keeping Huang entries over Bocquet
# SPTcl.sort(keys='SPT_ID')  # Resort by ID.

# Remove any unconfirmed clusters
SPTcl = Bocquet
SPTcl = SPTcl[SPTcl['M500'] > 0.0]

# Run the pipeline.
print('Starting Pipeline.')
start_time = time()
# Initialize the selector
selector = SelectIRAGN(sextractor_cat_dir=catalog_directory, irac_image_dir=image_directory,
                       region_file_dir=regions_directory, mask_dir=masks_directory, spt_catalog=SPTcl,
                       completeness_file=completeness_sim_results)

# Run the pipeline
selector.run_selection(excluded_clusters=clusters_to_exclude, max_image_catalog_sep=max_separation,
                       ch1_min_cov=ch1_min_coverage, ch2_min_cov=ch2_min_coverage, ch1_bright_mag=ch1_bright_mag,
                       ch2_bright_mag=ch2_bright_mag, selection_band_faint_mag=ch2_faint_mag,
                       ch1_ch2_color=ch1_ch2_color, spt_colnames=spt_column_names, output_name=output_catalog,
                       output_colnames=output_column_names)
print('Pipeline finished. Run time: {:.2f}s'.format(time() - start_time))

# Read in new output catalog
AGN_catalog = Table.read(output_catalog)
AGN_catalog_grp = AGN_catalog.group_by('SPT_ID')
number_of_clusters = len(AGN_catalog_grp.groups.keys)
total_number = len(AGN_catalog)
total_number_corrected = AGN_catalog['COMPLETENESS_CORRECTION'].sum()
number_per_cluster = total_number_corrected / number_of_clusters
median_z = np.median(AGN_catalog['REDSHIFT'])
median_m = np.median(AGN_catalog['M500'])

print("""Number of clusters:\t{num_cl}
Objects selected:\t{num_objs}
Objects selected (completeness corrected):\t{num_comp:.2f}
Objects per cluster (corrected):\t{num_per:.2f}
Median Redshift:\t{z:.2f}
Median Mass:\t{m:.2e}""".format(num_cl=number_of_clusters, num_objs=total_number, num_comp=total_number_corrected,
                                num_per=number_per_cluster, z=median_z, m=median_m))
