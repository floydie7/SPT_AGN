"""
SPTPol_AGN_Pipeline.py
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
catalog_directory = f'{prefix}Data/SPTPol/catalogs/cluster_cutouts'
image_directory = f'{prefix}Data/SPTPol/images/cluster_cutouts'
regions_directory = f'{prefix}Data/SPTPol/regions'
masks_directory = f'{prefix}Data/SPTPol/masks'

# Completeness simulation results file
completeness_sim_results = f'{prefix}Data/Comp_Sim/SPTpol/Results/' \
                           'SPTpol_I2_results_gaussian_fwhm2.02_corr-0.11_mag0.2.json'

# Clusters to manually exclude
clusters_to_exclude = {'SPT-CLJ0002-5214', 'SPT-CLJ2341-5640', 'SPT-CLJ2357-5953'}

# Maximum separation allowed between the cluster images' reference pixel and the SPT SZ centers
max_separation = 1.0 * u.arcmin

# Minimum coverage in 3.6 and 4.5 um bands allowed for good pixel map
ch1_min_coverage = 3
ch2_min_coverage = 3

# Photometric selection cuts
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.46  # Faint-end 4.5 um magnitude
ch1_ch2_color = 0.7  # Minimum [3.6] - [4.5] color

# Additional columns to include in catalog from SPT catalog
spt_column_names = ['REDSHIFT', 'REDSHIFT_UNC', 'M500', 'M500_uerr', 'M500_lerr']

# Output catalog file name
output_catalog = f'{prefix}Data/Output/SPTpol_IRAGN.fits'

# Requested columns for output catalog
output_column_names = ['SPT_ID', 'SZ_RA', 'SZ_DEC', 'ALPHA_J2000', 'DELTA_J2000', 'RADIAL_SEP_ARCMIN',
                       'REDSHIFT', 'REDSHIFT_UNC', 'M500', 'M500_uerr', 'M500_lerr', 'R500', 'RADIAL_SEP_R500',
                       'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I1_FLUX_APER4', 'I1_FLUXERR_APER4', 'I2_MAG_APER4',
                       'I2_MAGERR_APER4', 'I2_FLUX_APER4', 'I2_FLUXERR_APER4', 'COMPLETENESS_CORRECTION']

# Read in SPT cluster catalog and convert masses to [Msun] rather than [Msun/1e14]
Huang = Table.read(f'{prefix}Data/sptpol100d_catalog_huang19.fits')
Huang = Huang[Huang['M500'] > 0.0]  # Remove unconfirmed clusters
Huang['M500'] *= 1e14
Huang['M500_uerr'] *= 1e14
Huang['M500_lerr'] *= 1e14

# Because Nick doesn't capitalize some of his columns we need to standardize a few of his column names.
Huang.rename_columns(['Dec', 'xi', 'theta_core', 'redshift', 'redshift_unc'],
                     ['DEC', 'XI', 'THETA_CORE', 'REDSHIFT', 'REDSHIFT_UNC'])

# Run the pipeline.
print('Starting Pipeline.')
start_time = time()
# Initialize the selector
selector = SelectIRAGN(sextractor_cat_dir=catalog_directory, irac_image_dir=image_directory,
                       region_file_dir=regions_directory, mask_dir=masks_directory, spt_catalog=Huang,
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
