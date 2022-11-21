"""
SPTcl_AGN_Pipeline.py
Author: Benjamin Floyd

A unified IR-AGN selection pipeline for both the SPT-SZ + SPTpol 100d surveys.
"""
import json
from time import time

import astropy.units as u
import numpy as np
from astropy.table import Table
from synphot import SourceSpectrum, units

from AGN_Selection.Pipeline_functions import SelectSDWFS

# Define directories
# prefix = '/Users/btfkwd/Documents/SDWFS/'
prefix = '/home/ben-work/PycharmProjects/SPT_AGN/'

# SPT-SZ Directories
sdwfs_catalog_directory = f'{prefix}Data_Repository/Catalogs/Bootes/SDWFS/Cutouts'
sdwfs_image_directory = f'{prefix}Data_Repository/Images/Bootes/SDWFS/Cutouts'
sdwfs_regions_directory = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Regions/SDWFS'
sdwfs_masks_directory = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Masks/SDWFS'

# Completeness simulation results files
sdwfs_completeness_sim_results = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SDWFS/Results/' \
                                 f'I2_results_gaussian_fwhm172_corr005_mag02_final.json'

# Master cutout catalog (mimicking an SPT cluster catalog)
sdfws_master_cutout_catalog = Table.read(f'{prefix}Data_Repository/Catalogs/Bootes/SDWFS/Cutouts/'
                                         f'SDWFS_Cutout_master.fits')

# SDWFS number count distribution file (for purification)
sdwfs_number_count_dist = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/' \
                          f'SDWFS_number_count_distribution_normed.json'

# SDWFS 90% AGN purity color-redshift file (for color selection thresholds)
sdwfs_purity_color_threshold = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/' \
                               f'SDWFS_purity_color_4.5_17.48.json'

# Polletta QSO2 SED used for computing the J-band absolute magnitudes
polletta_qso2 = SourceSpectrum.from_file(f'{prefix}Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                         wave_unit=u.Angstrom, flux_unit=units.FLAM)

# Clusters to manually exclude
sdwfs_cutouts_to_exclude = {'SDWFS_cutout_026', 'SDWFS_cutout_028', 'SDWFS_cutout_046', 'SDWFS_cutout_049',
                            'SDWFS_cutout_062'}

# Maximum separation allowed between the cluster images' reference pixel and the SPT SZ centers
max_separation = 1.0 * u.arcsec

# Minimum coverage in 3.6 and 4.5 um bands allowed for good pixel map.
# SDWFS uses a minimum of 11 exposures
sdwfs_ch1_min_coverage = 11
sdwfs_ch2_min_coverage = 11

# Photometric selection cuts
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3   # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude
# ch1_ch2_color = 0.7  # Minimum [3.6] - [4.5] color

# Output catalog file name
output_catalog = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits'

# Get the color thresholds from the file
with open(sdwfs_purity_color_threshold, 'r') as f:
    color_threshold_data = json.load(f)
    color_thresholds = color_threshold_data['purity_90_colors']

# Requested columns for output catalog
output_column_names = ['SPT_ID', 'SZ_RA', 'SZ_DEC', 'SDWFS_ID', 'ALPHA_J2000', 'DELTA_J2000', 'RADIAL_SEP_ARCMIN',
                       'I1_FLUX_APER4', 'I2_FLUX_APER4', 'I3_FLUX_APER4', 'I4_FLUX_APER4',
                       'I1_FLUXERR_APER4', 'I2_FLUXERR_APER4', 'I3_FLUXERR_APER4', 'I4_FLUXERR_APER4',
                       'I1_MAG_APER4', 'I2_MAG_APER4', 'I3_MAG_APER4', 'I4_MAG_APER4',
                       'I1_MAGERR_APER4', 'I2_MAGERR_APER4', 'I3_MAGERR_APER4', 'I4_MAGERR_APER4',
                       'REDSHIFT', 'J_ABS_MAG',
                       'COMPLETENESS_CORRECTION',
                       *[f'SELECTION_MEMBERSHIP_{thresh:.2f}' for thresh in color_thresholds], 'MASK_NAME']

# Run the pipeline.
print('Starting Pipeline.')
pipeline_start_time = time()

# Initialize the SDWFS AGN selector
sdwfs_selector = SelectSDWFS(sextractor_cat_dir=sdwfs_catalog_directory, irac_image_dir=sdwfs_image_directory,
                             region_file_dir=sdwfs_regions_directory, mask_dir=sdwfs_masks_directory,
                             sdwfs_master_catalog=sdfws_master_cutout_catalog,
                             completeness_file=sdwfs_completeness_sim_results,
                             field_number_dist_file=sdwfs_number_count_dist,
                             sed=polletta_qso2,
                             irac_filter=f'{prefix}Data_Repository/filter_curves/Spitzer_IRAC/080924ch1trans_full.txt',
                             j_band_filter=f'{prefix}Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                                           f'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt')

# Run the SDWFS pipeline and return the catalog
sdwfs_agn_catalog = sdwfs_selector.run_selection(included_clusters=None,
                                                 excluded_clusters=sdwfs_cutouts_to_exclude,
                                                 max_image_catalog_sep=max_separation,
                                                 ch1_min_cov=sdwfs_ch1_min_coverage,
                                                 ch2_min_cov=sdwfs_ch2_min_coverage,
                                                 ch1_bright_mag=ch1_bright_mag,
                                                 ch2_bright_mag=ch2_bright_mag,
                                                 ch1_faint_mag=ch1_faint_mag,
                                                 selection_band_faint_mag=ch2_faint_mag,
                                                 ch1_ch2_color=color_thresholds,
                                                 spt_colnames=None,
                                                 output_colnames=output_column_names,
                                                 output_name=None)
sdwfs_agn_catalog.rename_column('SPT_ID', 'CUTOUT_ID')
sdwfs_agn_catalog.write(output_catalog, overwrite=True)

print('Full pipeline finished. Run time: {:.2f}s'.format(time() - pipeline_start_time))

# List catalog statistics
# SDWFS
for selection_membership_key in [colname for colname in sdwfs_agn_catalog.colnames if
                                 'SELECTION_MEMBERSHIP' in colname]:
    number_of_cutouts_sdwfs = len(sdwfs_agn_catalog.group_by('CUTOUT_ID').groups.keys)
    total_number_sdwfs = len(sdwfs_agn_catalog)
    total_number_comp_corrected_sdwfs = sdwfs_agn_catalog['COMPLETENESS_CORRECTION'].sum()
    total_number_corrected_sdwfs = np.sum(sdwfs_agn_catalog['COMPLETENESS_CORRECTION']
                                          * sdwfs_agn_catalog[selection_membership_key])
    number_per_cutout_sdwfs = total_number_corrected_sdwfs / number_of_cutouts_sdwfs

    print(f"""SDWFS ({selection_membership_key[-4:]})
    Number of cutouts:\t{number_of_cutouts_sdwfs}
    Objects selected:\t{total_number_sdwfs}
    Objects selected (completeness corrected):\t{total_number_corrected_sdwfs:.2f}
    Objects selected (comp + membership corrected):\t{total_number_corrected_sdwfs:.2f}
    Objects per cutout (comp + mem corrected):\t{number_per_cutout_sdwfs:.2f}""")
