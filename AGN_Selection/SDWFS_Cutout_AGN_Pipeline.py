"""
SPTcl_AGN_Pipeline.py
Author: Benjamin Floyd

A unified IR-AGN selection pipeline for both the SPT-SZ + SPTpol 100d surveys.
"""

from time import time

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from AGN_Selection.Pipeline_functions import SelectIRAGN

# Define directories
# prefix = '/Users/btfkwd/Documents/SDWFS/'
prefix = '/home/ben/PycharmProjects/SDWFS/'

# SPT-SZ Directories
sdwfs_catalog_directory = f'{prefix}Data/Data_Repository/Project_Data/SPT-IRAGN/SDWFS_cutouts/Catalogs'
sdwfs_image_directory = f'{prefix}Data/Images/cutouts'
sdwfs_regions_directory = f'{prefix}Data/Regions'
sdwfs_masks_directory = f'{prefix}Data/Masks'

# Completeness simulation results files
sdwfs_completeness_sim_results = f'{prefix}Data/Comp_Sim/Results/' \
                                 f'I2_results_gaussian_fwhm172_corr005_mag02_final.json'

# Master cutout catalog (mimicking an SPT cluster catalog)
sdfws_master_cutout_catalog = Table.read(f'{prefix}Data/Data_Repository/Project_Data/SPT-IRAGN/SDWFS_cutouts/'
                                         'SDWFS_Cutout_master.fits')

# SDWFS number count distribution file (for purification)
sdwfs_number_count_dist = f'{prefix}Data/Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/' \
                          f'SDWFS_number_count_distribution_normed.json'

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
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.46  # Faint-end 4.5 um magnitude
ch1_ch2_color = 0.7  # Minimum [3.6] - [4.5] color

# Output catalog file name
output_catalog = f'{prefix}Data/Output/SDWFS_cutout_IRAGN.fits'

# Requested columns for output catalog
output_column_names = ['Cutout_ID', 'SZ_RA', 'SZ_DEC', 'SDWFS_ID', 'ALPHA_J2000', 'DELTA_J2000',
                       'I1_FLUX_APER4', 'I2_FLUX_APER4', 'I3_FLUX_APER4', 'I4_FLUX_APER4',
                       'I1_FLUXERR_APER4', 'I2_FLUXERR_APER4', 'I3_FLUXERR_APER4', 'I4_FLUXERR_APER4',
                       'I1_MAG_APER4', 'I2_MAG_APER4', 'I3_MAG_APER4', 'I4_MAG_APER4',
                       'I1_MAGERR_APER4', 'I2_MAGERR_APER4', 'I3_MAGERR_APER4', 'I4_MAGERR_APER4',
                       'COMPLETENESS_CORRECTION', 'MASK_NAME']

# Run the pipeline.
print('Starting Pipeline.')
pipeline_start_time = time()

# Initialize the SPT-SZ selector
sdwfs_selector = SelectIRAGN(sextractor_cat_dir=sdwfs_catalog_directory, irac_image_dir=sdwfs_image_directory,
                             region_file_dir=sdwfs_regions_directory, mask_dir=sdwfs_masks_directory,
                             spt_catalog=sdfws_master_cutout_catalog,
                             completeness_file=sdwfs_completeness_sim_results,
                             sed=None, output_filter=None, output_zero_pt=None,
                             field_number_dist_file=sdwfs_number_count_dist)

# Run the SDWFS pipeline manually
sdwfs_selector.file_pairing(exclude=sdwfs_cutouts_to_exclude)
sdwfs_selector.image_to_catalog_match(max_image_catalog_sep=max_separation)
sdwfs_selector.coverage_mask(ch1_min_cov=sdwfs_ch1_min_coverage, ch2_min_cov=sdwfs_ch2_min_coverage)
sdwfs_selector.object_mask()
sdwfs_selector.object_selection(ch1_bright_mag=ch1_bright_mag, ch2_bright_mag=ch2_bright_mag,
                                selection_band_faint_mag=ch2_faint_mag,
                                absolute_mag=None,
                                ch1_ch2_color_cut=ch1_ch2_color)
sdwfs_selector.purify_selection(ch1_ch2_color_cut=ch1_ch2_color)
sdwfs_selector.completeness_value()
sdwfs_iragn_catalog = sdwfs_selector.final_catalogs(catalog_cols=output_column_names)

# Instead of messing with the object separation function in the selector, we will compute the separations in post
radial_sep_arcmin = []
sdwfs_iragn_catalog_grp = sdwfs_iragn_catalog.group_by('Cutout_ID')
for cutout in sdwfs_iragn_catalog_grp.groups:
    center_coord = SkyCoord(cutout['SZ_RA'][0], cutout['SZ_DEC'][0], unit=u.deg)
    object_coords = SkyCoord(cutout['ALPHA_J2000'], cutout['DELTA_J2000'], unit=u.deg)

    # Compute separations
    separations_arcmin = object_coords.separation(center_coord).to(u.arcmin)
    radial_sep_arcmin.append(separations_arcmin)
# Store the separations in the catalog
sdwfs_iragn_catalog['RADIAL_SEP_ARCMIN'] = np.concatenate(radial_sep_arcmin)

# Write it to disk
sdwfs_iragn_catalog.write(output_catalog, overwrite=True)

print('Full pipeline finished. Run time: {:.2f}s'.format(time() - pipeline_start_time))

# List catalog statistics
# SDWFS
number_of_cutouts_sdwfs = len(sdwfs_iragn_catalog_grp.groups.keys)
total_number_sdwfs = len(sdwfs_iragn_catalog_grp)
total_number_corrected_sdwfs = sdwfs_iragn_catalog_grp['COMPLETENESS_CORRECTION'].sum()
number_per_cutout_sdwfs = total_number_corrected_sdwfs / number_of_cutouts_sdwfs

print(f"""SDWFS
Number of clusters:\t{number_of_cutouts_sdwfs}
Objects selected:\t{total_number_sdwfs}
Objects selected (completeness corrected):\t{total_number_corrected_sdwfs:.2f}
Objects per cutout (corrected):\t{number_per_cutout_sdwfs:.2f}""")
