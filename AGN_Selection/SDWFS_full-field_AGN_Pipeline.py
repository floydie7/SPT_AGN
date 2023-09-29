"""
SDWFS_full-field_AGN_Pipeline.py
Author: Benjamin Floyd

Selects for IR-bright AGN over the entire SDWFS field and creates a catalog for use as a reference field for cluster
studies.
"""
import json
from time import time

import astropy.units as u
import numpy as np
from astropy.table import Table
from synphot import SourceSpectrum, units

from AGN_Selection.Pipeline_functions import SelectFullFieldSDWFS

# Define directories
prefix = '/home/ben-work/PycharmProjects/SPT_AGN/'

# SDWFS field files and directories
sdwfs_photometric_catalog = (f'{prefix}Data_Repository/Catalogs/Bootes/SDWFS/'
                             f'ch2v33_sdwfs_2009mar3_apcorr_matched_ap4_Main_v0.4.cat.gz')
sdwfs_images = [
    f'{prefix}Data_Repository/Images/Bootes/SDWFS/I1_bootes.v32.fits',
    f'{prefix}Data_Repository/Images/Bootes/SDWFS/I1_bootes.cov.fits.gz',
    f'{prefix}Data_Repository/Images/Bootes/SDWFS/I2_bootes.v32.fits',
    f'{prefix}Data_Repository/Images/Bootes/SDWFS/I2_bootes.cov.fits.gz'
]
sdwfs_object_mask = f'{prefix}Data_Repository/Images/Bootes/SDWFS/SDWFS_full_field_object_mask.fits.gz'
sdwfs_mask_dir = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Masks/SDWFS'

# For the photometric catalog we need to specify the column names
sdwfs_photometric_catalog_names = ['ID', 'ALPHA_J2000', 'DELTA_J2000',
                                   'B_APFLUX4', 'R_APFLUX4', 'I_APFLUX4',
                                   'B_APFLUXERR4', 'R_APFLUXERR4', 'I_APFLUXERR4',
                                   'B_APMAG4', 'R_APMAG4', 'I_APMAG4',
                                   'B_APMAGERR4', 'R_APMAGERR4', 'I_APMAGERR4',
                                   'I1_FLUX_APER4', 'I2_FLUX_APER4', 'I3_FLUX_APER4', 'I4_FLUX_APER4',
                                   'I1_FLUXERR_APER4', 'I2_FLUXERR_APER4', 'I3_FLUXERR_APER4', 'I4_FLUXERR_APER4',
                                   'I1_FLUX_APER4_BROWN', 'I2_FLUX_APER4_BROWN', 'I3_FLUX_APER4_BROWN',
                                   'I4_FLUX_APER4_BROWN',
                                   'I1_MAG_APER4', 'I2_MAG_APER4', 'I3_MAG_APER4', 'I4_MAG_APER4',
                                   'I1_MAGERR_APER4', 'I2_MAGERR_APER4', 'I3_MAGERR_APER4', 'I4_MAGERR_APER4',
                                   'I1_MAGERR_APER4_BROWN', 'I2_MAGERR_APER4_BROWN', 'I3_MAGERR_APER4_BROWN',
                                   'I4_MAGERR_APER4_BROWN',
                                   'STARS_COLOR', 'STARS_MORPH', 'CLASS_STAR', 'MBZ_FLAG_4_4_4']

# Completeness simulation results files
sdwfs_completeness_sim_results = (f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Comp_Sim/SDWFS/Results/'
                                  f'I2_full-field_SDWFS_Brodwin.json')

# Read in the SDWFS Photo-z catalog
sdwfs_photo_z_catalog = Table.read(f'{prefix}Data_Repository/Catalogs/Bootes/SDWFS/mbz_v0.06_prior_bri12_18p8.cat.gz',
                                   names=['ID', 'REDSHIFT', 'col3', 'col4', 'col5', 'col6', 'col7'],
                                   format='ascii',
                                   include_names=['ID', 'REDSHIFT'])

# SDWFS 90% AGN purity color-redshift file (for color selection thresholds)
sdwfs_purity_color_threshold = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/' \
                               f'SDWFS_purity_color_4.5_17.48.json'

# Polletta QSO2 SED used for computing the J-band absolute magnitudes
polletta_qso2 = SourceSpectrum.from_file(f'{prefix}Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                         wave_unit=u.Angstrom, flux_unit=units.FLAM)

# Filter file names
irac_36um_filter = f'{prefix}Data_Repository/filter_curves/Spitzer_IRAC/080924ch1trans_full.txt'
flamingos_j_filter = (f'{prefix}Data_Repository/filter_curves/KPNO/KPNO_2.1m/FLAMINGOS/'
                      f'FLAMINGOS.BARR.J.MAN240.ColdWitness.txt')

# Minimum coverage in 3.6 and 4.5 um bands allowed for good pixel map.
# SDWFS uses a minimum of 11 exposures
sdwfs_ch1_min_coverage = 11
sdwfs_ch2_min_coverage = 11

# Photometric selection cuts
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

# Output catalog file name
output_catalog = f'{prefix}Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_full-field_IRAGN.fits'

# Get the color thresholds from the file
with open(sdwfs_purity_color_threshold, 'r') as f:
    color_threshold_data = json.load(f)
    color_thresholds = color_threshold_data['purity_90_colors']

# Requested columns for output catalog
output_column_names = ['ID', 'ALPHA_J2000', 'DELTA_J2000',
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

sdwfs_selector = SelectFullFieldSDWFS(sextractor_cat=sdwfs_photometric_catalog, irac_images=sdwfs_images,
                                      object_mask=sdwfs_object_mask, mask_dir=sdwfs_mask_dir,
                                      photoz_catalog=sdwfs_photo_z_catalog,
                                      completeness_file=sdwfs_completeness_sim_results,
                                      purity_color_threshold_file=sdwfs_purity_color_threshold, sed=polletta_qso2,
                                      irac_filter=irac_36um_filter, j_band_filter=flamingos_j_filter)
sdwfs_agn_catalog = sdwfs_selector.run_selection(ch1_min_cov=sdwfs_ch1_min_coverage,
                                                 ch2_min_cov=sdwfs_ch2_min_coverage,
                                                 ch1_bright_mag=ch1_bright_mag,
                                                 ch2_bright_mag=ch2_bright_mag,
                                                 ch1_faint_mag=ch1_faint_mag,
                                                 selection_band_faint_mag=ch2_faint_mag,
                                                 ch1_ch2_color=color_thresholds,
                                                 photo_cat_colnames=sdwfs_photometric_catalog_names,
                                                 output_name=None,
                                                 output_colnames=output_column_names)
sdwfs_agn_catalog.write(output_catalog, overwrite=True)

print('Full pipeline finished. Run time: {:.2f}s'.format(time() - pipeline_start_time))

# List catalog statistics
# SDWFS
for selection_membership_key in [colname for colname in sdwfs_agn_catalog.colnames if
                                 'SELECTION_MEMBERSHIP' in colname]:
    total_number_sdwfs = len(sdwfs_agn_catalog)
    total_number_comp_corrected_sdwfs = sdwfs_agn_catalog['COMPLETENESS_CORRECTION'].sum()
    total_number_corrected_sdwfs = np.sum(sdwfs_agn_catalog['COMPLETENESS_CORRECTION']
                                          * sdwfs_agn_catalog[selection_membership_key])

    print(f"""SDWFS ({selection_membership_key[-4:]})
    Objects selected:\t{total_number_sdwfs}
    Objects selected (completeness corrected):\t{total_number_corrected_sdwfs:.2f}
    Objects selected (comp + membership corrected):\t{total_number_corrected_sdwfs:.2f}""")
