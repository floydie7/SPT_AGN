"""
SDWFS-local_bkg_WISE-IRAC_scaling.py
Author: Benjamin Floyd

Calculates the magnitude-dependent scaling factors between the WISE galaxies and the IRAC AGN in the Boötes field.
This replaces the function in `SPT-SDWFS_local_bkg_measurement`.
"""
import json

import astropy.units as u
import numpy as np
from astro_compendium.utils.json_helpers import NumpyArrayEncoder
from astro_compendium.utils.small_poisson import small_poisson
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import interp1d

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Read in the SDWFS IRAC AGN catalog
sdwfs_irac_agn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_full-field_IRAGN.fits')

# Read in the SDWFS WISE galaxy catalog
sdwfs_wise_gal = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/SDWFS_catWISE.ecsv')

# Read in the SDWFS mask image and WCS
sdwfs_mask_img, sdwfs_mask_hdr = fits.getdata('Data_Repository/Project_Data/SPT-IRAGN/Masks/SDWFS/'
                                              'SDWFS_full-field_cov_mask11_11.fits', header=True)
sdwfs_wcs = WCS(sdwfs_mask_hdr)

# Determine the area of the mask
sdwfs_area = np.count_nonzero(sdwfs_mask_img) * sdwfs_wcs.proj_plane_pixel_area()

# Convert the mask image into a boolean mask
sdwfs_mask_img = sdwfs_mask_img.astype(bool)

xy_coords = np.array(sdwfs_wcs.world_to_array_index(SkyCoord(sdwfs_wise_gal['ra'], sdwfs_wise_gal['dec'],
                                                             unit=u.deg)))

# Filter the WISE galaxies using the mask
sdwfs_wise_gal = sdwfs_wise_gal[sdwfs_mask_img[*xy_coords]]

# Select objects within our magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

sdwfs_irac_agn = sdwfs_irac_agn[(ch1_bright_mag < sdwfs_irac_agn['I1_MAG_APER4']) &
                                (sdwfs_irac_agn['I1_MAG_APER4'] <= ch1_faint_mag) &
                                (ch2_bright_mag < sdwfs_irac_agn['I2_MAG_APER4']) &
                                (sdwfs_irac_agn['I2_MAG_APER4'] <= ch2_faint_mag)]
sdwfs_wise_gal = sdwfs_wise_gal[(ch1_bright_mag < sdwfs_wise_gal['w1mpro']) &
                                (sdwfs_wise_gal['w1mpro'] <= ch1_faint_mag) &
                                (ch2_bright_mag < sdwfs_wise_gal['w2mpro']) &
                                (sdwfs_wise_gal['w2mpro'] <= ch2_faint_mag)]

mag_bin_width = 0.25
num_counts_mag_bins = np.arange(ch2_bright_mag, ch2_faint_mag, mag_bin_width)

# Create histogram for the WISE galaxies
wise_gal_dn_dm, _ = np.histogram(sdwfs_wise_gal['w2mpro'], bins=num_counts_mag_bins)
wise_gal_dn_dm_weighted = wise_gal_dn_dm / (sdwfs_area.value * mag_bin_width)

# Compute the WISE galaxy errors
wise_gal_dn_dm_err = tuple(err / (sdwfs_area.value * mag_bin_width) for err in small_poisson(wise_gal_dn_dm))[::-1]

# Iterate through the AGN selection thresholds and build the IRAC AGN histograms
sdwfs_irac_agn_dn_dm = {}
selection_membership_columns = [colname for colname in sdwfs_irac_agn.colnames if 'SELECTION_MEMBERSHIP' in colname]
for selection_membership in selection_membership_columns:
    # Make the AGN selection for the color threshold
    irac_agn = sdwfs_irac_agn[sdwfs_irac_agn[selection_membership] >= 0.5]

    # Create the IRAC AGN histogram
    irac_agn_dn_dm, _ = np.histogram(irac_agn['I2_MAG_APER4'], bins=num_counts_mag_bins,
                                     weights=irac_agn['COMPLETENESS_CORRECTION'])
    irac_agn_dn_dm_weighted = irac_agn_dn_dm / (sdwfs_area.value * mag_bin_width)

    # Compute the IRAC AGN errors
    irac_agn_dn_dm_err = tuple(err / (sdwfs_area.value * mag_bin_width)
                               for err in small_poisson(irac_agn_dn_dm))[::-1]

    # Compute the scaling fractions from the WISE galaxy dN/dm to the IRAC AGN dN/dm
    scaling_fract = irac_agn_dn_dm_weighted / wise_gal_dn_dm_weighted
    sdwfs_irac_agn_dn_dm[selection_membership] = {'hist': irac_agn_dn_dm_weighted,
                                                  'err': irac_agn_dn_dm_err,
                                                  'scaling_frac': scaling_fract}

# Write the results to file
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
          'SDWFS_WISEgal-IRACagn_scaling_factors.json', 'w') as f:
    json.dump(sdwfs_irac_agn_dn_dm, f, cls=NumpyArrayEncoder)
