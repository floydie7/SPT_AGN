"""
SDWFS_background_WISE_AGN_gal_scaling.py
Author: Benjamin Floyd

Measures the galaxy and AGN number-count distributions to produce a calibrated scaling factor between the two.
"""

import json

import astropy.units as u
import numpy as np
from astro_compendium.utils.json_helpers import NumpyArrayEncoder
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import interp1d

# Read in the WISE catalog
wise_catalog = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/SDWFS_catWISE.ecsv')

# We first need to select for the IR-bright AGN in the field
# Select objects within our magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

wise_catalog = wise_catalog[(ch1_bright_mag < wise_catalog['w1mpro']) & (wise_catalog['w1mpro'] <= ch1_faint_mag) &
                            (ch2_bright_mag < wise_catalog['w2mpro']) & (wise_catalog['w2mpro'] <= ch2_faint_mag)]

# Filter the objects to the SDWFS footprint
mask_img, mask_hdr = fits.getdata('Data_Repository/Project_Data/SPT-IRAGN/Masks/SDWFS/'
                                  'SDWFS_full-field_cov_mask11_11.fits', header=True)
mask_wcs = WCS(mask_hdr)

# Determine the area of the mask
sdwfs_area = np.count_nonzero(mask_img) * mask_wcs.proj_plane_pixel_area()

# Convert the mask image into a boolean mask
mask_img = mask_img.astype(bool)

xy_coords = np.array(mask_wcs.world_to_array_index(SkyCoord(wise_catalog['ra'], wise_catalog['dec'], unit=u.deg)))

wise_catalog = wise_catalog[mask_img[*xy_coords]]

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
color_thresholds = sdwfs_purity_data['purity_90_colors']
agn_purity_color = interp1d(z_bins, color_thresholds, kind='previous')

# Further filter the catalog to objects within a magnitude range for fitting the number count distributions
# Magnitude cuts
bright_end_cut = 14.00
faint_end_cut = 16.25

# Filter the catalog within the magnitude range
wise_catalog = wise_catalog[(bright_end_cut < wise_catalog['w2mpro']) & (wise_catalog['w2mpro'] <= faint_end_cut)]

# Bin into 0.25 magnitude bins
mag_bin_width = 0.25
num_counts_mag_bins = np.arange(bright_end_cut, faint_end_cut, mag_bin_width)

f_agn = {}
for color_threshold in color_thresholds:
    wise_agn = wise_catalog[wise_catalog['w1mpro'] - wise_catalog['w2mpro'] >= color_threshold]
    # Create AGN histogram
    dn_dm_agn, _ = np.histogram(wise_agn['w2mpro'], bins=num_counts_mag_bins)
    dn_dm_agn_weighted = dn_dm_agn / (sdwfs_area.value * mag_bin_width)

    # Create galaxy histogram
    dn_dm_gal, _ = np.histogram(wise_catalog['w2mpro'], bins=num_counts_mag_bins)
    dn_dm_gal_weighted = dn_dm_gal / (sdwfs_area.value * mag_bin_width)

    # Compute the AGN fractions
    f_agn[f'{color_threshold:.2f}'] = dn_dm_agn_weighted / dn_dm_gal_weighted

# Save the data to file
with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/model_fits/SDWFS/SDWFS-WISE_fAGN.json', 'w') as f:
    json.dump(f_agn, f, cls=NumpyArrayEncoder)
