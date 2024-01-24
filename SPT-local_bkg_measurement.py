"""
SPT-local_bkg_measurement.py
Author: Benjamin Floyd

Uses the WISE galaxy catalogs centered around the SPT cluster coordinates and the WISE galaxy--IRAC AGN magnitude
dependent scaling factors measured in the SDWFS survey in the Boötes field to convert the native galaxy number count
distributions into scaled AGN number count distributions. A true AGN number count distribution from SDWFS is then scaled
to match the local background expectation. This distribution is then integrated over the selection band range of
magnitudes to find the background estimation of the AGN surface density. The background estimation will be used in
broader analysis of the AGN populations along the line-of-sight to the SPT clusters.

This script replaces the functionalities originally in `SPT-SDWFS_local_bkg_measurement`.
"""
import glob
import json
import re
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astro_compendium.utils.json_helpers import NumpyArrayEncoder
from astro_compendium.utils.small_poisson import small_poisson
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.wcs import WCS
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from tqdm import tqdm

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
cluster_id = re.compile(r'SPT-CLJ\d+-\d+')

# Set our selection magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

# Set our correction factors needed to be able to use IRAC magnitude cuts
w1_correction = -0.11 * u.mag
w2_correction = -0.07 * u.mag

# Set our magnitude binning
mag_bin_width = 0.25
magnitude_bins = np.arange(ch2_bright_mag, ch2_faint_mag, mag_bin_width)
magnitude_bin_centers = magnitude_bins[:-1] + np.diff(magnitude_bins) / 2

# Set our narrow magnitude ranges
# (For fine-tuning scaling of SDWFS IRAC AGN to SPT WISE scaled "AGN")
narrow_bright_mag = 15.
narrow_faint_mag = 17.
narrow_magnitude_filter = (narrow_bright_mag < magnitude_bin_centers) & (magnitude_bin_centers <= narrow_faint_mag)

# Set our background annulus ranges in terms of r200 radii
inner_radius_factor = 3
outer_radius_factor = 7


@dataclass
class ClusterInfo:
    catalog: QTable
    annulus_area: u.Quantity
    sdwfs_irac_dndm: np.ndarray = None
    sdwfs_irac_dndm_err: np.ndarray = None
    sdwfs_irac_dndm_scaled: np.ndarray = None
    wise_scaled_dndm: np.ndarray = None
    wise_scaled_dndm_err: np.ndarray = None


# Read in the annulus information that we previously calculated
with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/SPTcl-local_bkg_annulus.json', 'r') as f:
    spt_wise_annuli_data = json.load(f)

# Read in and process the SPT WISE galaxy catalogs
spt_wise_gal_data = {}
catalog_names = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/*_wise_local_bkg.ecsv')
for catalog_name in tqdm(catalog_names, desc='Processing SPT WISE Catalogs'):
    cluster_name = cluster_id.search(catalog_name).group(0)
    spt_wise_gal = QTable.read(catalog_name)

    # Apply photometric correction factors
    spt_wise_gal['w1mpro'] = spt_wise_gal['w1mpro'] + w1_correction
    spt_wise_gal['w2mpro'] = spt_wise_gal['w2mpro'] + w2_correction

    # Select only objects within the magnitude ranges
    spt_wise_gal = spt_wise_gal[(ch1_bright_mag < spt_wise_gal['w1mpro'].value) &
                                (spt_wise_gal['w1mpro'].value <= ch1_faint_mag) &
                                (ch2_bright_mag < spt_wise_gal['w2mpro'].value) &
                                (spt_wise_gal['w2mpro'].value <= ch2_faint_mag)]

    # Excise the cluster and only select objects in our chosen annulus
    spt_wise_gal_cluster_coord = SkyCoord(spt_wise_gal['SZ_RA'][0], spt_wise_gal['SZ_DEC'][0], unit=u.deg)
    spt_wise_gal_coords = SkyCoord(spt_wise_gal['ra'], spt_wise_gal['dec'], unit=u.deg)
    spt_wise_gal_sep_deg = spt_wise_gal_cluster_coord.separation(spt_wise_gal_coords)

    # Retrieve the annulus radii and area
    inner_radius_deg = spt_wise_annuli_data[cluster_name]['inner_radius_deg'] * u.deg
    outer_radius_deg = spt_wise_annuli_data[cluster_name]['outer_radius_deg'] * u.deg
    spt_bkg_area = spt_wise_annuli_data[cluster_name]['annulus_area'] * u.deg ** 2

    # Select for the objects within the background annulus
    spt_wise_gal = spt_wise_gal[(inner_radius_deg < spt_wise_gal_sep_deg) & (spt_wise_gal_sep_deg <= outer_radius_deg)]

    spt_wise_gal_data[cluster_name] = ClusterInfo(catalog=spt_wise_gal, annulus_area=spt_bkg_area)

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

# Select the WISE galaxies within our selection magnitudes
sdwfs_wise_gal = sdwfs_wise_gal[(ch1_bright_mag < sdwfs_wise_gal['w1mpro']) &
                                (sdwfs_wise_gal['w1mpro'] <= ch1_faint_mag) &
                                (ch2_bright_mag < sdwfs_wise_gal['w2mpro']) &
                                (sdwfs_wise_gal['w2mpro'] <= ch2_faint_mag)]

# Create histogram for the WISE galaxies
sdwfs_wise_gal_dn_dm, _ = np.histogram(sdwfs_wise_gal['w2mpro'], bins=magnitude_bins)
sdwfs_wise_gal_dn_dm_weighted = sdwfs_wise_gal_dn_dm / (sdwfs_area.value * mag_bin_width)

# Compute the WISE galaxy errors
wise_gal_dn_dm_err = tuple(err / (sdwfs_area.value * mag_bin_width)
                           for err in small_poisson(sdwfs_wise_gal_dn_dm))[::-1]

# Read in the SDWFS WISE galaxy--IRAC AGN scaling factor data
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
          'SDWFS_WISEgal-IRACagn_scaling_factors.json', 'r') as f:
    sdwfs_wise_irac_scaling_data = json.load(f)
for d in sdwfs_wise_irac_scaling_data.values():
    for k, v in d.items():
        d[k] = np.array(v)

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Compute the number count distribution of the WISE galaxies in the SPT background annulus
local_wise_sdwfs_wise_scaling = {}
for cluster_data in tqdm(spt_wise_gal_data.values(), desc='Computing dN/dm distributions'):
    catalog = cluster_data.catalog
    spt_bkg_area = cluster_data.annulus_area

    # Pull the appropriate SDWFS WISE galaxy--SDWFS IRAC AGN scaling factors
    selection_color = agn_purity_color(catalog["REDSHIFT"][0])
    sdwfs_wise_to_irac_scaling_factors = sdwfs_wise_irac_scaling_data[f'SELECTION_MEMBERSHIP_{selection_color:.2f}']['scaling_frac']

    # Calculate our weighting factor
    dndm_weight = spt_bkg_area.value * mag_bin_width

    # Create histogram
    spt_wise_dndm, _ = np.histogram(catalog['w2mpro'].value, bins=magnitude_bins)
    spt_wise_dndm_weighted = spt_wise_dndm / dndm_weight

    # Compute the errors
    spt_wise_dndm_err = tuple(err / dndm_weight for err in small_poisson(spt_wise_dndm))[::-1]

    # Determine the fractional error
    spt_wise_frac_err = spt_wise_dndm_err / spt_wise_dndm_weighted

    # Compute the local WISE galaxy--SDWFS WISE galaxy scaling factors
    local_to_sdwfs_wise_scaling_factors = sdwfs_wise_gal_dn_dm_weighted / spt_wise_dndm_weighted

    # record the scaling factors
    local_wise_sdwfs_wise_scaling[catalog['SPT_ID'][0]] = local_to_sdwfs_wise_scaling_factors

    # Combine the two scaling factors to get the total scaling factor
    total_scaling_factors = local_to_sdwfs_wise_scaling_factors * sdwfs_wise_to_irac_scaling_factors

    # Scale the number count distributions to the IRAC AGN levels
    spt_wise_dndm_scaled = spt_wise_dndm_weighted * total_scaling_factors

    # Fix the errors using a constant fractional error
    spt_wise_dndm_scaled_err = spt_wise_dndm_scaled * spt_wise_frac_err

    # Store the scaled number count distribution and errors for later
    cluster_data.wise_scaled_dndm = spt_wise_dndm_scaled
    cluster_data.wise_scaled_dndm_err = spt_wise_dndm_scaled_err

    # Also store the associated SDWFS number count distribution and errors
    cluster_data.sdwfs_irac_dndm = sdwfs_wise_irac_scaling_data[f'SELECTION_MEMBERSHIP_{selection_color:.2f}']['hist']
    cluster_data.sdwfs_irac_dndm_err = sdwfs_wise_irac_scaling_data[f'SELECTION_MEMBERSHIP_{selection_color:.2f}']['err']

# Store the scaling factors
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
          'SPT_WISEgal-SDWFS_WISEgal_scaling_factors.json', 'w') as f:
    json.dump(local_wise_sdwfs_wise_scaling, f, cls=NumpyArrayEncoder)

# For each cluster, renormalize the SDWFS number count distribution to match the scaled WISE galaxy level
for cluster_data in tqdm(spt_wise_gal_data.values(), desc='Renormalizing SDWFS dN/dm to local levels'):
    # First, we need to only select objects within a narrow range of magnitudes where the data is reliable
    spt_dndm_narrow = cluster_data.wise_scaled_dndm[narrow_magnitude_filter]
    sdwfs_dndm_narrow = cluster_data.sdwfs_irac_dndm[narrow_magnitude_filter]

    # Do the same with the errors
    spt_dndm_err_narrow = cluster_data.wise_scaled_dndm_err[:, narrow_magnitude_filter]
    sdwfs_dndm_err_narrow = cluster_data.sdwfs_irac_dndm_err[:, narrow_magnitude_filter]

    # Symmetrize the errors
    spt_dndm_symerr_narrow = np.sqrt(spt_dndm_err_narrow[0] * spt_dndm_err_narrow[1])
    sdwfs_dndm_symerr_narrow = np.sqrt(sdwfs_dndm_err_narrow[0] * sdwfs_dndm_err_narrow[1])

    # NaNs can show up in errors if the counts in a bin were 0
    # We want to handle this by setting the error to a large number so that it does not contribute to the fit
    np.nan_to_num(spt_dndm_symerr_narrow, nan=1e64, posinf=1e64, neginf=-1e64, copy=False)
    np.nan_to_num(spt_dndm_symerr_narrow, nan=1e64, posinf=1e64, neginf=-1e64, copy=False)

    # For our model fitting we will use errors that combine the two sources together
    combined_errors = np.sqrt(spt_dndm_symerr_narrow ** 2 + sdwfs_dndm_symerr_narrow ** 2)

    # Fit a simple model that relates the SDWFS IRAC and SPT scaled WISE distributions together
    renorm_popt, _ = curve_fit(lambda data, a: a * data, sdwfs_dndm_narrow, spt_dndm_narrow, sigma=combined_errors)

    # Apply the scaling factor to the data
    cluster_data.sdwfs_irac_dndm_scaled = cluster_data.sdwfs_irac_dndm * renorm_popt

# print('Integrating dN/dm')
# For each cluster integrate the scaled SDWFS number count distribution over the full selection magnitude range to find
# the local background AGN surface density estimation
spt_local_bkg_agn_surf_den = {cluster_name: simpson(cluster_data.sdwfs_irac_dndm_scaled, magnitude_bin_centers)
                              for cluster_name, cluster_data in tqdm(spt_wise_gal_data.items(),
                                                                     desc='Integrating scaled SDWFS dN/dm')}

# Write the results to file
with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/SPTcl-local_bkg_frac_err.json', 'w') as f:
    json.dump(spt_local_bkg_agn_surf_den, f, cls=NumpyArrayEncoder)
