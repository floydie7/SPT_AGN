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
from astropy.table import QTable
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
cluster_id = re.compile(r'SPT-CLJ\d+-\d+')

# Set our selection magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

# Set our magnitude binning
mag_bin_width = 0.25
magnitude_bins = np.arange(ch2_bright_mag, ch2_faint_mag, mag_bin_width)
magnitude_bin_centers = magnitude_bins[:-1] + np.diff(magnitude_bins) / 2

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


# Read in and process the SPT WISE galaxy catalogs
spt_wise_gal_data = {}
catalog_names = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/*_wise_local_bkg.ecsv')
for catalog_name in catalog_names:
    spt_wise_gal = QTable.read(catalog_name)

    # Select only objects within the magnitude ranges
    spt_wise_gal = spt_wise_gal[(ch1_bright_mag < spt_wise_gal['w1mpro'].value) &
                                (spt_wise_gal['w1mpro'].value <= ch1_faint_mag) &
                                (ch2_bright_mag < spt_wise_gal['w2mpro'].value) &
                                (spt_wise_gal['w2mpro'].value <= ch2_faint_mag)]

    # Excise the cluster and only select objects in our chosen annulus
    spt_wise_gal_cluster_coord = SkyCoord(spt_wise_gal['SZ_RA'][0], spt_wise_gal['SZ_DEC'][0], unit=u.deg)
    spt_wise_gal_coords = SkyCoord(spt_wise_gal['ra'], spt_wise_gal['dec'], unit=u.deg)
    spt_wise_gal_sep_deg = spt_wise_gal_cluster_coord.separation(spt_wise_gal_coords)
    spt_wise_gal_sep_mpc = (spt_wise_gal_sep_deg * cosmo.kpc_proper_per_arcmin(spt_wise_gal['REDSHIFT'][0])
                            .to(u.Mpc / spt_wise_gal_sep_deg.unit))

    inner_radius_mpc = inner_radius_factor * spt_wise_gal['R200'][0]
    outer_radius_mpc = outer_radius_factor * spt_wise_gal['R200'][0]
    inner_radius_deg = inner_radius_mpc * cosmo.arcsec_per_kpc_proper(spt_wise_gal['REDSHIFT'][0]).to(u.deg / u.Mpc)
    outer_radius_deg = outer_radius_mpc * cosmo.arcsec_per_kpc_proper(spt_wise_gal['REDSHIFT'][0]).to(u.deg / u.Mpc)

    spt_wise_gal = spt_wise_gal[(inner_radius_mpc < spt_wise_gal_sep_mpc) & (spt_wise_gal_sep_mpc <= outer_radius_mpc)]

    # Also compute the annulus area
    spt_bkg_area = np.pi * (outer_radius_deg ** 2 - inner_radius_deg ** 2)

    spt_wise_gal_data[cluster_id.search(catalog_name).group(0)] = ClusterInfo(catalog=spt_wise_gal,
                                                                              annulus_area=spt_bkg_area)

# Read in the SDWFS WISE galaxy--IRAC AGN scaling factor data
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/'
          'SDWFS_WISEgal-IRACagn_scaling_factors.json', 'r') as f:
    sdwfs_scaling_data = json.load(f)
for d in sdwfs_scaling_data.values():
    for k, v in d.items():
        d[k] = np.array(v)

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Compute the number count distribution of the WISE galaxies in the SPT background annulus
for cluster_data in spt_wise_gal_data.values():
    catalog = cluster_data.catalog
    spt_bkg_area = cluster_data.annulus_area
    selection_color = agn_purity_color(catalog["REDSHIFT"][0])
    scaling_factors = sdwfs_scaling_data[f'SELECTION_MEMBERSHIP_{selection_color:.2f}']['scaling_frac']

    # Calculate our weighting factor
    dndm_weight = spt_bkg_area.value * mag_bin_width

    # Create histogram
    spt_wise_dndm, _ = np.histogram(catalog['w2mpro'].value, bins=magnitude_bins)
    spt_wise_dndm_weighted = spt_wise_dndm / dndm_weight

    # Compute the errors
    spt_wise_dndm_err = tuple(err / dndm_weight for err in small_poisson(spt_wise_dndm))[::-1]

    # Determine the fractional error
    spt_wise_frac_err = spt_wise_dndm_err / spt_wise_dndm_weighted

    # Scale the number count distributions to the IRAC AGN levels
    spt_wise_dndm_scaled = spt_wise_dndm_weighted * scaling_factors

    # Fix the errors using a constant fractional error
    spt_wise_dndm_scaled_err = spt_wise_dndm_scaled * spt_wise_frac_err

    # Store the scaled number count distribution and errors for later
    cluster_data.wise_scaled_dndm = spt_wise_dndm_scaled
    cluster_data.wise_scaled_dndm_err = spt_wise_dndm_scaled_err

    # Also store the associated SDWFS number count distribution and errors
    cluster_data.sdwfs_irac_dndm = sdwfs_scaling_data[f'SELECTION_MEMBERSHIP_{selection_color:.2f}']['hist']
    cluster_data.sdwfs_irac_dndm_err = sdwfs_scaling_data[f'SELECTION_MEMBERSHIP_{selection_color:.2f}']['err']

# For each cluster, renormalize the SDWFS number count distribution to match the scaled WISE galaxy level
for cluster_data in spt_wise_gal_data.values():
    # First, we need to only select a narrow range of magnitudes where the data is reliable
    narrow_magnitude_filter = (15. < magnitude_bin_centers) & (magnitude_bin_centers <= 17.)
    spt_dndm_narrow = cluster_data.wise_scaled_dndm[narrow_magnitude_filter]
    sdwfs_dndm_narrow = cluster_data.sdwfs_irac_dndm[narrow_magnitude_filter]

    # Do the same with the errors
    spt_dndm_err_narrow = cluster_data.wise_scaled_dndm_err[:, narrow_magnitude_filter]
    sdwfs_dndm_err_narrow = cluster_data.sdwfs_irac_dndm_err[:, narrow_magnitude_filter]

    # Symmetrize the errors
    spt_dndm_symerr_narrow = np.sqrt(spt_dndm_err_narrow[0] * spt_dndm_err_narrow[1])
    sdwfs_dndm_symerr_narrow = np.sqrt(sdwfs_dndm_err_narrow[0] * sdwfs_dndm_err_narrow[1])

    # For our model fitting we will use errors that combine the two sources together
    combined_errors = np.sqrt(spt_dndm_symerr_narrow ** 2 + sdwfs_dndm_symerr_narrow ** 2)

    # Fit a simple model that relates the SDWFS IRAC and SPT scaled WISE distributions together
    renorm_popt, remorm_pcov = curve_fit(lambda data, a: a * data, sdwfs_dndm_narrow, spt_dndm_narrow,
                                         sigma=combined_errors)

    # Apply the scaling factor to the data
    cluster_data.sdwfs_irac_dndm_scaled = cluster_data.sdwfs_irac_dndm * renorm_popt

# For each cluster integrate the scaled SDWFS number count distribution over the full selection magnitude range to find
# the local background AGN surface density estimation
spt_local_bkg_agn_surf_den = {cluster_name: simpson(cluster_data.sdwfs_irac_dndm_scaled, magnitude_bin_centers)
                              for cluster_name, cluster_data in spt_wise_gal_data.items()}

# Write the results to file
with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/SPTcl-local_bkg.json', 'w') as f:
    json.dump(spt_local_bkg_agn_surf_den, f, cls=NumpyArrayEncoder)
