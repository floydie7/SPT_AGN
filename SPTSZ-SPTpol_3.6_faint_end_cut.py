"""
SPTSZ-SPTpol_3.6_faint_end_cut.py
Author: Benjamin Floyd

Determines the faint-end cut to be used in AGN selections going forward based on the SNR found in the common clusters
of SPT-SZ and SPTpol 100d surveys.
"""

import glob
import json

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, setdiff


def merge_catalogs(targeted_cat, ssdf_cat):
    # Merge the catalogs into a unified catalog
    targeted_coord = SkyCoord(targeted_cat['ALPHA_J2000'], targeted_cat['DELTA_J2000'], unit=u.deg)
    ssdf_coord = SkyCoord(ssdf_cat['ALPHA_J2000'], ssdf_cat['DELTA_J2000'], unit=u.deg)

    # Match the catalogs
    idx, sep, _ = targeted_coord.match_to_catalog_sky(ssdf_coord)
    sep_constraint = sep <= 1 * u.arcsec
    targeted_matches = targeted_cat[sep_constraint]
    ssdf_matches = ssdf_cat[idx[sep_constraint]]
    # Also identify the non-matches
    targeted_nonmatches = setdiff(targeted_cat, targeted_matches)
    ssdf_nonmatches = setdiff(ssdf_cat, ssdf_matches)

    # Concatenate the flux error columns using the scheme: (common matches, targeted non-matches, ssdf non-matches)
    targeted_flux_err_i1 = np.concatenate([targeted_matches[f'I1_FLUXERR_APER4'],
                                       targeted_nonmatches[f'I1_FLUXERR_APER4'],
                                       np.full_like(ssdf_nonmatches[f'I1_FLUXERR_APER4'], -99.)])
    ssdf_flux_err_i1 = np.concatenate([ssdf_matches[f'I1_FLUXERR_APER4'],
                                   np.full_like(targeted_nonmatches[f'I1_FLUXERR_APER4'], -99.),
                                   ssdf_nonmatches[f'I1_FLUXERR_APER4']])
    targeted_flux_err_i2 = np.concatenate([targeted_matches[f'I2_FLUXERR_APER4'],
                                       targeted_nonmatches[f'I2_FLUXERR_APER4'],
                                       np.full_like(ssdf_nonmatches[f'I2_FLUXERR_APER4'], -99.)])
    ssdf_flux_err_i2 = np.concatenate([ssdf_matches[f'I2_FLUXERR_APER4'],
                                   np.full_like(targeted_nonmatches[f'I2_FLUXERR_APER4'], -99.),
                                   ssdf_nonmatches[f'I2_FLUXERR_APER4']])

    # Finally, collect everything into a single table
    merged_catalog = Table([targeted_flux_err_i1, targeted_flux_err_i2, ssdf_flux_err_i1, ssdf_flux_err_i2],
                           names=[f'TARGETED_I1_FLUXERR', f'TARGETED_I2_FLUXERR', f'SSDF_I1_FLUXERR',
                                  f'SSDF_I2_FLUXERR'])

    return merged_catalog


# Read in the catalog of all clusters common to both the targeted and SSDF programs
common_cluster_cat = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Misc/common_clusters_2500d_100d.fits')

# Remove clusters that don't have IRAC imaging in both surveys (1 = targeted, 2 = SSDF, 3 = both)
common_cluster_cat = common_cluster_cat[common_cluster_cat['IRAC_imaging'] == 3]

# Read in the sky error catalogs
targeted_sky_errors = Table.read(
    'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPT-SZ_2500d/SPT-SZ_sky_errors_qphot.fits')
ssdf_sky_errors = Table.read(
    'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPTpol_100d/SPTpol_sky_errors_qphot.fits')

# Read in the Observed-Official ID lookup tables for the targeted sample.
with open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_observed_to_official_ids.json', 'r') as f, \
        open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_official_to_observed_ids.json', 'r') as g:
    obs_to_off_ids = json.load(f)
    off_to_obs_ids = json.load(g)

# Get list of photometric catalogs
targeted_cat_names = glob.glob('Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPT-SZ_2500d/*.cat')
ssdf_cat_names = glob.glob('Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPTpol_100d/*.cat')

targeted_catalogs, ssdf_catalogs = [], []
for cluster_id in common_cluster_cat['SPT_ID']:
    # Pull the sky error values
    targeted_I1_flux_err = targeted_sky_errors['I1_flux_error'][
        targeted_sky_errors['SPT_ID'] == off_to_obs_ids[cluster_id]]
    targeted_I2_flux_err = targeted_sky_errors['I2_flux_error'][
        targeted_sky_errors['SPT_ID'] == off_to_obs_ids[cluster_id]]
    ssdf_I1_flux_err = ssdf_sky_errors['I1_flux_error'][ssdf_sky_errors['SPT_ID'] == cluster_id]
    ssdf_I2_flux_err = ssdf_sky_errors['I2_flux_error'][ssdf_sky_errors['SPT_ID'] == cluster_id]

    # Extract the catalog names
    targeted_catalog_name = ''.join(s for s in targeted_cat_names if off_to_obs_ids[cluster_id] in s)
    ssdf_catalog_name = ''.join(s for s in ssdf_cat_names if cluster_id in s)

    # Read in the catalogs
    targeted_catalog = Table.read(targeted_catalog_name, format='ascii')
    ssdf_catalog = Table.read(ssdf_catalog_name, format='ascii')

    # Fix a problem that arises later with column-names
    if 'TILE' in targeted_catalog.colnames:
        targeted_catalog.rename_column('TILE', 'TARGET')

    # To create a common footprint, we will select all objects within 1.5 arcmin of the Huang+20 (SPTpol) center.
    center = common_cluster_cat['RA_Huang', 'DEC_Huang'][common_cluster_cat['SPT_ID'] == cluster_id]
    center_coord = SkyCoord(center['RA_Huang'], center['DEC_Huang'], unit=u.deg)
    targeted_coords = SkyCoord(targeted_catalog['ALPHA_J2000'], targeted_catalog['DELTA_J2000'], unit=u.deg)
    ssdf_coords = SkyCoord(ssdf_catalog['ALPHA_J2000'], ssdf_catalog['DELTA_J2000'], unit=u.deg)

    targeted_catalog = targeted_catalog[targeted_coords.separation(center_coord) <= 2 * u.arcmin]
    ssdf_catalog = ssdf_catalog[ssdf_coords.separation(center_coord) <= 2 * u.arcmin]

    # Add columns to store the sky flux errors
    targeted_catalog['I1_flux_error'] = targeted_I1_flux_err
    targeted_catalog['I2_flux_error'] = targeted_I2_flux_err
    ssdf_catalog['I1_flux_error'] = ssdf_I1_flux_err
    ssdf_catalog['I2_flux_error'] = ssdf_I2_flux_err

    # Store the catalogs for stacking later
    targeted_catalogs.append(targeted_catalog)
    ssdf_catalogs.append(ssdf_catalog)

# Combine all the catalogs
targeted_catalogs = vstack(targeted_catalogs)
ssdf_catalogs = vstack(ssdf_catalogs)
master_catalog = merge_catalogs(targeted_cat=targeted_catalogs, ssdf_cat=ssdf_catalogs)

# Compute the fluxes corresponding to SNR = 10 (filtering out bad flux errors)
targeted_i1_flux_snr10 = 10 * master_catalog['TARGETED_I1_FLUXERR'][master_catalog['TARGETED_I1_FLUXERR'] > 0.]
targeted_i2_flux_snr10 = 10 * master_catalog['TARGETED_I2_FLUXERR'][master_catalog['TARGETED_I2_FLUXERR'] > 0.]
ssdf_i1_flux_snr10 = 10 * master_catalog['SSDF_I1_FLUXERR'][master_catalog['SSDF_I1_FLUXERR'] > 0.]
ssdf_i2_flux_snr10 = 10 * master_catalog['SSDF_I2_FLUXERR'][master_catalog['SSDF_I2_FLUXERR'] > 0.]

# The limiting flux (in uJy) in each band will be the minimum flux corresponding to SNR = 10
targeted_i1_limiting_flux = targeted_i1_flux_snr10.min()
targeted_i2_limiting_flux = targeted_i2_flux_snr10.min()
ssdf_i1_limiting_flux = ssdf_i1_flux_snr10.min()
ssdf_i2_limiting_flux = ssdf_i2_flux_snr10.min()

# Convert to Vega magnitudes
i1_zpt_mag = 2.5 * np.log10((280.9 * u.Jy).to_value(u.uJy))
i2_zpt_mag = 2.5 * np.log10((179.7 * u.Jy).to_value(u.uJy))
targeted_i1_limiting_mag = -2.5 * np.log10(targeted_i1_limiting_flux) + i1_zpt_mag
targeted_i2_limiting_mag = -2.5 * np.log10(targeted_i2_limiting_flux) + i2_zpt_mag
ssdf_i1_limiting_mag = -2.5 * np.log10(ssdf_i1_limiting_flux) + i1_zpt_mag
ssdf_i2_limiting_mag = -2.5 * np.log10(ssdf_i2_limiting_flux) + i2_zpt_mag

print((f'{"":^7}\t{"targeted":^12}\t{"ssdf":^8}\n'
       f'{"":^7}\t{"":{"="}^24}\n'
       f'I1 Flux\t{targeted_i1_limiting_flux:^12.2f}\t{ssdf_i1_limiting_flux:^8.2f}\n'
       f'I1 Mag\t{targeted_i1_limiting_mag:^12.2f}\t{ssdf_i1_limiting_mag:^8.2f}\n'
       f'I2 Flux\t{targeted_i2_limiting_flux:^12.2f}\t{ssdf_i2_limiting_flux:^8.2f}\n'
       f'I2 Mag\t{targeted_i2_limiting_mag:^12.2f}\t{ssdf_i2_limiting_mag:^8.2f}'))
print(f'Minimum (brightest) limiting magnitudes:\n'
      f'I1 = {np.min([targeted_i1_limiting_mag, ssdf_i1_limiting_mag]):.2f}\n'
      f'I2 = {np.min([targeted_i2_limiting_mag, ssdf_i2_limiting_mag]):.2f}')
