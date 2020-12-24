"""
SPT_IRAC-PISCO_colors.py
Author: Benjamin Floyd

Creates color-color plots from all SPT clusters with IRAC imaging using both PISCO and IRAC photometry.
"""
import itertools
import re
import json
import glob
from collections import defaultdict

import numpy as np
import scipy.io as sio
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.table import Table, hstack, vstack
from astropy.coordinates import SkyCoord


def match_catalogs(catalogs):
    # Get the correct IRAC key
    irac_key = 'Targeted_IRAC' if 'Targeted_IRAC' in catalogs.keys() else 'SSDF_IRAC'

    # Read in the catalogs
    pisco = Table(sio.readsav(catalogs['PISCO'])['cat'])
    irac = Table.read(catalogs[irac_key], format='ascii')

    # Apply SNR cuts to PISCO catalog F/dF = 2.5 / (ln(10) * dm)
    snr_cut = 5
    dm_snr_prefactor = 2.5 / np.log(10)
    pisco = pisco[(dm_snr_prefactor / np.abs(pisco['GERR']) >= snr_cut) &
                  (dm_snr_prefactor / np.abs(pisco['RERR']) >= snr_cut) &
                  (dm_snr_prefactor / np.abs(pisco['IERR']) >= snr_cut) &
                  (dm_snr_prefactor / np.abs(pisco['ZERR']) >= snr_cut)]

    # Remove non-detections from the PISCO catalog
    pisco = pisco[(-20. < pisco['G']) & (pisco['G'] < 99.) &
                  (-20. < pisco['R']) & (pisco['R'] < 99.) &
                  (-20. < pisco['I']) & (pisco['I'] < 99.) &
                  (-20. < pisco['Z']) & (pisco['Z'] < 99.)]

    # Add columns for the IRAC bands in AB mags for plotting. We will still make selections in Vega mags.
    irac['I1_MAG_APER4_AB'] = irac['I1_MAG_APER4'] + 2.79
    irac['I2_MAG_APER4_AB'] = irac['I2_MAG_APER4'] + 3.26

    # Apply magnitude cuts to the IRAC catalogs
    irac = irac[(irac['I1_MAG_APER4'] > 10.0) & (irac['I2_MAG_APER4'] > 10.45) & (irac['I2_MAG_APER4'] <= 17.46)]

    if not pisco or not irac:
        return

        # Create SkyCoords for both catalogs
    pisco_coords = SkyCoord(pisco['OBJ_RA'], pisco['OBJ_DEC'], unit=u.deg)
    irac_coords = SkyCoord(irac['ALPHA_J2000'], irac['DELTA_J2000'], unit=u.deg)

    # Match the PISCO and IRAC catalogs
    pisco_idx, pisco_sep, _ = irac_coords.match_to_catalog_sky(pisco_coords)

    # Make the initial selection by taking the closest match and removing matches with separations larger than 3 arcsec
    init_sep_constraint = pisco_sep <= 3 * u.arcsec
    pisco_coords_matches = pisco_coords[pisco_idx[init_sep_constraint]]
    irac_coords_matches = irac_coords[init_sep_constraint]

    # Find the offset coordinates from the IRAC to PISCO coordinates of the initial matches
    dra_ddec = pisco_coords_matches.spherical_offsets_to(irac_coords_matches)

    # Find the centroid of the offsets
    offset_centroid = np.median(dra_ddec, axis=1) * u.deg

    # Cast the offset centroid into SkyCoord
    offset_coord = SkyCoord(offset_centroid[0], offset_centroid[1])

    # Find the position angle and separation of the centroids relative to the origin
    origin = SkyCoord(0, 0, unit=u.deg)
    offset_pa, offset_sep = origin.position_angle(offset_coord), origin.separation(offset_coord)

    # Now that we know the offset corrections, apply the correction to the PISCO coordinates
    pisco_coords_corrected = pisco_coords.directional_offset_by(offset_pa, offset_sep)

    # Rematch the corrected PISCO coordinates to the IRAC coordinates
    pisco_corr_idx, pisco_corr_sep, _ = irac_coords.match_to_catalog_sky(pisco_coords_corrected)

    # Match the catalogs using a maximum separation of 0.5 arcsec as a cutoff
    max_sep = 0.5 * u.arcsec
    pisco = pisco[pisco_corr_idx[pisco_corr_sep <= max_sep]]
    irac = irac[pisco_corr_sep <= max_sep]

    # Merge the matched catalogs
    matched_catalog = hstack([pisco, irac])
    return matched_catalog


# Get a list of all the PISCO catalogs
pisco_cat_list = glob.glob('Data_Repository/Catalogs/SPT/PISCO_catalogs/catalogs/*.sav')

# Get lists of both IRAC catalog versions
targeted_irac_cat_list = glob.glob('Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPT-SZ_2500d/*.cat')
ssdf_irac_cat_list = glob.glob('Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPTpol_100d/*.cat')

# Load in the look-up table to convert the official ids to the old observed ids for the targeted observations
with open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_official_to_observed_ids.json', 'r') as f, \
        open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_observed_to_official_ids.json', 'r') as g:
    off_to_obs_ids = json.load(f)
    obs_to_off_ids = json.load(g)

# Create ID pattern for SPT IDs
spt_id = re.compile(r'SPT-CLJ\d+-\d+')

# Find all cluster ids with IRAC data
cluster_ids = {obs_to_off_ids.get(spt_id.search(fname).group(0), None) for fname in targeted_irac_cat_list}
cluster_ids = cluster_ids.union({spt_id.search(fname).group(0).upper() for fname in ssdf_irac_cat_list})
cluster_ids.discard(None)

cluster_dict = defaultdict(dict)
for cluster_id in cluster_ids:
    # Get the catalog names
    pisco_cat_name = ''.join(s for s in pisco_cat_list if cluster_id in s)
    targeted_irac_cat_name = ''.join(s for s in targeted_irac_cat_list if cluster_id in off_to_obs_ids
                                     and off_to_obs_ids.get(cluster_id, '') in s)
    ssdf_irac_cat_name = ''.join(s for s in ssdf_irac_cat_list if cluster_id in s)

    # Read in the catalogs now if we have them
    if pisco_cat_name:
        cluster_dict[cluster_id]['PISCO'] = pisco_cat_name
    if targeted_irac_cat_name:
        cluster_dict[cluster_id]['Targeted_IRAC'] = targeted_irac_cat_name
    if ssdf_irac_cat_name:
        cluster_dict[cluster_id]['SSDF_IRAC'] = ssdf_irac_cat_name

# Remove any cluster that doesn't have a PISCO catalog
cluster_dict = {cluster_id: catalogs for cluster_id, catalogs in cluster_dict.items()
                if catalogs.get('PISCO', None) is not None}

# Separate the dictionary by IRAC catalog (and filter to insure we have both PISCO and IRAC)
cluster_dict_pisco_targeted = {cluster_id: {catalog_name: catalogs.get(catalog_name, None)
                                            for catalog_name in catalogs.keys() & {'PISCO', 'Targeted_IRAC'}}
                               for cluster_id, catalogs in cluster_dict.items()}
cluster_dict_pisco_targeted = {cluster_id: catalogs for cluster_id, catalogs in cluster_dict_pisco_targeted.items()
                               if len(catalogs) == 2}
cluster_dict_pisco_ssdf = {cluster_id: {catalog_name: catalogs.get(catalog_name, None)
                                        for catalog_name in catalogs.keys() & {'PISCO', 'SSDF_IRAC'}}
                           for cluster_id, catalogs in cluster_dict.items()}
cluster_dict_pisco_ssdf = {cluster_id: catalogs for cluster_id, catalogs in cluster_dict_pisco_ssdf.items()
                           if len(catalogs) == 2}

# Match and merge the PISCO and IRAC catalogs
pisco_targeted_catalogs = {cluster_id: match_catalogs(cats) for cluster_id, cats in cluster_dict_pisco_targeted.items()}
pisco_ssdf_catalogs = {cluster_id: match_catalogs(cats) for cluster_id, cats in cluster_dict_pisco_ssdf.items()}
pisco_targeted_catalogs = {cluster_id: matched_cat for cluster_id, matched_cat in pisco_targeted_catalogs.items()
                           if matched_cat is not None}
pisco_ssdf_catalogs = {cluster_id: matched_cat for cluster_id, matched_cat in pisco_ssdf_catalogs.items()
                       if matched_cat is not None}

# Stack the catalogs
pisco_targeted_stacked = vstack([cat for cat in pisco_targeted_catalogs.values()])
pisco_ssdf_stacked = vstack([cat for cat in pisco_ssdf_catalogs.values()])

# Make a rough AGN selection
pisco_targeted_stacked_agn = pisco_targeted_stacked[
    pisco_targeted_stacked['I1_MAG_APER4'] - pisco_targeted_stacked['I2_MAG_APER4'] >= 0.7
    ]
pisco_ssdf_stacked_agn = pisco_ssdf_stacked[
    pisco_ssdf_stacked['I1_MAG_APER4'] - pisco_ssdf_stacked['I2_MAG_APER4'] >= 0.7
    ]

# And a galaxy selection as the complement of the AGN
pisco_targeted_stacked_gal = pisco_targeted_stacked[
    pisco_targeted_stacked['I1_MAG_APER4'] - pisco_targeted_stacked['I2_MAG_APER4'] < 0.7
    ]
pisco_ssdf_stacked_gal = pisco_ssdf_stacked[
    pisco_ssdf_stacked['I1_MAG_APER4'] - pisco_ssdf_stacked['I2_MAG_APER4'] < 0.7
    ]

# We want to make plot with the following combinations
phot_bands = ['G', 'R', 'I', 'Z', 'I1_MAG_APER4_AB', 'I2_MAG_APER4_AB']
color_list = list(itertools.combinations(phot_bands, 2))
color_color_list = list(itertools.combinations_with_replacement(color_list, 2))
color_color_list = [color_color for color_color in color_color_list if color_color[0] != color_color[1]]

#%% Make the plots
labels = {'G': 'g', 'R': 'r', 'I': 'i', 'Z': 'z', 'I1_MAG_APER4_AB': '[3.6]', 'I2_MAG_APER4_AB': '[4.5]'}
for ((y_blue, y_red), (x_blue, x_red)) in color_color_list:
    fig, ax = plt.subplots()
    ax.scatter(pisco_targeted_stacked_gal[x_blue] - pisco_targeted_stacked_gal[x_red],
               pisco_targeted_stacked_gal[y_blue] - pisco_targeted_stacked_gal[y_red],
               marker='.', color='k', alpha=0.1, label='Galaxies')
    ax.scatter(pisco_ssdf_stacked_gal[x_blue] - pisco_ssdf_stacked_gal[x_red],
               pisco_ssdf_stacked_gal[y_blue] - pisco_ssdf_stacked_gal[y_red],
               marker='.', color='k', alpha=0.1)
    ax.scatter(pisco_targeted_stacked_agn[x_blue] - pisco_targeted_stacked_agn[x_red],
               pisco_targeted_stacked_agn[y_blue] - pisco_targeted_stacked_agn[y_red],
               marker='o', edgecolor='C0', facecolor='none', alpha=1, label='Targeted AGN')
    ax.scatter(pisco_ssdf_stacked_agn[x_blue] - pisco_ssdf_stacked_agn[x_red],
               pisco_ssdf_stacked_agn[y_blue] - pisco_ssdf_stacked_agn[y_red],
               s=7.0 ** 2, marker='o', edgecolor='C1', facecolor='none', alpha=1, label='SSDF AGN')
    ax.legend()
    ax.set(xlabel=f'{labels[x_blue]} - {labels[x_red]} (AB)', ylabel=f'{labels[y_blue]} - {labels[y_red]} (AB)')
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/PISCO_photometry/'
                'Color-Color/all_clusters/'
                f'SPT_stacked_{labels[y_blue]}-{labels[y_red]}_{labels[x_blue]}-{labels[x_red]}.pdf')
    plt.close('all')
