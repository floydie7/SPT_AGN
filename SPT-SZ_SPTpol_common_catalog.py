"""
SPT-SZ_SPTpol_common_catalog.py
Author: Benjamin Floyd

Matches IRAC catalogs of clusters common to both the SPT-SZ/targeted IRAC observations and the SPTpol/SSDF observations.
"""

import glob
import json

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

# %% Get the image filenames
sptsz_image_names = glob.glob('Data/Images/I1*_mosaic.cutout.fits')
sptpol_image_names = glob.glob('Data/SPTPol/images/cluster_cutouts/I1*_mosaic.cutout.fits')

# Get the IRAC catalog filenames
sptsz_catalog_names = glob.glob('Data/Catalogs/*.cat')
sptpol_catalog_names = glob.glob('Data/SPTPol/catalogs/cluster_cutouts/*.cat')

# Because the SPT-SZ images use the older IDs, we will need to use a conversion dictionary from "official" to "observed"
with open('Data/SPT-SZ_official_to_observed_ids.json', 'r') as f:
    sptsz_official_to_obs_id = json.load(f)

# Read in the two catalogs
sptsz_agn = Table.read('Data/Output/SPTSZ_IRAGN.fits')
sptpol_agn = Table.read('Data/Output/SPTpol_IRAGN.fits')

# Find the common clusters
common_ids = sptpol_agn[np.in1d(sptpol_agn['SPT_ID'], sptsz_agn['SPT_ID'])].group_by('SPT_ID').groups.keys

# %%
for cluster_id in common_ids['SPT_ID']:
    # To find the SPT-SZ IRAC catalog we will need to convert from the official ID to the observed ID
    sptsz_obs_id = sptsz_official_to_obs_id[cluster_id]

    # Read in the IRAC catalogs
    sptsz_catalog_name = ''.join(s for s in sptsz_catalog_names if sptsz_obs_id in s)
    sptpol_catalog_name = ''.join(s for s in sptpol_catalog_names if cluster_id in s)
    sptsz_objects = Table.read(sptsz_catalog_name, format='ascii')
    sptpol_objects = Table.read(sptpol_catalog_name, format='ascii')

    # Make magnitude cuts that would be applied in our normal selection pipeline
    sptsz_objects = sptsz_objects[(sptsz_objects['I1_MAG_APER4'] >= 10.0) &
                                  (sptsz_objects['I2_MAG_APER4'] >= 10.45) &
                                  (sptsz_objects['I2_MAG_APER4'] <= 17.46)]
    sptpol_objects = sptpol_objects[(sptpol_objects['I1_MAG_APER4'] >= 10.0) &
                                    (sptpol_objects['I2_MAG_APER4'] >= 10.45) &
                                    (sptpol_objects['I2_MAG_APER4'] <= 17.46)]

    sptsz_agn_candidates = sptsz_objects[sptsz_objects['I1_MAG_APER4'] - sptsz_objects['I2_MAG_APER4'] >= 0.7]
    sptpol_agn_candidates = sptpol_objects[sptpol_objects['I1_MAG_APER4'] - sptpol_objects['I2_MAG_APER4'] >= 0.7]

    # Match the catalogs
    sptsz_coords = SkyCoord(sptsz_objects['ALPHA_J2000'], sptsz_objects['DELTA_J2000'], unit='deg')
    sptpol_coords = SkyCoord(sptpol_objects['ALPHA_J2000'], sptpol_objects['DELTA_J2000'], unit='deg')
    idx, sep, _ = sptsz_coords.match_to_catalog_sky(sptpol_coords)

    # Inspection of separation histograms show that a separation of ~1.5 arcsec is a reasonable cutoff
    sep_constraint = sep <= 1.5 * u.arcsec
    sptsz_matches = sptsz_objects[sep_constraint]
    sptpol_matches = sptpol_objects[idx[sep_constraint]]

    # Select for only the faint objects and rematch
    faint_sptsz_coords = sptsz_coords[sptsz_objects['I2_MAG_APER4'] >= 16.46]
    faint_sptpol_coords = sptpol_coords[sptpol_objects['I2_MAG_APER4'] >= 16.46]
    faint_idx, faint_sep, _ = faint_sptsz_coords.match_to_catalog_sky(faint_sptpol_coords)

    faint_sep_constraint = faint_sep <= 1.5 * u.arcsec
    faint_sptsz_matches = sptsz_objects[sptsz_objects['I2_MAG_APER4'] >= 16.46][faint_sep_constraint]
    faint_sptpol_matches = sptpol_objects[sptpol_objects['I2_MAG_APER4'] >= 16.46][faint_idx[faint_sep_constraint]]

    # Compute the color and color errors for the matched objects
    sptsz_color = sptsz_matches['I1_MAG_APER4'] - sptsz_matches['I2_MAG_APER4']
    sptpol_color = sptpol_matches['I1_MAG_APER4'] - sptpol_matches['I2_MAG_APER4']
    sptsz_color_err = np.sqrt(
        (2.5 * sptsz_matches['I1_FLUXERR_APER4'] / (sptsz_matches['I1_FLUX_APER4'] * np.log(10))) ** 2 +
        (2.5 * sptsz_matches['I2_FLUXERR_APER4'] / (sptsz_matches['I2_FLUX_APER4'] * np.log(10))) ** 2)
    sptpol_color_err = np.sqrt(
        (2.5 * sptpol_matches['I1_FLUXERR_APER4'] / (sptpol_matches['I1_FLUX_APER4'] * np.log(10))) ** 2 +
        (2.5 * sptpol_matches['I2_FLUXERR_APER4'] / (sptpol_matches['I2_FLUX_APER4'] * np.log(10))) ** 2)

    bin_width = 0.05
    # Full catalog comparison
    i1_mag_diff = sptsz_matches['I1_MAG_APER4'] - sptpol_matches['I1_MAG_APER4']
    i1_bins = np.arange(i1_mag_diff.min(), i1_mag_diff.max() + bin_width, bin_width)
    i2_mag_diff = sptsz_matches['I2_MAG_APER4'] - sptpol_matches['I2_MAG_APER4']
    i2_bins = np.arange(i2_mag_diff.min(), i2_mag_diff.max() + bin_width, bin_width)

    # Faint objects only
    faint_i1_mag_diff = faint_sptsz_matches['I1_MAG_APER4'] - faint_sptpol_matches['I1_MAG_APER4']
    faint_i1_bins = np.arange(faint_i1_mag_diff.min(), faint_i1_mag_diff.max() + bin_width, bin_width)
    faint_i2_mag_diff = faint_sptsz_matches['I2_MAG_APER4'] - faint_sptpol_matches['I2_MAG_APER4']
    faint_i2_bins = np.arange(faint_i2_mag_diff.min(), faint_i2_mag_diff.max() + bin_width, bin_width)

    i1_mag_diff_scaled = i1_mag_diff / np.sqrt(
        sptsz_matches['I1_MAGERR_APER4'] ** 2 + sptpol_matches['I1_MAGERR_APER4'] ** 2)
    i2_mag_diff_scaled = i2_mag_diff / np.sqrt(
        sptsz_matches['I2_MAGERR_APER4'] ** 2 + sptpol_matches['I2_MAGERR_APER4'] ** 2)

    #%% Magnitude distributions
    # fig, (ax, bx) = plt.subplots(ncols=2)
    # # [3.6] comparison
    # ax.hist(i1_mag_diff, bins=i1_bins, range=(-2, 2), color='C0')
    # ax.hist(faint_i1_mag_diff, bins=faint_i1_bins, range=(-2, 2), color='C1')
    # ax.axvline(np.median(i1_mag_diff), ls='--', color='k', alpha=0.5, label=f'Median: {np.median(i1_mag_diff):.2f}')
    # ax.legend()
    # ax.set(xlabel=r'$[3.6]_{\rm SPT-SZ} - [3.6]_{\rm SPTpol}$')
    #
    # # [4.5] comparison
    # bx.hist(i2_mag_diff, bins=i2_bins, range=(-2, 2), label='All Matches')
    # bx.hist(faint_i2_mag_diff, bins=faint_i2_bins, range=(-2, 2), label=r'$16.46 \leq [4.5] \leq 17.46$')
    # bx.axvline(np.median(i2_mag_diff), ls='--', color='k', alpha=0.5, label=f'Median: {np.median(i2_mag_diff):.2f}')
    # bx.legend()
    # bx.set(xlabel=r'$[4.5]_{\rm SPT-SZ} - [4.5]_{\rm SPTpol}$')
    #
    # fig.suptitle(f'{cluster_id}')
    # plt.tight_layout()
    # fig.savefig(f'Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/'
    #             f'Magnitude_Distribution/{cluster_id}_I1_I2_magnitudes.pdf')


    #%% Color error
    # fig, ax = plt.subplots()
    # ax.scatter(sptsz_color, sptsz_color_err, marker='.', label='SPT-SZ/targeted')
    # ax.scatter(sptpol_color, sptpol_color_err, marker='.', label='SPTpol/SSDF')
    # ax.axvline(0.7, ls='--', c='k', alpha=0.5)
    # ax.legend()
    # ax.set(title=f'{cluster_id}', xlabel=r'$[3.6] - [4.5]$ (Vega)', ylabel=r'$\sigma_{[3.6] - [4.5]}$ (Vega)')
    # fig.savefig(f'Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/Color_Error/'
    #             f'{cluster_id}_color_err.pdf')

    #%% Additionally, plot the magnitude error trends for the SSDF catalogs
    # fig, axarr = plt.subplots(ncols=2, figsize=(2 * 6.4, 4.8))
    # axarr[0].scatter(sptsz_objects['I1_MAG_APER4'], sptsz_objects['I1_MAGERR_APER4'], marker='.',
    #                  label='All SPT-SZ/Targeted objects')
    # # axarr[0].scatter(sptsz_agn_candidates['I1_MAG_APER4'], sptsz_agn_candidates['I1_MAGERR_APER4'], marker='.',
    # #                  label='SPT-SZ/Targeted AGN candidates')
    # axarr[0].scatter(sptpol_objects['I1_MAG_APER4'], sptpol_objects['I1_MAGERR_APER4'], marker='.',
    #                  label='All SPTpol/SSDF objects')
    # # axarr[0].scatter(sptpol_agn_candidates['I1_MAG_APER4'], sptpol_agn_candidates['I1_MAGERR_APER4'], marker='.',
    # #                  label='SPTpol/SSDF AGN candidates')
    # axarr[0].set(xlabel='[3.6] (Vega)', ylabel=r'$\sigma_{[3.6]}$ (Vega)')
    #
    # axarr[1].scatter(sptsz_objects['I2_MAG_APER4'], sptsz_objects['I2_MAGERR_APER4'], marker='.',
    #                  label='All SPT-SZ/Targeted objects')
    # # axarr[1].scatter(sptsz_agn_candidates['I2_MAG_APER4'], sptsz_agn_candidates['I2_MAGERR_APER4'], marker='.',
    # #                  label='SPT-SZ/Targeted AGN candidates')
    # axarr[1].scatter(sptpol_objects['I2_MAG_APER4'], sptpol_objects['I2_MAGERR_APER4'], marker='.',
    #                  label='All SPTpol/SSDF objects')
    # # axarr[1].scatter(sptpol_agn_candidates['I2_MAG_APER4']/////
    # axarr[1].axvline(x=17.45, ls='--', color='k')
    # axarr[1].set(xlabel='[4.5] (Vega)', ylabel=r'$\sigma_{[4.5]}$ (Vega)')
    #
    # axarr[0].legend()
    # fig.suptitle(f'{cluster_id}')
    # fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/Magnitude_Error/'
    #             f'{cluster_id}_mag_err.pdf')

    #%% Magnitude difference trends
    fig, (ax, bx) = plt.subplots(ncols=2)
    # [3.6] comparison
    ax.scatter(sptsz_matches['I1_MAG_APER4'], i1_mag_diff, marker='.')
    ax.axhline(np.median(i1_mag_diff), ls='--', c='k', label=f'Median: {np.median(i1_mag_diff):.2f}')
    ax.legend()
    ax.set(xlabel=r'$[3.6]_{\rm SPT-SZ}$', ylabel=r'$[3.6]_{\rm SPT-SZ} - [3.6]_{\rm SPTpol}$')

    # [4.5] comparison
    bx.scatter(sptsz_matches['I2_MAG_APER4'], i2_mag_diff, marker='.')
    bx.axhline(np.median(i2_mag_diff), ls='--', c='k', label=f'Median: {np.median(i2_mag_diff):.2f}')
    bx.legend()
    bx.set(xlabel=r'$[4.5]_{\rm SPT-SZ}$', ylabel=r'$[4.5]_{\rm SPT-SZ} - [4.5]_{\rm SPTpol}$')

    fig.suptitle(f'{cluster_id}')
    plt.tight_layout()
    fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/Magnitude_Trend/'
                f'{cluster_id}_mag_diff_trend.pdf')

    #%% Magnitude difference trend scaled by errors
    fig, (ax, bx) = plt.subplots(ncols=2)
    # [3.6] comparison
    ax.scatter(sptsz_matches['I1_MAG_APER4'], i1_mag_diff_scaled, marker='.')
    ax.axhline(np.median(i1_mag_diff), ls='--', c='k', label=f'Median: {np.median(i1_mag_diff):.2f}')
    ax.legend()
    ax.set(xlabel=r'$[3.6]_{\rm SPT-SZ}$',
           ylabel=r'$([3.6]_{\rm SPT-SZ} - [3.6]_{\rm SPTpol})/\sqrt{\sigma_{[3.6]_{\rm SPT-SZ}}^2 + \sigma_{[3.6]_{\rm SPTpol}}^2}$')

    # [4.5] comparison
    bx.scatter(sptsz_matches['I2_MAG_APER4'], i2_mag_diff_scaled, marker='.')
    bx.axhline(np.median(i2_mag_diff), ls='--', c='k', label=f'Median: {np.median(i2_mag_diff):.2f}')
    bx.legend()
    bx.set(xlabel=r'$[4.5]_{\rm SPT-SZ}$',
           ylabel=r'$([4.5]_{\rm SPT-SZ} - [4.5]_{\rm SPTpol})/\sqrt{\sigma_{[4.5]_{\rm SPT-SZ}}^2 + \sigma_{[4.5]_{\rm SPTpol}}^2}$')

    fig.suptitle(f'{cluster_id}')
    plt.tight_layout()
    fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/Magnitude_Trend_Scaled/'
                f'{cluster_id}_mag_diff_trend_scaled.pdf')

    plt.close('all')
