"""
SPTSZ-SPTpol_catalog_comparison.py
Author: Benjamin Floyd

Compares the targeted vs SSDF catalogs by applying both common footprint and magnitude cuts to investigate differences
in number counts of the two catalogs.
"""

import glob
import json
import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, setdiff
from astropy.visualization import imshow_norm, LinearStretch, ZScaleInterval
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from astro_compendium.utils.small_poisson import small_poisson


def flux_plot(ax, targeted_cat, ssdf_cat, sigma_levels, targeted_agn_cat=None, ssdf_agn_cat=None):
    sigma_3, sigma_5, sigma_10 = sigma_levels
    stacked_data = np.hstack((np.log10(targeted_catalog['I2_FLUX_APER4']), np.log10(ssdf_catalog['I2_FLUX_APER4'])))
    shared_bins = np.histogram_bin_edges(stacked_data, bins='auto')
    ax.hist(np.log10(targeted_cat['I2_FLUX_APER4']), bins=shared_bins, cumulative=-1, histtype='step', log=True,
            color='C0', label='Targeted')
    ax.hist(np.log10(ssdf_cat['I2_FLUX_APER4']), bins=shared_bins, cumulative=-1, histtype='step', log=True,
            color='C1', label='SSDF')
    if targeted_agn_cat is not None:
        ax.hist(np.log10(targeted_agn_cat['I2_FLUX_APER4']), bins=shared_bins, cumulative=-1, histtype='bar', log=True,
                color='C0', label='Targeted AGN', alpha=0.6)
    if ssdf_agn_cat is not None:
        ax.hist(np.log10(ssdf_agn_cat['I2_FLUX_APER4']), bins=shared_bins, cumulative=-1, histtype='bar', log=True,
                color='C1', label='SSDF AGN', alpha=0.4)
    ax.axvline(x=sigma_3, color='k', ls='dotted')
    ax.axvline(x=sigma_5, color='k', ls='dashdot')
    ax.axvline(x=sigma_10, color='k', ls='dashed')
    ax.axvline(x=)
    ax.legend()
    ax.set(title=f'{cluster_id}', xlabel=r'$\log\/S_{{4.5\mu\rm m}}\/[\mu\rm Jy]$', ylabel=r'$N(>S_{{4.5\mu\rm m}})$')


def mag_plot(ax, targeted_cat, ssdf_cat, targeted_agn_cat=None, ssdf_agn_cat=None):
    stacked_data = np.hstack((targeted_catalog['I2_MAG_APER4'], ssdf_catalog['I2_MAG_APER4']))
    shared_bins_mag = np.histogram_bin_edges(stacked_data, bins='auto')
    ax.hist(targeted_cat['I2_MAG_APER4'], bins=shared_bins_mag, cumulative=True, histtype='step', log=True,
            color='C0', label='Targeted')
    ax.hist(ssdf_cat['I2_MAG_APER4'], bins=shared_bins_mag, cumulative=True, histtype='step', log=True,
            color='C1', label='SSDF')
    if targeted_agn_cat is not None:
        ax.hist(targeted_agn_cat['I2_MAG_APER4'], bins=shared_bins_mag, cumulative=True, histtype='bar', log=True,
                color='C0', alpha=0.6, label='Targeted AGN')
    if ssdf_agn_cat is not None:
        ax.hist(ssdf_agn_cat['I2_MAG_APER4'], bins=shared_bins_mag, cumulative=True, histtype='bar', log=True,
                color='C1', alpha=0.6, label='SSDF AGN')
    ax.axvline(x=17.46, color='k', ls='--')
    ax.legend()
    ax.set(title=f'{cluster_id}', xlabel='[4.5] (Vega)', ylabel='N(<[4.5])')
    ax.invert_xaxis()

# Read in the catalog of all clusters common to both the targeted and SSDF programs
common_cluster_cat = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Misc/common_clusters_2500d_100d.fits')

# Remove clusters that don't have IRAC imaging in both surveys (1 = targeted, 2 = SSDF, 3 = both)
common_cluster_cat = common_cluster_cat[common_cluster_cat['IRAC_imaging'] == 3]

# Read in the sky error catalogs
targeted_sky_errors = Table.read('Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPT-SZ_2500d/SPT-SZ_sky_errors_qphot.fits')
ssdf_sky_errors = Table.read('Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPTpol_100d/SPTpol_sky_errors_qphot.fits')

# Read in the Observed-Official ID lookup tables for the targeted sample.
with open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_observed_to_official_ids.json', 'r') as f, \
        open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_official_to_observed_ids.json', 'r') as g:
    obs_to_off_ids = json.load(f)
    off_to_obs_ids = json.load(g)

# Get list of photometric catalogs
targeted_cat_names = glob.glob('Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPT-SZ_2500d/*.cat')
ssdf_cat_names = glob.glob('Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPTpol_100d/*.cat')

# Also read in the targeted IRAC image
image_names = glob.glob('Data_Repository/Images/SPT/Spitzer_IRAC/SPT-SZ_2500d/I2*_mosaic.cutout.fits')

spt_id = re.compile(r'SPT-CLJ\d+-\d+')
for cluster_id in common_cluster_cat['SPT_ID']:
    # Extract the catalog names
    targeted_catalog_name = ''.join(s for s in targeted_cat_names if off_to_obs_ids[cluster_id] in s)
    ssdf_catalog_name = ''.join(s for s in ssdf_cat_names if cluster_id in s)

    # Read in the catalogs
    targeted_catalog = Table.read(targeted_catalog_name, format='ascii')
    ssdf_catalog = Table.read(ssdf_catalog_name, format='ascii')

    # To create a common footprint, we will select all objects within 1.5 arcmin of the Huang+20 (SPTpol) center.
    center = common_cluster_cat['RA_Huang', 'DEC_Huang'][common_cluster_cat['SPT_ID'] == cluster_id]
    center_coord = SkyCoord(center['RA_Huang'], center['DEC_Huang'], unit=u.deg)
    targeted_coords = SkyCoord(targeted_catalog['ALPHA_J2000'], targeted_catalog['DELTA_J2000'], unit=u.deg)
    ssdf_coords = SkyCoord(ssdf_catalog['ALPHA_J2000'], ssdf_catalog['DELTA_J2000'], unit=u.deg)

    targeted_catalog = targeted_catalog[targeted_coords.separation(center_coord) <= 1.5 * u.arcmin]
    ssdf_catalog = ssdf_catalog[ssdf_coords.separation(center_coord) <= 1.5 * u.arcmin]

    # Make SNR cuts
    targeted_catalog = targeted_catalog[(targeted_catalog['I2_FLUX_APER4'] / targeted_catalog['I2_FLUXERR_APER4'] >= 10) &
                                        (targeted_catalog['I1_FLUX_APER4'] / targeted_catalog['I1_FLUXERR_APER4'] >= 10)]
    ssdf_catalog = ssdf_catalog[(ssdf_catalog['I2_FLUX_APER4'] / ssdf_catalog['I2_FLUXERR_APER4'] >= 10) &
                                (ssdf_catalog['I1_FLUX_APER4'] / ssdf_catalog['I1_FLUXERR_APER4'] >= 10)]

    # Make magnitude cuts
    targeted_catalog = targeted_catalog[(10.0 <= targeted_catalog['I1_MAG_APER4']) &
                                        (10.45 <= targeted_catalog['I2_MAG_APER4']) &
                                        (targeted_catalog['I2_MAG_APER4'] <= 17.46)]
    ssdf_catalog = ssdf_catalog[(10.0 <= ssdf_catalog['I1_MAG_APER4']) &
                                (10.45 <= ssdf_catalog['I2_MAG_APER4']) &
                                (ssdf_catalog['I2_MAG_APER4'] <= 17.46)]

    # Make AGN color cut
    targeted_agn = targeted_catalog[targeted_catalog['I1_MAG_APER4'] - targeted_catalog['I2_MAG_APER4'] >= 0.7]
    ssdf_agn = ssdf_catalog[ssdf_catalog['I1_MAG_APER4'] - ssdf_catalog['I2_MAG_APER4'] >= 0.7]

    # Strangely there are some non-detections present in the 4.5 um band columns of the SSDF catalogs so we will remove
    # them for now until a better understanding or solution is found.
    ssdf_catalog = ssdf_catalog[ssdf_catalog['I2_FLUX_APER4'] != -99.0]

    # First create a logN-logS plot
    shared_bins = np.histogram_bin_edges(np.hstack((np.log10(targeted_catalog['I2_FLUX_APER4']),
                                                    np.log10(ssdf_catalog['I2_FLUX_APER4']))), bins='auto')
    fig, ax = plt.subplots()
    ax.hist(np.log10(targeted_catalog['I2_FLUX_APER4']), bins=shared_bins, cumulative=-1, histtype='step', log=True,
            color='C0', label='Targeted')
    ax.hist(np.log10(targeted_agn['I2_FLUX_APER4']), bins=shared_bins, cumulative=-1, histtype='bar', log=True,
            color='C0', label='Targeted AGN', alpha=0.6)
    ax.hist(np.log10(ssdf_catalog['I2_FLUX_APER4']), bins=shared_bins, cumulative=-1, histtype='step', log=True,
            color='C1', label='SSDF')
    ax.hist(np.log10(ssdf_agn['I2_FLUX_APER4']), bins=shared_bins, cumulative=-1, histtype='bar', log=True,
            color='C1', label='SSDF AGN', alpha=0.4)
    ax.legend()
    ax.set(title=f'{cluster_id}', xlabel=r'$\log\/S_{{4.5\mu\rm m}}\/[\mu\rm Jy]$',
           ylabel=r'$N(>S_{{4.5\mu\rm m}})$')
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/Source_Counts/SNR_cut_I1_and_I2_mag_cut_AGN/'
                f'flux/{cluster_id}_I2_flux.pdf')

    # Make the same plot but with magnitudes
    shared_bins_mag = np.histogram_bin_edges(np.hstack((targeted_catalog['I2_MAG_APER4'],
                                                        ssdf_catalog['I2_MAG_APER4'])), bins='auto')
    fig, ax = plt.subplots()
    ax.hist(targeted_catalog['I2_MAG_APER4'], bins=shared_bins_mag, cumulative=True, histtype='step', log=False,
            color='C0', label='Targeted')
    ax.hist(targeted_agn['I2_MAG_APER4'], bins=shared_bins_mag, cumulative=True, histtype='bar', log=False,
            color='C0', alpha=0.6, label='Targeted AGN')
    ax.hist(ssdf_catalog['I2_MAG_APER4'], bins=shared_bins_mag, cumulative=True, histtype='step', log=False,
            color='C1', label='SSDF')
    ax.hist(ssdf_agn['I2_MAG_APER4'], bins=shared_bins_mag, cumulative=True, histtype='bar', log=False,
            color='C1', alpha=0.6, label='SSDF AGN')
    ax.axvline(x=17.46, color='k', ls='--')
    ax.legend()
    ax.set(title=f'{cluster_id}', xlabel='[4.5] (Vega)', ylabel='N(<[4.5])')
    ax.invert_xaxis()
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/Source_Counts/SNR_cut_I1_and_I2_mag_cut_AGN/'
                f'mag/{cluster_id}_I2_mag.pdf')

    # Read in the image
    image_name = ''.join(s for s in image_names if off_to_obs_ids[cluster_id] in s)
    image, header = fits.getdata(image_name, header=True)
    wcs = WCS(header)

    # Plot the objects on the deep targeted image
    if targeted_agn:
        targeted_galaxies = setdiff(targeted_catalog, targeted_agn)
    else:
        targeted_galaxies = targeted_catalog
    if ssdf_agn:
        ssdf_galaxies = setdiff(ssdf_catalog, ssdf_agn)
    else:
        ssdf_galaxies = ssdf_catalog
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=wcs))
    imshow_norm(image, ax=ax, origin='lower', cmap='Greys', interval=ZScaleInterval(), stretch=LinearStretch())
    ax.scatter(targeted_galaxies['ALPHA_J2000'], targeted_galaxies['DELTA_J2000'], marker='s', edgecolor='w',
               facecolor='none', s=55, linewidths=1, transform=ax.get_transform('world'), label='Targeted')
    if targeted_agn:
        ax.scatter(targeted_agn['ALPHA_J2000'], targeted_agn['DELTA_J2000'], marker='*', edgecolor='w', facecolor='none',
                   s=65, linewidths=1, transform=ax.get_transform('world'), label='Targeted AGN')
    ax.scatter(ssdf_galaxies['ALPHA_J2000'], ssdf_galaxies['DELTA_J2000'], marker='o', edgecolor='r',
               facecolor='none', s=50, linewidths=1, transform=ax.get_transform('world'), label='SSDF')
    if ssdf_agn:
        ax.scatter(ssdf_agn['ALPHA_J2000'], ssdf_agn['DELTA_J2000'], marker='*', edgecolor='r', facecolor='none', s=60,
                   linewidths=1, transform=ax.get_transform('world'), label='SSDF AGN')
    ax.add_patch(SphericalCircle(center=(center['RA_Huang'] * u.deg, center['DEC_Huang'] * u.deg),
                                 radius=1.5 * u.arcmin, edgecolor='green', facecolor='none', linestyle='--',
                                 transform=ax.get_transform('world')))
    ax.legend()
    ax.set(title=f'{cluster_id}', xlabel='Right Ascension', ylabel='Declination')
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/Source_Counts/SNR_cut_I1_and_I2_mag_cut_AGN/'
                f'images/{cluster_id}_I2_image.pdf')
    plt.close('all')
