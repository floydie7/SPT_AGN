"""
SPTSZ-SPTpol_AGN_comparison.py
Author: Benjamin Floyd

For clusters common to both SPT-SZ and SPTpol 100d, plot the selected objects on the images to confirm if we are
selecting for the same objects independent of the image/catalog source (targeted IRAC vs SSDF).
"""

import glob
import json

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import imshow_norm, LinearStretch, ZScaleInterval
from astropy.wcs import WCS

# Get the image filenames
sptsz_image_names = glob.glob('Data/Images/I1*_mosaic.cutout.fits')
sptpol_image_names = glob.glob('Data/SPTPol/images/cluster_cutouts/I1*_mosaic.cutout.fits')

# Because the SPT-SZ images use the older IDs, we will need to use a conversion dictionary from "official" to "observed"
with open('Data/SPT-SZ_official_to_observed_ids.json', 'r') as f:
    sptsz_official_to_obs_id = json.load(f)

# Read in the two catalogs
sptsz_agn = Table.read('Data/Output/SPTSZ_IRAGN.fits')
sptpol_agn = Table.read('Data/Output/SPTpol_IRAGN.fits')

# Find the common clusters
common_ids = sptpol_agn[np.in1d(sptpol_agn['SPT_ID'], sptsz_agn['SPT_ID'])].group_by('SPT_ID').groups.keys

for cluster_id in common_ids['SPT_ID']:
    # Select for the cluster objects in the catalogs
    sptsz_objects = sptsz_agn[cluster_id == sptsz_agn['SPT_ID']]
    sptpol_objects = sptpol_agn[cluster_id == sptpol_agn['SPT_ID']]

    # To find the SPT-SZ image we will need to convert from the official ID to the observed ID
    sptsz_obs_id = sptsz_official_to_obs_id[cluster_id]

    # Find the images
    sptsz_image_name = ''.join(s for s in sptsz_image_names if sptsz_obs_id in s)
    sptpol_image_name = ''.join(s for s in sptpol_image_names if cluster_id in s)

    # Read in the images
    sptsz_image, sptsz_header = fits.getdata(sptsz_image_name, header=True)
    sptpol_image, sptpol_header = fits.getdata(sptpol_image_name, header=True)

    # Make WCS objects
    sptsz_wcs = WCS(sptsz_header)
    sptpol_wcs = WCS(sptpol_header)

    # Plot the objects on the SPT-SZ image
    fig = plt.figure(figsize=(10.0, 4.8))
    ax_sz = fig.add_subplot(121, projection=sptsz_wcs)
    imshow_norm(sptsz_image, ax=ax_sz, origin='lower', cmap='Greys', stretch=LinearStretch(), interval=ZScaleInterval())
    ax_sz.scatter(sptsz_objects['ALPHA_J2000'], sptsz_objects['DELTA_J2000'], s=10,
               facecolor='none', edgecolor='magenta', marker='o', transform=ax_sz.get_transform('world'), label='SPT-SZ')
    ax_sz.scatter(sptpol_objects['ALPHA_J2000'], sptpol_objects['DELTA_J2000'],
               facecolor='none', edgecolor='cyan', marker='s', transform=ax_sz.get_transform('world'), label='SPTpol')
    # ax_sz.legend()
    ax_sz.set(title='Targeted IRAC', xlabel='RA', ylabel='Dec')

    ax_pol = fig.add_subplot(122, projection=sptpol_wcs)
    imshow_norm(sptpol_image, ax=ax_pol, origin='lower', cmap='Greys', stretch=LinearStretch(), interval=ZScaleInterval())
    ax_pol.scatter(sptsz_objects['ALPHA_J2000'], sptsz_objects['DELTA_J2000'], s=10,
               facecolor='none', edgecolor='magenta', marker='o', transform=ax_pol.get_transform('world'), label='SPT-SZ')
    ax_pol.scatter(sptpol_objects['ALPHA_J2000'], sptpol_objects['DELTA_J2000'],
               facecolor='none', edgecolor='cyan', marker='s', transform=ax_pol.get_transform('world'), label='SPTpol')
    ax_pol.legend()
    ax_pol.set(title='SSDF IRAC', xlabel='RA', ylabel=' ')

    fig.suptitle(rf'{cluster_id} (3.6 $\mu\rm m$)')
    # fig.subplots_adjust(wspace=0.35)
    fig.savefig(f'Data/Plots/SPT-SZ_SPTpol_AGN_comparison/SPT-SZ_SPTpol100d_AGN_comparison_{cluster_id}.pdf')

    # Compare the magnitudes of the objects that are found by both catalogs
    sptsz_coords = SkyCoord(sptsz_objects['ALPHA_J2000'], sptsz_objects['DELTA_J2000'], unit='deg')
    sptpol_coords = SkyCoord(sptpol_objects['ALPHA_J2000'], sptpol_objects['DELTA_J2000'], unit='deg')
    idx, sep, _ = sptsz_coords.match_to_catalog_sky(sptpol_coords)
    sep_constraint = sep <= 2 * u.arcsec

    sptsz_matches = sptsz_objects[sep_constraint]
    sptpol_matches = sptpol_objects[idx[sep_constraint]]

    fig, (ax, bx) = plt.subplots(ncols=2)
    bin_width = 0.01
    i1_mag_diff = sptsz_matches['I1_MAG_APER4'] - sptpol_matches['I1_MAG_APER4']
    # i1_bins = np.arange(i1_mag_diff.min(), i1_mag_diff.max() + bin_width, bin_width)
    i2_mag_diff = sptsz_matches['I2_MAG_APER4'] - sptpol_matches['I2_MAG_APER4']
    # i2_bins = np.arange(i2_mag_diff.min(), i2_mag_diff.max() + bin_width, bin_width)
    ax.scatter(sptsz_matches['I1_MAG_APER4'], i1_mag_diff)
    ax.axhline(0.0, ls='--', alpha=0.5)
    ax.set(ylabel=r'$[3.6]_{\rm SPT-SZ} - [3.6]_{\rm SPTpol}$', xlabel=r'$[3.6]_{\rm SPT-SZ}$')
    bx.scatter(sptsz_matches['I2_MAG_APER4'], i2_mag_diff)
    bx.axhline(0.0, ls='--', alpha=0.5)
    bx.set(ylabel=r'$[4.5]_{\rm SPT-SZ} - [4.5]_{\rm SPTpol}$', xlabel=r'$[4.5]_{\rm SPT-SZ}$')
    plt.tight_layout()
    fig.savefig(f'Data/Plots/SPT-SZ_SPTpol_AGN_comparison/Photometry_Comparison_Matched_{cluster_id}.pdf')
    plt.close('all')
