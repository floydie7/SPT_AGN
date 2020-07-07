"""
low-z_multiwavelength_comparison.py
Author: Benjamin Floyd

Visualizes X-ray and radio counterparts to the IR-bright AGN in the 7 lowest redshift SPT clusters.
"""
import glob
import json

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import imshow_norm, LinearStretch, ZScaleInterval
from astropy.wcs import WCS

# Read in the SPTcl-IRAGN catalog
sptcl_iragn = Table.read('Data/Output/SPTcl_IRAGN.fits')

# Select the low-z clusters
low_z_clusters = sptcl_iragn[sptcl_iragn['REDSHIFT'] <= 0.25]

# Read in the SUMSS sources found near the clusters above
sumss = Table.read('Data/Data_Repository/Catalogs/SUMSS/SPT-SUMSS_sources.fits')
sumss_coords = SkyCoord(sumss['RA'].data, sumss['DEC'].data, unit=u.deg)

# Read in the ROSAT catalog of sources found near the clusters above
rass2rxs = Table.read('Data/Data_Repository/Catalogs/ROSAT/SPT-RASS2RXS_sources.fits')
rass2rxs_coords = SkyCoord(rass2rxs['RA'].data, rass2rxs['DEC'].data, unit=u.deg)

# Get the IRAC image filenames
sptsz_image_names = glob.glob('Data/Images/I1*_mosaic.cutout.fits')
sptpol_image_names = glob.glob('Data/SPTPol/images/cluster_cutouts/I1*_mosaic.cutout.fits')


# Because the SPT-SZ images use the older IDs, we will need to use a conversion dictionary from "official" to "observed"
with open('Data/SPT-SZ_official_to_observed_ids.json', 'r') as f:
    sptsz_official_to_obs_id = json.load(f)

for cluster in low_z_clusters.group_by('SPT_ID').groups:
    cluster_id = cluster['SPT_ID'][0]
    cluster_sz_center = SkyCoord(cluster['SZ_RA'][0], cluster['SZ_DEC'][0], unit=u.deg)

    # Select the multiwavelength sources associated with the cluster
    radio_idx, radio_sep, _ = cluster_sz_center.match_to_catalog_sky(sumss_coords)
    radio_idx = radio_idx.reshape(-1)
    radio_sources = sumss[radio_idx[radio_sep <= 2.5 * u.arcmin]]

    xray_idx, xray_sep, _ = cluster_sz_center.match_to_catalog_sky(rass2rxs_coords)
    xray_idx = xray_idx.reshape(-1)
    xray_sources = rass2rxs[xray_idx[xray_sep <= 2.5 * u.arcmin]]

    # Load in the IRAC image
    if any(sptsz_official_to_obs_id.get(cluster_id, 'SPTPOL_IMAGE') in s for s in sptsz_image_names):
        image_name = ''.join(s for s in sptsz_image_names if sptsz_official_to_obs_id[cluster_id] in s)
    else:
        image_name = ''.join(s for s in sptpol_image_names if cluster_id in s)

    image, header = fits.getdata(image_name, header=True)
    wcs = WCS(header)

    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    imshow_norm(image, ax=ax, origin='lower', cmap='Greys', stretch=LinearStretch(), interval=ZScaleInterval())
    ax.scatter(radio_sources['RA'], radio_sources['DEC'], marker='s', facecolor='none', edgecolor='red',
               transform=ax.get_transform('world'), label=f'SUMSS Source ({len(radio_sources)})')
    ax.scatter(cluster['ALPHA_J2000'], cluster['DELTA_J2000'], marker='o', facecolor='none', edgecolor='green',
               transform=ax.get_transform('world'), label=f'SPTcl-IRAGN ({len(cluster)})')
    ax.scatter(xray_sources['RA'], xray_sources['DEC'], marker='^', facecolor='none', edgecolor='blue',
               transform=ax.get_transform('world'), label=f'RASS2RXS Source ({len(xray_sources)})')
    ax.legend()
    ax.set(title=f'{cluster_id}', xlabel='RA', ylabel='Dec')
    plt.tight_layout()
    fig.savefig(f'Data/Plots/low-z_multiwavelength_comparison/{cluster_id}.png', bbox_inches='tight')
