"""
SPTPol_SSDF_cutouts.py
Author: Benjamin Floyd

Uses the SPTPol 100d cluster catalog, the SSDF image tiles, and the SSDF photometric catalog to generate cutouts
centered around the SPTPol 100d clusters in both image and catalog spaces.
"""

import glob
import logging
from collections import defaultdict

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError, PartialOverlapError
from astropy.table import Table
from astropy.wcs import WCS

logger = logging.getLogger('SPTPol_cutouts.log')
logger.setLevel(logging.INFO)

hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
# hcc_prefix = ''

logger.info('Reading in source catalogs')
# Read in the SPTPol 100d cluster catalog
huang = Table.read(hcc_prefix + 'Data/sptpol100d_catalog_huang19.fits')

# Select only clusters with IR imaging
huang = huang[huang['imaging'] >= 2]

# Read in the SSDF photometric catalog
ssdf_template = Table.read(hcc_prefix+'Data/ssdf_table_template.cat', format='ascii.sextractor')
ssdf_catalog = pd.read_csv(hcc_prefix + 'Data/SPTPol/catalogs/SSDF2.20130918.v9.private.cat',
                           delim_whitespace=True, skiprows=52, names=ssdf_template.colnames)

logger.info('Performing cluster search')
# Make SkyCoords for the cluster centers and the SSDF objects
cluster_centers = SkyCoord(huang['RA'], huang['Dec'], unit='deg')
ssdf_coords = SkyCoord(ssdf_catalog['ALPHA_J2000'], ssdf_catalog['DELTA_J2000'], unit='deg')

# Search around each cluster to find the subset of objects nearby
idx_cluster, idx_ssdf_objs, sep, _ = ssdf_coords.search_around_sky(cluster_centers, 12 * u.arcmin)
assert np.all(sep < 12 * u.arcmin)

# Combine the idx arrays
idx_array = np.array([idx_cluster, idx_ssdf_objs]).T

# Set up a default dictionary so we can sort through the objects
cluster_idx_dict = defaultdict(list)
for cluster, ssdf_obj in idx_array:
    cluster_idx_dict[cluster].append(ssdf_obj)

logger.info('Beginning table and image cutouts')
# For each cluster, create a sub-table of the full SSDF catalog
for cluster_key, ssdf_obj_keys in cluster_idx_dict.items():
    spt_id = huang['SPT_ID'][cluster_key]
    logger.debug('Working on cluster: {}'.format(spt_id))
    cluster_objs = Table.from_pandas(ssdf_catalog.iloc[ssdf_obj_keys])
    tiles = np.unique(cluster_objs['TILE'])

    # For now we will only make cutouts of clusters that are fully contained within a single SSDF tile
    if len(tiles) == 1:
        tile_id = cluster_objs['TILE'][0]
        logger.debug('Cluster {spt_id} is located on SSDF tile {tile_id}'.format(spt_id=spt_id, tile_id=tile_id))

        # Read in the appropriate SSDF tiles
        ssdf_tile_files = glob.glob(hcc_prefix + 'Data/SPTPol/images/ssdf_tiles/*{tile}*.fits'.format(tile=tile_id))
        image_dict = {}
        for image in ssdf_tile_files:
            if 'I1' in image and 'cov' not in image:
                image_dict['I1_sci'] = fits.getdata(image, header=True)
            elif 'I1' in image and 'cov' in image:
                image_dict['I1_cov'] = fits.getdata(image, header=True)
            elif 'I2' in image and 'cov' not in image:
                image_dict['I2_sci'] = fits.getdata(image, header=True)
            elif 'I2' in image and 'cov' in image:
                image_dict['I2_cov'] = fits.getdata(image, header=True)

        for img_type, image_info in image_dict.items():
            data = image_info[0]
            header = image_info[1]
            wcs = WCS(header)

            # Our cutout will be centered on the cluster center and have an angular size of 16' x 16'
            position = cluster_centers[cluster_key]
            cutout_size = 16.0 * u.arcmin

            # Make the cutout
            try:
                cutout = Cutout2D(data, position, cutout_size, wcs=wcs, mode='strict')

                # Update the header from the original image
                header.update(cutout.wcs.to_header())

                # Create the HDU
                cutout_hdu = fits.PrimaryHDU(cutout.data, header=header)

                # The cutout names use the standard format used for the SPT-SZ cutouts
                cutout_fname = hcc_prefix + 'Data/SPTPol/images/cluster_cutouts/' \
                                            '{band}_{spt_id}_mosaic{cov}.cutout.fits' \
                    .format(band='I1' if 'I1' in img_type else 'I2',
                            spt_id=spt_id,
                            cov='_cov' if 'cov' in img_type else '')

                # Write the cutout to disk
                cutout_hdu.writeto(cutout_fname)

                if img_type == 'I2_sci':
                    # Update the x and y pixel coordinates in the catalog
                    new_xy_image = cutout.to_cutout_position((cluster_objs['X_IMAGE'], cluster_objs['Y_IMAGE']))

                    cluster_objs['X_IMAGE'] = new_xy_image[0]
                    cluster_objs['Y_IMAGE'] = new_xy_image[1]

                    # Trim the catalog to only include objects within the image
                    image_objs = np.where((cutout.xmin_cutout <= cluster_objs['X_IMAGE'] <= cutout.xmax_cutout) and
                                          (cutout.ymin_cutout <= cluster_objs['Y_NAME'] <= cutout.ymax_cutout))
                    cluster_objs = cluster_objs[image_objs]

                    # Write the catalog to disk using the standard format for the SPT-SZ cutouts
                    cluster_objs.write(hcc_prefix+'Data/SPTPol/catalogs/cutout_catalogs/{spt_id}.SSDFv9.cat',
                                       format='ascii')

            except NoOverlapError or PartialOverlapError:
                logger.warning('{spt_id} raised an OverlapError for image {img}'.format(spt_id=spt_id, img=img_type))
                continue

    else:
        logger.warning('{spt_id} has objects on multiple SSDF tiles: {tiles}'.format(spt_id=spt_id, tiles=tiles))
