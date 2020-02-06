"""
SPTPol_SSDF_cutouts.py
Author: Benjamin Floyd

Uses the SPTPol 100d cluster catalog, the SSDF image tiles, and the SSDF photometric catalog to generate cutouts
centered around the SPTPol 100d clusters in both image and catalog spaces.
"""

import glob
import logging
import re
from collections import defaultdict

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table, join
from astropy.wcs import WCS
from mpi4py import MPI
from mpipool import Pool

from mpi_logger import MPIFileHandler

# Set up logging
comm = MPI.COMM_WORLD
logger = logging.getLogger('node[{:d}]'.format(comm.rank))
logger.setLevel(logging.DEBUG)
mpi_handler = MPIFileHandler('SPTPol_cutouts_with_mosaics_common.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mpi_handler.setFormatter(formatter)
logger.addHandler(mpi_handler)

hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'


# hcc_prefix = '/Users/btfkwd/Documents/SPT_AGN/'


def make_cutout(cluster_key):
    # For each cluster, create a sub-table of the full SSDF catalog
    ssdf_obj_keys = cluster_idx_dict[cluster_key]
    # spt_id = huang['SPT_ID'][cluster_key]
    spt_id = common['SPT_ID'][cluster_key]
    logger.debug('Working on cluster: {}'.format(spt_id))
    cluster_objs = Table.from_pandas(ssdf_catalog.iloc[ssdf_obj_keys])
    tiles = np.unique(cluster_objs['TILE'])

    # For now we will only make cutouts of clusters that are fully contained within a single SSDF tile
    if len(tiles) == 1:
        tile_id = cluster_objs['TILE'][0]
        logger.debug('Cluster {spt_id} is located on SSDF tile {tile_id}'.format(spt_id=spt_id, tile_id=tile_id))

        # Read in the appropriate SSDF tiles
        ssdf_tile_files = glob.glob(hcc_prefix + 'Data/SPTPol/images/ssdf_tiles/*{tile}*.fits'.format(tile=tile_id))

    else:
        logger.info('{spt_id} has objects on multiple SSDF tiles: {tiles}. Switching to mosaics.'
                    .format(spt_id=spt_id, tiles=list(tiles)))
        tile_id = 'SSDF{}'.format('_'.join(sorted([re.search(r'\d\.\d', tile_id).group(0) for tile_id in tiles])))

        ssdf_tile_files = glob.glob(hcc_prefix + 'Data/SPTPol/images/mosaic_tiles/completed/*{tile}*.fits'.format(tile=tile_id))

        if not ssdf_tile_files:
            tile_id_nums = [re.search(r'\d\.\d', tile_id).group(0) for tile_id in tiles]
            ssdf_tile_files = [f for f in glob.glob(hcc_prefix + 'Data/SPTPol/images/mosaic_tiles/completed/*.fits')
                               if all(id_num in f for id_num in tile_id_nums)]

    image_dict = {}
    try:
        for image in ssdf_tile_files:
            if 'I1' in image and 'cov' not in image:
                image_dict['I1_sci'] = fits.getdata(image, header=True)
            elif 'I1' in image and 'cov' in image:
                image_dict['I1_cov'] = fits.getdata(image, header=True)
            elif 'I2' in image and 'cov' not in image:
                image_dict['I2_sci'] = fits.getdata(image, header=True)
            elif 'I2' in image and 'cov' in image:
                image_dict['I2_cov'] = fits.getdata(image, header=True)
    except TypeError as e:
        logger.warning('Cluster {spt_id} is located on SSDF tile {tile_id} and raised an TypeError: {error}. Skipping'
                       .format(spt_id=spt_id, tile_id=tile_id, error=e))

    for img_type, image_info in image_dict.items():
        data = image_info[0]
        header = image_info[1]
        wcs = WCS(header)

        # Our cutout will be centered on the cluster center and have an angular size of 16' x 16'
        position = cluster_centers[cluster_key]
        cutout_size = 5.0 * u.arcmin

        # Make the cutout
        cutout = Cutout2D(data, position, cutout_size, wcs=wcs, mode='partial')

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
        cutout_hdu.writeto(cutout_fname, overwrite=True)

        if img_type == 'I2_sci':
            # Update the x and y pixel coordinates in the catalog
            new_xy_image = cutout.wcs.wcs_world2pix(cluster_objs['ALPHA_J2000'], cluster_objs['DELTA_J2000'], 0)
            cluster_objs['X_IMAGE'] = new_xy_image[0]
            cluster_objs['Y_IMAGE'] = new_xy_image[1]

            # Trim the catalog to only include objects within the image
            image_objs = ((cutout.xmin_cutout <= cluster_objs['X_IMAGE']) &
                          (cluster_objs['X_IMAGE'] <= cutout.xmax_cutout) &
                          (cutout.ymin_cutout <= cluster_objs['Y_IMAGE']) &
                          (cluster_objs['Y_IMAGE'] <= cutout.ymax_cutout))
            cluster_objs = cluster_objs[image_objs]

            # Add the units and description information from the template
            for col_cluster_objs, col_ssdf_template in zip(cluster_objs.itercols(), ssdf_template.itercols()):
                col_cluster_objs.unit = col_ssdf_template.unit
                col_cluster_objs.description = col_ssdf_template.description

            # Write the catalog to disk using the standard format for the SPT-SZ cutouts
            cluster_objs.write(hcc_prefix + 'Data/SPTPol/catalogs/cluster_cutouts/{spt_id}.SSDFv9.fits'
                               .format(spt_id=spt_id), overwrite=True)


logger.info('Reading in source catalogs')
# Read in the SPT-SZ 2500d cluster catalog
# bocquet = Table.read(hcc_prefix + 'Data/2500d_cluster_sample_Bocquet18.fits')

# Read in the SPTPol 100d cluster catalog
# huang = Table.read(hcc_prefix + 'Data/sptpol100d_catalog_huang19.fits')

# Read in the common cluster catalog
common = Table.read(hcc_prefix + 'Data/common_clusters_2500d_100d.fits')

# Select only clusters that have not already been discovered in SPT-SZ 2500d
# huang = setdiff(huang, bocquet, keys=['SPT_ID'])

# Select only clusters with IR imaging
# huang = huang[huang['imaging'] >= 2]
common = common[common['imaging_Huang'] > 2]

# Select only clusters with redshift and mass information (confirmed clusters)
# huang = huang[huang['redshift'] > 0]
common = common[common['REDSHIFT_Huang'] > 0]

# Remove any clusters that we've already processed
# cluster_id_pattern = re.compile(r'SPT-CLJ\d+-\d+')
# completed_filenames = glob.glob(hcc_prefix + 'Data/SPTPol/catalogs/cluster_cutouts/*.SSDFv9.fits')
# completed_ids = Table([[cluster_id_pattern.search(f).group(0) for f in completed_filenames]], names=['SPT_ID'])
# huang = setdiff(huang, completed_ids, keys='SPT_ID')

# Read in the SSDF photometric catalog
ssdf_template = Table.read(hcc_prefix + 'Data/ssdf_table_template.cat', format='ascii.sextractor')
ssdf_catalog = pd.read_csv(hcc_prefix + 'Data/SPTPol/catalogs/SSDF2.20130918.v9.private.cat',
                           delim_whitespace=True, skiprows=52, names=ssdf_template.colnames)

logger.info('Performing cluster search')
# Make SkyCoords for the cluster centers and the SSDF objects
# cluster_centers = SkyCoord(huang['RA'], huang['Dec'], unit='deg')
cluster_centers = SkyCoord(common['RA_Huang'], common['DEC_Huang'], unit='dec')
ssdf_coords = SkyCoord(ssdf_catalog['ALPHA_J2000'], ssdf_catalog['DELTA_J2000'], unit='deg')

# Search around each cluster to find the subset of objects nearby
idx_cluster, idx_ssdf_objs, sep, _ = ssdf_coords.search_around_sky(cluster_centers, 5 * u.arcmin)
assert np.all(sep < 5 * u.arcmin)

# Combine the idx arrays
idx_array = np.array([idx_cluster, idx_ssdf_objs]).T

# Set up a default dictionary so we can sort through the objects
cluster_idx_dict = defaultdict(list)
for cluster, ssdf_obj in idx_array:
    cluster_idx_dict[cluster].append(ssdf_obj)

logger.info('Beginning table and image cutouts')
with Pool() as pool:
    # if not pool.is_master():
    #     pool.wait()
    #     sys.exit(0)

    pool.map(make_cutout, cluster_idx_dict)
