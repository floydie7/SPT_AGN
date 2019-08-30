"""
SSDF_mosaics.py
Author: Benjamin Floyd

Generates mosaics using Montage from the SSDF tiles where SPTPol 100d clusters are located within the tile boundaries.
"""

import glob
import logging
import re
import sys
from itertools import product, chain

from mpi4py import MPI
from schwimmbad import MPIPool

from montage_mosaic import montage_mosaic
from mpi_logger import MPIFileHandler

# Set up logging
comm = MPI.COMM_WORLD
logger = logging.getLogger('node[{:d}]'.format(comm.rank))
logger.setLevel(logging.INFO)
mpi_handler = MPIFileHandler('SSDF_mosaics.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mpi_handler.setFormatter(formatter)
logger.addHandler(mpi_handler)


def multiple_tiles(f):
    buffer = []
    for ln in f:
        if 'WARNING' in ln:
            if buffer:
                yield buffer
            buffer = [ln]
        else:
            buffer.append(ln)
    yield buffer


def make_mosaics(tile_mosaic_id):
    # Get the file lists
    file_set = tiles_to_mosaic_file[tile_mosaic_id]

    # Group the files into the four mosaic types we will make
    all_files = list(chain(*file_set))
    I1_sci_imgs = [img_name for img_name in all_files if 'I1' in img_name and 'cov' not in img_name]
    I1_cov_imgs = [img_name for img_name in all_files if 'I1' in img_name and 'cov' in img_name]
    I2_sci_imgs = [img_name for img_name in all_files if 'I2' in img_name and 'cov' not in img_name]
    I2_cov_imgs = [img_name for img_name in all_files if 'I2' in img_name and 'cov' in img_name]

    # Mosaic file names
    I1_sci_mosaic_name = out_dir+'I1_{mosaic_id}_mosaic.fits'.format(mosaic_id=tile_mosaic_id)
    I1_cov_mosaic_name = out_dir+'I1_{mosaic_id}_mosaic_cov.fits'.format(mosaic_id=tile_mosaic_id)
    I2_sci_mosaic_name = out_dir+'I2_{mosaic_id}_mosaic.fits'.format(mosaic_id=tile_mosaic_id)
    I2_cov_mosaic_name = out_dir+'I2_{mosaic_id}_mosaic_cov.fits'.format(mosaic_id=tile_mosaic_id)

    # Make the IRAC Channel 1 science mosaic
    montage_mosaic(I1_sci_imgs, out_file=I1_sci_mosaic_name, workdir=out_dir+'I1_'+tile_mosaic_id+'_sci')

    # Make the IRAC Channel 1 coverage mosaic
    montage_mosaic(I1_cov_imgs, out_file=I1_cov_mosaic_name, workdir=out_dir+'I1_'+tile_mosaic_id+'_cov')

    # Make the IRAC Channel 2 science mosaic
    montage_mosaic(I2_sci_imgs, out_file=I2_sci_mosaic_name, workdir=out_dir+'I2_'+tile_mosaic_id+'_sci')

    # Make the IRAC Channel 2 coverage mosaic
    montage_mosaic(I2_cov_imgs, out_file=I2_cov_mosaic_name, workdir=out_dir+'I2_'+tile_mosaic_id+'_cov')


hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
out_dir = hcc_prefix+'Data/SPTPol/images/mosaic_tiles/'

# Read the cutouts log to find the tiles needed to mosaic for each cluster
id_pattern = re.compile(r'SPT-CLJ\d+-\d+')
cluster_tiles = {}
with open('Data/SPTPol_cutouts.log', 'r') as log:
    for warnings_tiles in multiple_tiles(log):
        if 'TILE' in warnings_tiles[0]:
            cluster_id = id_pattern.search(warnings_tiles[0]).group(0)
            tile_names = [tile.strip() for tile in warnings_tiles[1:][1:]]
            cluster_tiles[cluster_id] = tile_names

# Find the minimum tile sets needed to make mosaics from
tile_set_set = set(frozenset(tile_group) for tile_group in cluster_tiles.values())
tile_subsets = set(set_element[0] for set_element in product(tile_set_set, repeat=2)
                   if set_element[0] != set_element[1] and set_element[0].issubset(set_element[1]))
tiles_to_mosaic = tile_set_set - tile_subsets

ssdf_tile_dir = hcc_prefix + 'Data/SPTPol/images/ssdf_tiles'
file_names = [[glob.glob(ssdf_tile_dir + '/*{tile}*'.format(tile=tile)) for tile in tile_set]
              for tile_set in tiles_to_mosaic]
tiles_to_mosaic_file = {'SSDF{}'.format('_'.join(sorted([re.search(r'\d\.\d', fname[0]).group(0) for fname in file_set]))):
                        file_set for file_set in file_names}

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    pool.map(make_mosaics, tiles_to_mosaic_file)
