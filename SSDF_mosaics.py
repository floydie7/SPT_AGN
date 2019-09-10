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
logger = logging.getLogger('node[{rank:d}]: {name}'.format(rank=comm.rank, name=__name__))
logger.setLevel(logging.DEBUG)
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


def make_mosaics(mosaic_label):
    logger.info('node[{rank:d}] working on mosaic: {mosaic_id}'.format(rank=comm.rank, mosaic_id=mosaic_label))
    tile_list, out_file, bkg_correction = mosaic_tasks[mosaic_label]

    montage_mosaic(tile_list, out_file=out_file, workdir=temp_work_dir+mosaic_label, correct_bg=bkg_correction)


hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
out_dir = hcc_prefix+'Data/SPTPol/images/mosaic_tiles/'
temp_work_dir = out_dir + 'temp_work_dirs/'

# Read the cutouts log to find the tiles needed to mosaic for each cluster
id_pattern = re.compile(r'SPT-CLJ\d+-\d+')
cluster_tiles = {}
with open('SPTPol_cutouts.log', 'r') as log:
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

# Skip mosaics with tile SSDF4.2 for now as the IRAC 1 coverage map is corrupted.
tiles_to_remove = []
for k in tiles_to_mosaic_file:
    if '4.2' in k:
        tiles_to_remove.append(k)
logger.info('Removing mosaic tiles: {} to avoid tile 4.2'.format(tiles_to_remove))
for k in tiles_to_remove:
    tiles_to_mosaic_file.pop(k, None)

mosaic_tasks = {}
for tile_mosaic_id in tiles_to_mosaic_file_test:
    file_set = tiles_to_mosaic_file[tile_mosaic_id]

    # Group the files into the four mosaic types we will make
    all_files = list(chain(*file_set))
    I1_sci_imgs = [img_name for img_name in all_files if 'I1' in img_name and 'cov' not in img_name]
    I1_cov_imgs = [img_name for img_name in all_files if 'I1' in img_name and 'cov' in img_name]
    I2_sci_imgs = [img_name for img_name in all_files if 'I2' in img_name and 'cov' not in img_name]
    I2_cov_imgs = [img_name for img_name in all_files if 'I2' in img_name and 'cov' in img_name]
    image_list = [I1_sci_imgs, I1_cov_imgs, I2_sci_imgs, I2_cov_imgs]

    # Mosaic file names
    I1_sci_mosaic_name = out_dir + 'I1_{mosaic_id}_mosaic.fits'.format(mosaic_id=tile_mosaic_id)
    I1_cov_mosaic_name = out_dir + 'I1_{mosaic_id}_mosaic_cov.fits'.format(mosaic_id=tile_mosaic_id)
    I2_sci_mosaic_name = out_dir + 'I2_{mosaic_id}_mosaic.fits'.format(mosaic_id=tile_mosaic_id)
    I2_cov_mosaic_name = out_dir + 'I2_{mosaic_id}_mosaic_cov.fits'.format(mosaic_id=tile_mosaic_id)
    mosaic_filenames = [I1_sci_mosaic_name, I1_cov_mosaic_name, I2_sci_mosaic_name, I2_cov_mosaic_name]

    # Mosaic labels
    I1_sci_label = 'I1_'+tile_mosaic_id+'_sci'
    I1_cov_label = 'I1_'+tile_mosaic_id+'_cov'
    I2_sci_label = 'I2_'+tile_mosaic_id+'_sci'
    I2_cov_label = 'I2_'+tile_mosaic_id+'_cov'
    labels = [I1_sci_label, I1_cov_label, I2_sci_label, I2_cov_label]

    mosaic_tasks.update({lbl: [img_list, mosaic_name, False if '_cov' in lbl else True]
                         for lbl, img_list, mosaic_name in zip(labels, image_list, mosaic_filenames)})


with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    pool.map(make_mosaics, mosaic_tasks)
