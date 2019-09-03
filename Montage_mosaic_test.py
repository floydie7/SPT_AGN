"""
Montage_mosaic_test.py
Author: Benjamin Floyd

Integration test for the montage_mosaic.py API to see if it can complete the mosaicking process from start to finish.
"""

import logging
from itertools import chain

from montage_mosaic import montage_mosaic

# Set up logging
logging.basicConfig(filename='SPTPol_mosaics.log', filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    I1_sci_mosaic_name = out_dir + 'I1_{mosaic_id}_mosaic.fits'.format(mosaic_id=tile_mosaic_id)
    I1_cov_mosaic_name = out_dir + 'I1_{mosaic_id}_mosaic_cov.fits'.format(mosaic_id=tile_mosaic_id)
    I2_sci_mosaic_name = out_dir + 'I2_{mosaic_id}_mosaic.fits'.format(mosaic_id=tile_mosaic_id)
    I2_cov_mosaic_name = out_dir + 'I2_{mosaic_id}_mosaic_cov.fits'.format(mosaic_id=tile_mosaic_id)

    # Make the IRAC Channel 1 science mosaic
    montage_mosaic(I1_sci_imgs, out_file=I1_sci_mosaic_name, workdir=out_dir + 'I1_' + tile_mosaic_id + '_sci')

    # Make the IRAC Channel 1 coverage mosaic
    montage_mosaic(I1_cov_imgs, out_file=I1_cov_mosaic_name, workdir=out_dir + 'I1_' + tile_mosaic_id + '_cov')

    # Make the IRAC Channel 2 science mosaic
    montage_mosaic(I2_sci_imgs, out_file=I2_sci_mosaic_name, workdir=out_dir + 'I2_' + tile_mosaic_id + '_sci')

    # Make the IRAC Channel 2 coverage mosaic
    montage_mosaic(I2_cov_imgs, out_file=I2_cov_mosaic_name, workdir=out_dir + 'I2_' + tile_mosaic_id + '_cov')


hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
out_dir = hcc_prefix + 'SPTpol_mosaics/'

# Generate a full mosaic set for the tiles. (both IRAC science images and associated coverage maps)
tiles_to_mosaic_file = {'SSDF0.2_0.3': [['/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/ssdf_tiles/I1_SSDF0.2_mosaic.fits',
                                         '/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/ssdf_tiles/I1_SSDF0.2_mosaic_cov.fits',
                                         '/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/ssdf_tiles/I2_SSDF0.2_mosaic.fits',
                                         '/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/ssdf_tiles/I2_SSDF0.2_mosaic_cov.fits'],
                                        ['/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/ssdf_tiles/I1_SSDF0.3_mosaic.fits',
                                         '/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/ssdf_tiles/I1_SSDF0.3_mosaic_cov.fits',
                                         '/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/ssdf_tiles/I2_SSDF0.3_mosaic.fits',
                                         '/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/ssdf_tiles/I2_SSDF0.3_mosaic_cov.fits']]}

make_mosaics('SSDF0.2_0.3')
