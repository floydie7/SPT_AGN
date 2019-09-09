"""
montage_mosaic.py
Author: Benjamin Floyd

Automates the Montage mosaicking process using the MontagePy API.
"""
import datetime
import glob
import logging
import os
import shutil
import types
from functools import wraps

import MontagePy.main as m
import numpy as np
from astropy.io import fits
from mpi4py import MPI

from mpi_logger import MPIFileHandler

# Set up logging for the module
comm = MPI.COMM_WORLD
log = logging.getLogger('node[{rank:d}]: montage_mosaic: {name}'.format(rank=comm.rank, name=__name__))
log.setLevel(logging.DEBUG)
mpi_handler = MPIFileHandler('SSDF_mosaics_funcs.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mpi_handler.setFormatter(formatter)
log.addHandler(mpi_handler)


# Add an exception to the Montage API by decorating the functions with a wrapper
class MontageError(Exception):
    pass


def decorate_montage(module, decorator):
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, types.BuiltinFunctionType):
            setattr(module, name, decorator(obj))


def montage_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        rtn = func(*args, **kwargs)
        if rtn['status'] == '1':
            log.critical('{func_name} raised : {error}'.format(func_name=func.__name__,
                                                               error=MontageError(rtn['msg'].decode('utf-8'))))
            raise MontageError(rtn['msg'].decode('utf-8'))
        else:
            log.info('{func_name}: {func_return}'.format(func_name=func.__name__, func_return=rtn))
    return wrapper


decorate_montage(m, montage_exception)


def montage_mosaic(tiles, out_file, quick_proj=False, coadd_type='average', correct_bg=True, workdir=None,
                   clean_workdir=True):
    """
    Automates the Montage mosaicking process on the input tiles.

    Parameters
    ----------
    tiles : list-like
        Input images to be mosaicked.
    out_file : str
        File name of the output mosaic image.
    quick_proj : bool, optional
        Flag to use the Quick Look projection method. Defaults to `False`.
    coadd_type : {'average', 'median', 'sum'}, optional
        Defines the coaddition type in stacking. Defaults to `'average'`.
    correct_bg : bool, optional
        Determines if we should background correct the images before coadding. Defaults to `True`.
    workdir : str, optional
        Name for temporary work directory. If not provided defaults to `'montage_workdir'`.
    clean_workdir : bool, optional
        Removes the temporary work directory structure once complete. Defaults to `True`.

    """

    coadd_dict = {'average': 0,
                  'median': 1,
                  'sum': 2}

    # Build the temporary work directory structure
    if workdir is None:
        workdir = 'montage_workdir'
        log.debug('Work directory not provided, using default: {}'.format(workdir))

    raw_dir = workdir + '/raw'
    projected_dir = workdir + '/projected'
    diff_dir = workdir + '/diffs'
    corrected_dir = workdir + '/corrected'

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(projected_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(corrected_dir, exist_ok=True)

    # Symlink the files into the raw directory
    for tile in tiles:
        os.symlink(os.path.abspath(tile), os.path.join(raw_dir, os.path.basename(tile)))
    log.debug('Working with tiles: {}'.format([os.path.basename(tile) for tile in tiles]))

    # Generate the metatable for the raw images
    log.debug('Generating raw image metatable.')
    raw_metatable = workdir + '/rImages.tbl'
    m.mImgtbl(raw_dir, raw_metatable)

    # Create the region header to cover the mosaic area
    log.debug('Generating region header.')
    region_hdr = workdir + '/region.hdr'
    m.mMakeHdr(raw_metatable, region_hdr)

    # Perform reprojectioning
    log.debug('Performing reprojectioning. Quick Look is set to: {}'.format(quick_proj))
    m.mProjExec(raw_dir, raw_metatable, region_hdr, projdir=projected_dir, quickMode=quick_proj)

    # Generate the metatable for the projected images
    log.debug('Generating projected image metatable.')
    projected_metatable = workdir + '/pImages.tbl'
    m.mImgtbl(projected_dir, projected_metatable)

    if correct_bg:
        log.debug('Background correction requested.')

        # Set any boundary pixels to NaNs to avoid background fitting
        for projected_image in glob.glob(projected_dir + '*.fits'):
            if 'area' not in projected_image:
                image_data, image_header = fits.getdata(projected_image, header=True)
                image_data_nan = np.where(image_data == 0, np.nan, image_data)
                hdu = fits.PrimaryHDU(image_data_nan, header=image_header)
                hdu.writeto(projected_image, overwrite=True)

        # Determine overlaps between the tiles
        log.debug('Calculating Overlap table.')
        diff_table = workdir + '/diffs.tbl'
        m.mOverlaps(projected_metatable, diff_table)

        # Generate the difference images and fit them
        log.debug('Fitting image overlaps and fitting the differences.')
        diff_fit_table = workdir + '/fits.tbl'
        m.mDiffFitExec(projected_dir, diff_table, region_hdr, diff_dir, diff_fit_table)

        # Model the background corrections
        log.debug('Modeling the background.')
        corrections_table = workdir + '/corrections.tbl'
        m.mBgModel(projected_metatable, diff_fit_table, corrections_table)

        # Background correct the projected images
        log.debug('Subtracting background fit.')
        m.mBgExec(projected_dir, projected_metatable, corrections_table, corrected_dir)

        # Create the metatable for the corrected images
        log.debug('Generating corrected image metatable')
        corrected_metatable = workdir + '/cImages.tbl'
        m.mImgtbl(corrected_dir, corrected_metatable)

        # Coadd the background-corrected, projected images
        log.debug('Coadding images using coadd type: {}'.format(coadd_type))
        m.mAdd(corrected_dir, corrected_metatable, region_hdr, out_file, coadd=coadd_dict[coadd_type])

    else:
        # Coadd the projected images without background corrections
        log.debug('No background correction requested. Coadding (uncorrected) projected images.')
        m.mAdd(projected_dir, projected_metatable, region_hdr, out_file, coadd=coadd_dict[coadd_type])

    log.info('Performing post-mosaicking processes.')
    # log.debug('Setting NaN values in mosaic to 0.')
    # Set any NaN values in the output file to "0" to conform with the original files.
    out_data, out_header = fits.getdata(out_file, header=True)
    # np.nan_to_num(out_data, nan=0, copy=False)  # Replace NaN values to 0 in-place.

    log.debug('Updating header with data from original header')
    # Read in one of the original file's header
    original_header = fits.getheader(glob.glob(raw_dir+'/*.fits')[0])

    # Using the mosaic header, update the WCS in the original header
    original_header.update(out_header)

    # Add comments to the header
    original_header['history'] = 'Mosaic created using MontagePy v{mpy_ver}; Montage v{m_ver}'.format(mpy_ver='1.2.0',
                                                                                                      m_ver='6.0')
    original_header['history'] = datetime.datetime.now().isoformat(' ', timespec='seconds')
    original_header['history'] = 'Created by Benjamin Floyd'
    original_header['history'] = 'Mosaic created by Benjamin Floyd'

    # Write modified file back to disk
    hdu = fits.PrimaryHDU(data=out_data, header=original_header)
    hdu.writeto(out_file, overwrite=True)

    # Clean up the work directory
    if clean_workdir:
        log.debug('Cleaning up work directory.')
        shutil.rmtree(workdir)
