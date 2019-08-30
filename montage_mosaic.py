"""
montage_mosaic.py
Author: Benjamin Floyd

Automates the Montage mosaicking process using the MontagePy API.
"""

import logging
import os
import shutil
import types
from functools import wraps

import MontagePy.main as m

# Set up logging for the module
log = logging.getLogger(__name__)


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

    # Clean up the work directory
    if clean_workdir:
        log.debug('Cleaning up work directory.')
        shutil.rmtree(workdir)
