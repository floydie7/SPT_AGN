"""
.. Completeness_Simulation_Functions.py
.. Author: Benjamin Floyd
This file contains the functions needed to run a completeness simulation on a fits image.
"""

from __future__ import print_function, division

import os
from subprocess import Popen, PIPE

import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from pyraf import iraf


def run_sex(image, output_catalog, mag_zero, seeing_fwhm, sex_config='default.sex', param_file='default.param',
            _print=False):
    """
    Executes SExtractor to generate a single band catalog.

    :param image: 
        The path to the source image.
    :param output_catalog: 
        The path to where the output catalog will be placed.
    :param mag_zero:
        The zero point magnitude to be used by SExtractor.
    :param seeing_fwhm:
        The seeing FWHM for the image to be used by SExtractor.
    :param sex_config: 
        Name of the SExtractor configuration file. Defaults to 'default.sex'.
    :param param_file: 
        Name of the SExtractor parameter file used to sextract the image. Defaults to 'default.param'.
    :param _print:
        Boolean flag determining if the SExtractor output should be printed to the screen. Defaults to 'False'.
    :type image: str
    :type output_catalog: str
    :type mag_zero: float
    :type seeing_fwhm: float
    :type sex_config: str
    :type param_file: str
    :type _print: bool
    """

    # Store the current working directory so we can return to it when completed.
    proj_cwd = os.getcwd()

    # Define locations of files in absolute paths.
    se_executable = '/home/mei/bfloyd/bin/sex'
    # se_executable = '/opt/local/bin/sex'
    sex_config = os.path.abspath(sex_config)
    param_file = os.path.abspath(param_file)
    image = os.path.abspath(image)
    # weight = os.path.abspath(weight)
    output_catalog = os.path.abspath(output_catalog)

    # Convert the zero point magnitude and the seeing fwhm into strings
    mag_zero = str(mag_zero)
    seeing_fwhm = str(seeing_fwhm)

    # Move to the directory where the configuration files are
    os.chdir(os.path.dirname(sex_config))

    # Run the detection image in single image mode
    s_img = Popen([se_executable, image, '-c', sex_config, '-PARAMETERS_NAME', param_file,
                   '-CATALOG_NAME', output_catalog, '-MAG_ZEROPOINT', mag_zero, '-SEEING_FWHM', seeing_fwhm],
                  stdout=PIPE, stderr=PIPE)
    out, err = s_img.communicate()

    if _print:
        print(out, '\n', err)

    # Return to the project directory
    os.chdir(proj_cwd)


def make_stars(image, out_image, starlist_dir, model, fwhm, mag_zero, min_mag, max_mag, nstars,
               placement_bounds=None, _print=False):
    """
    Uses IRAF tasks (via PyRAF) starlist and mkobjects to generate artificial stars and place them on the image.

    :param image: 
        Path to the original image.
    :param out_image:
        Name of the output image.
    :param starlist_dir:
        Path to the directory of the starlist files.
    :param model:
        Model to use for the stars. Can be 'gaussian', 'moffat', or the path name to a PSF image.
    :param fwhm: 
        Full with, half maximum of the PSF of the input image.
    :param mag_zero:
        The appropriate zero-point magnitude for the pass band.
    :param min_mag: 
        Minimum magnitude allowed for the stars to have.
    :param max_mag: 
        Maximum magnitude allowed for the stars to have.
    :param nstars: 
        Number of stars to place on the image.
    :param placement_bounds:
        List of tuples indicating the bounds for allowable placements as (xmin, ymin, xmax, ymax).
    :param _print:
        Boolean flag determining if progress messages should be printed to the screen. Defaults to 'False'.
    :type image: str
    :type out_image: str
    :type starlist_dir: str
    :type model: str
    :type fwhm: float
    :type mag_zero: float
    :type min_mag: float
    :type max_mag: float
    :type nstars: int
    :type _print: bool
    """

    # Store the project working directory
    proj_cwd = os.getcwd()

    # Resolve the paths to the images to absolute path names
    image = os.path.abspath(image)
    out_image = os.path.abspath(out_image)

    # Get the absolute path names to the output directories
    starlist_dir = os.path.abspath(starlist_dir)

    # If a PSF is given then resolve the relative path to an absolute path
    if model not in ['gaussian', 'moffat']:
        model = os.path.abspath(model)

    # We need to load the correct IRAF packages in order to access the correct tasks.
    # Load NOAO IRAF package
    # iraf.noao(_doprint=0)

    # Load artdata subpackage
    iraf.artdata(_doprint=0)

    # Load the image WCS
    w = WCS(image)

    # Get the pixel scale
    try:
        assert w.pixel_scale_matrix[0, 1] == 0.
        pix_scale = (w.pixel_scale_matrix[1, 1] * w.wcs.cunit[1]).to_value(u.arcsec)
    except AssertionError:
        cd = w.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        pix_scale = (cd_diag[1, 1] * w.wcs.cunit[1]).to_value(u.arcsec)

    # # Get the zero point magnitude from the image.
    # zpoint = fits.getval(image, 'ZEROPT')

    if placement_bounds is None:
        # Get the bounds of the image.
        xlen, ylen = w._naxis1, w._naxis2
        xmin, xmax = 5., xlen - 5.
        ymin, ymax = 5., ylen - 5.
    else:
        xmin, ymin, xmax, ymax = placement_bounds

    # Radius used in gaussian of mkobjects is half of the FWHM of the PSF in pixels.
    radius = fwhm * 0.5 / pix_scale

    # Move to the Starlist directory to run the IRAF tasks.
    os.chdir(starlist_dir)

    # Name of starlist temporary file.
    star_fname = out_image[-30:-5] + '.dat'

    # Check to see if the starlist or if the output image exists and if true then delete them so that the IRAF tasks
    # don't try to append to them. IRAF tasks are stupid and don't just overwrite the file if it exists already.
    try:
        os.remove(out_image)
        os.remove(star_fname)
    except OSError:
        pass

    if _print:
        print('Generating star list.')

    # Generate the starlist.
    iraf.starlist(star_fname, nstars=nstars, spatial='uniform', xmin=xmin, xmax=xmax,
                  ymin=ymin, ymax=ymax, sseed='INDEF', luminosity='uniform', minmag=min_mag, maxmag=max_mag,
                  mzero=mag_zero, lseed='INDEF')

    if _print:
        print('Placing stars.')

    # Place the stars on the image.
    iraf.mkobjects(image, output=out_image, objects=star_fname, star=model, radius=radius, magzero=mag_zero,
                   seed='INDEF')

    # Return to the project directory
    os.chdir(proj_cwd)
