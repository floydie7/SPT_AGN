"""
SPTcl_sky_errors_qphot.py
Author: Benjamin Floyd

A brief update to `SPT_Sky_Errors.py` to be able to run qphot sky errors for comparison to the new pure-python
`SkyError` package.
"""

from __future__ import print_function, division

import glob
import re
import sys
import warnings
from functools import partial
from itertools import groupby

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from schwimmbad import MPIPool

# Filter out Astropy warnings
warnings.simplefilter('ignore', AstropyWarning)


from Sky_Error_Functions import *


def spt_id(f):
    return re.search(r'SPT-CLJ\d+-\d+', f).group(0)


def categorize_dict(d):
    clusters_to_remove = []
    for cluster_id, files in d.items():
        d[cluster_id] = {}
        for f in files:
            if 'I1' in f and '_cov' not in f:
                d[cluster_id]['I1_data'] = f
            elif 'I1' in f and '_cov' in f:
                d[cluster_id]['I1_cov'] = f
            elif 'I2' in f and '_cov' not in f:
                d[cluster_id]['I2_data'] = f
            elif 'I2' in f and '_cov' in f:
                d[cluster_id]['I2_cov'] = f
        if len(d[cluster_id].keys()) != 4:
            clusters_to_remove.append(cluster_id)
    for cluster_id in clusters_to_remove:
        del d[cluster_id]
    return d


def pixel_scale(wcs):
    """Sets the pixel scale. Will diagonalize the pixel scale matrix if needed."""
    try:
        assert np.all(np.isclose([wcs.pixel_scale_matrix[0, 1], wcs.pixel_scale_matrix[1, 0]], 0.))
        return u.pixel_scale(wcs.pixel_scale_matrix[1, 1] * wcs.wcs.cunit[1] / u.pixel)
    except AssertionError:
        cd = wcs.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        return u.pixel_scale(cd_diag[1, 1] * wcs.wcs.cunit[1] / u.pixel)


def qphot_sky_errors(args, survey):
    cluster_id, image_dict = args

    # Get the file names
    I1_data = image_dict['I1_data']
    I1_cov = image_dict['I1_cov']
    I2_data = image_dict['I2_data']
    I2_cov = image_dict['I2_cov']

    # Set file names for the aperture coordinate and qphot catalog files
    I1_coo = 'coordinates/{survey}/I1_{cluster_id}.coo'.format(survey=survey, cluster_id=cluster_id)
    I2_coo = 'coordinates/{survey}/I2_{cluster_id}.coo'.format(survey=survey, cluster_id=cluster_id)
    I1_mag = 'qphot_catalogs/{survey}/I1_{cluster_id}.mag'.format(survey=survey, cluster_id=cluster_id)
    I2_mag = 'qphot_catalogs/{survey}/I2_{cluster_id}.mag'.format(survey=survey, cluster_id=cluster_id)

    # Get the WCS from the I1 image
    wcs = WCS(I1_data)

    # Convert the 4 arcsec aperture diameter to pixels
    aper_pix = (4 * u.arcsec).to(u.pixel, pixel_scale(wcs)).value

    # Conversion from native pixel units of MJy/sr to uJy
    flux_conv = 23.5045 * (pixel_scale(wcs)[0][1].to(u.arcsec))**2

    # Generate the aperture coordinate files
    generate_apertures(I1_data, output_coo=I1_coo, aper_size=aper_pix,
                       xmin=2 * aper_pix, xmax=wcs._naxis[0] - 2 * aper_pix,
                       ymin=2 * aper_pix, ymax=wcs._naxis[1] - 2 * aper_pix)
    generate_apertures(I2_data, output_coo=I2_coo, aper_size=aper_pix,
                       xmin=2 * aper_pix, xmax=wcs._naxis[0] - 2 * aper_pix,
                       ymin=2 * aper_pix, ymax=wcs._naxis[1] - 2 * aper_pix)

    # Run qhot on the images
    I1_zmag = 17.997 if survey == 'SPT-SZ' else 18.789
    I2_zmag = 17.538 if survey == 'SPT-SZ' else 18.316
    run_qphot(I1_data, coord_list=I1_coo, output_file=I1_mag, aper_size=aper_pix, zpt_mag=I1_zmag)
    run_qphot(I2_data, coord_list=I2_coo, output_file=I2_mag, aper_size=aper_pix, zpt_mag=I2_zmag)

    # Clean the qphot catalog to be useful for fitting
    cov = 4.0 if survey == 'SPT-SZ' else 3.0
    I1_flux_catalog = catalog_management(I1_mag, I1_cov, min_cov=cov)
    I2_flux_catalog = catalog_management(I2_mag, I2_cov, min_cov=cov)

    # Fit the data to a Gaussian model
    I1_bin_centers, I1_hist, I1_popt = fit_gaussian(I1_flux_catalog, nbins=60, hist_range=(-0.2, 0.4), cutoff=0.02)
    I2_bin_centers, I2_hist, I2_popt = fit_gaussian(I2_flux_catalog, nbins=60, hist_range=(-0.2, 0.4), cutoff=0.02)

    # Convert the sky error to uJy
    I1_sky_error = np.abs(I1_popt[1])
    I2_sky_error = np.abs(I2_popt[1])

    return cluster_id, I1_sky_error, I2_sky_error


# Get lists of all images
spt_sz_files = glob.glob('/work/mei/bfloyd/SPT_AGN/Data/Images/*.fits')
sptpol_files = glob.glob('/work/mei/bfloyd/SPT_AGN/Data/SPTPol/images/cluster_cutouts/*.fits')

# Compile the lists into manageable dictionaries keyed by the SPT ID
spt_sz = {cluster_id: list(files) for cluster_id, files in groupby(sorted(spt_sz_files, key=spt_id), key=spt_id)}
sptpol = {cluster_id: list(files) for cluster_id, files in groupby(sorted(sptpol_files, key=spt_id), key=spt_id)}

# Further categorize the files by type.
spt_sz = categorize_dict(spt_sz)
sptpol = categorize_dict(sptpol)

# Process the SPT-SZ sky errors
spt_sz_qphot_sky_errors = partial(qphot_sky_errors, survey='SPT-SZ')
sptpol_qphot_sky_errors = partial(qphot_sky_errors, survey='SPTpol')
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    spt_sz_pool_results = pool.map(spt_sz_qphot_sky_errors, spt_sz.items())
    sptpol_pool_results = pool.map(sptpol_qphot_sky_errors, sptpol.items())

    if pool.is_master():
        spt_sz_sky_errors = [cluster_results for cluster_results in spt_sz_pool_results]
        sptpol_sky_errors = [cluster_results for cluster_results in sptpol_pool_results]

# Cast the results to a Table and write to disk
spt_sz_sky_errors = Table(rows=spt_sz_sky_errors, names=['SPT_ID', 'I1_flux_err', 'I2_flux_err'])
sptpol_sky_errors = Table(rows=sptpol_sky_errors, names=['SPT_ID', 'I1_flux_err', 'I2_flux_err'])
spt_sz_sky_errors.write('SPT-SZ_sky_errors.fits')
sptpol_sky_errors.write('SPTpol_sky_errors.fits')
