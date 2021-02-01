"""
SPTcl_sky_errors_qphot_post.py
Author: Benjamin Floyd

Runs post processing for the qphot version of the sky error script.
"""

import glob
import re
from itertools import groupby

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
import warnings

warnings.simplefilter('ignore', AstropyWarning)


def spt_id(f):
    return re.search(r'SPT-CLJ\d+-\d+(-\d+)?', f).group(0)


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


def gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def post_process(cluster_id, catalogs, min_cov, survey):
    I1_catalog_name = ''.join(s for s in catalogs if 'I1' in s)
    I2_catalog_name = ''.join(s for s in catalogs if 'I2' in s)
    I1_cov_name = f'Data_Repository/Images/SPT/Spitzer_IRAC/{survey}/I1_{cluster_id}_mosaic_cov.cutout.fits'
    I2_cov_name = f'Data_Repository/Images/SPT/Spitzer_IRAC/{survey}/I2_{cluster_id}_mosaic_cov.cutout.fits'

    # Read in the catalogs
    I1_catalog = Table.read(I1_catalog_name, format='ascii')
    I2_catalog = Table.read(I2_catalog_name, format='ascii')

    # Read in the coverage images
    I1_cov, header = fits.getdata(I1_cov_name, header=True, ignore_missing_end=True)
    I2_cov = fits.getdata(I2_cov_name, ignore_missing_end=True)

    # Get the pixel scale
    wcs = WCS(header)
    pscale = pixel_scale(wcs)

    # For some reason apertures have been allowed to be placed off image which causes an error in coverage filtering.
    # To avoid this we will first filter any apertures that have center coordinates outside the image bounds.
    I1_catalog = I1_catalog[np.all([0 <= I1_catalog['XCENTER'], I1_catalog['XCENTER'] <= wcs.pixel_shape[0],
                                    0 <= I1_catalog['YCENTER'], I1_catalog['YCENTER'] <= wcs.pixel_shape[1]], axis=0)]
    I2_catalog = I2_catalog[np.all([0 <= I2_catalog['XCENTER'], I2_catalog['XCENTER'] <= wcs.pixel_shape[0],
                                    0 <= I2_catalog['YCENTER'], I2_catalog['YCENTER'] <= wcs.pixel_shape[1]], axis=0)]

    # Select only apertures with good coverage at center
    I1_catalog = I1_catalog[I1_cov[np.floor(I1_catalog['YCENTER']).astype(int),
                                   np.floor(I1_catalog['XCENTER']).astype(int)] >= min_cov]
    I2_catalog = I2_catalog[I2_cov[np.floor(I2_catalog['YCENTER']).astype(int),
                                   np.floor(I2_catalog['XCENTER']).astype(int)] >= min_cov]

    # Bin the fluxes
    I1_hist, I1_bins = np.histogram(I1_catalog['FLUX'], bins=60, range=(-0.5, 0.5))
    I2_hist, I2_bins = np.histogram(I2_catalog['FLUX'], bins=60, range=(-0.5, 0.5))

    # Compute midpoints
    I1_bin_centers = I1_bins[:-1] + np.diff(I1_bins) / 2
    I2_bin_centers = I2_bins[:-1] + np.diff(I2_bins) / 2

    # Filter only up to the cutoff flux
    I1_bin_centers_fit = I1_bin_centers[I1_bin_centers <= 0.02]
    I1_hist_fit = I1_hist[I1_bin_centers <= 0.02]
    I2_bin_centers_fit = I2_bin_centers[I2_bin_centers <= 0.02]
    I2_hist_fit = I2_hist[I2_bin_centers <= 0.02]

    # Fit the data
    I1_popt, I1_pcov = curve_fit(gaussian, I1_bin_centers_fit, I1_hist_fit)
    I2_popt, I2_pcov = curve_fit(gaussian, I2_bin_centers_fit, I2_hist_fit)

    # Convert fluxes from native MJy/sr to uJy for plotting and final output
    flux_conv = u.Unit(u.MJy / u.sr).to(u.uJy / u.arcsec ** 2) * u.uJy / u.arcsec ** 2
    pixel_size = pscale[0][1].to(u.arcsec) ** 2 * u.arcsec ** 2
    I1_bin_centers_uJy = I1_bin_centers * flux_conv * pixel_size
    I2_bin_centers_uJy = I2_bin_centers * flux_conv * pixel_size

    # To make it clear what points were used to fit, split at the cutoff point
    I1_fit_bins_uJy = I1_bin_centers_fit * flux_conv * pixel_size
    I2_fit_bins_uJy = I2_bin_centers_fit * flux_conv * pixel_size

    # Also convert the relevant model parameters
    I1_mu_uJy = I1_popt[1] * flux_conv * pixel_size
    I1_sigma_uJy = np.abs(I1_popt[2] * flux_conv * pixel_size)
    I2_mu_uJy = I2_popt[1] * flux_conv * pixel_size
    I2_sigma_uJy = np.abs(I2_popt[2] * flux_conv * pixel_size)

    # Make the plots
    fig, (ax_I1, ax_I2) = plt.subplots(ncols=2, figsize=(16, 8))
    ax_I1.bar(I1_bin_centers_uJy, I1_hist, width=np.diff(I1_bin_centers_uJy)[0].value)
    ax_I1.plot(I1_fit_bins_uJy, gaussian(I1_fit_bins_uJy, I1_popt[0], I1_mu_uJy, I1_sigma_uJy), ls='-',
               color='C1', label=rf'Sky Error = {I1_sigma_uJy:.3f}' + '\n' +
                                 rf'$\mu\pm\sigma$ = ({I1_popt[1]:.2f}$\pm${np.abs(I1_popt[2]):.3f}) MJy/sr')
    ax_I1.plot(I1_bin_centers_uJy, gaussian(I1_bin_centers_uJy, I1_popt[0], I1_mu_uJy, I1_sigma_uJy), ls='--',
               color='C1')
    ax_I1.legend(loc='upper right', frameon=False)
    ax_I1.yaxis.set_visible(False)
    ax_I1.set(xlabel=r'$S_{{3.6\mu\rm m}}$ [uJy]')
    ax_I2.bar(I2_bin_centers_uJy, I2_hist, width=np.diff(I2_bin_centers_uJy)[0].value)
    ax_I2.plot(I2_fit_bins_uJy, gaussian(I2_fit_bins_uJy, I2_popt[0], I2_mu_uJy, I2_sigma_uJy), ls='-',
               color='C1', label=rf'Sky Error = {I2_sigma_uJy:.3f}' + '\n' +
                                 rf'$\mu\pm\sigma$ = ({I2_popt[1]:.2f}$\pm${np.abs(I2_popt[2]):.3f}) MJy/sr')
    ax_I2.plot(I2_bin_centers_uJy, gaussian(I2_bin_centers_uJy, I2_popt[0], I2_mu_uJy, I2_sigma_uJy), ls='--',
               color='C1')
    ax_I2.legend(loc='upper right', frameon=False)
    ax_I2.yaxis.set_visible(False)
    ax_I2.set(xlabel=r'$S_{{4.5\mu\rm m}}$ [uJy]')
    fig.suptitle(f'Sky Errors for {cluster_id}')
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/{survey}/plots/{cluster_id}_sky_errors_qphot.pdf')
    plt.close('all')

    return cluster_id, I1_popt[2], I1_sigma_uJy.value, I2_popt[2], I2_sigma_uJy.value


spt_sz_catalogs = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPT-SZ_2500d/qphot_catalogs/*.mag')
sptpol_catalogs = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPTpol_100d/qphot_catalogs/*.mag')

# Group by cluster
spt_sz = {cluster_id: list(files) for cluster_id, files in groupby(sorted(spt_sz_catalogs, key=spt_id), key=spt_id)}
sptpol = {cluster_id: list(files) for cluster_id, files in groupby(sorted(sptpol_catalogs, key=spt_id), key=spt_id)}

# Run post-processing and create tables of the sky errors (both in native MJy/sr and in true flux uJy)
spt_sz_sky_errors = Table(rows=[post_process(cluster_id, catalogs, min_cov=4, survey='SPT-SZ_2500d')
                                for cluster_id, catalogs in spt_sz.items()],
                          names=['SPT_ID', 'I1_SB_error', 'I1_flux_error', 'I2_SB_error', 'I2_flux_error'])
sptpol_sky_errors = Table(rows=[post_process(cluster_id, catalogs, min_cov=3, survey='SPTpol_100d')
                                for cluster_id, catalogs in sptpol.items()],
                          names=['SPT_ID', 'I1_SB_error', 'I1_flux_error', 'I2_SB_error', 'I2_flux_error'])
spt_sz_sky_errors.write('Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPT-SZ_2500d/SPT-SZ_sky_errors_qphot.fits',
                        overwrite=True)
sptpol_sky_errors.write('Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPTpol_100d/SPTpol_sky_errors_qphot.fits',
                        overwrite=True)
