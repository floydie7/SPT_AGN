"""
SPTcl_sky_errors.py
Author: Benjamin Floyd

Computes IRAC sky errors for the full SPTcl (SPT-SZ + SPTpol 100d) sample.
"""

import glob
import re
import warnings
from itertools import groupby
from astro_compendium.SkyError.SkyError import SkyError
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning

# Suppress Astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)


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


# Get lists of all images
spt_sz_files = glob.glob('Data_Repository/Images/SPT/Spitzer_IRAC/SPT-SZ_2500d/*.fits')
sptpol_files = glob.glob('Data_Repository/Images/SPT/Spitzer_IRAC/SPTpol_100d/*.fits')

# Compile the lists into manageable dictionaries keyed by the SPT ID
spt_sz = {cluster_id: list(files) for cluster_id, files in groupby(sorted(spt_sz_files, key=spt_id), key=spt_id)}
sptpol = {cluster_id: list(files) for cluster_id, files in groupby(sorted(sptpol_files, key=spt_id), key=spt_id)}

# Further categorize the files by type.
spt_sz = categorize_dict(spt_sz)
sptpol = categorize_dict(sptpol)

# Process the SPT-SZ sky errors
for cluster_id, images in spt_sz.items():
    I1_data, I1_header = fits.getdata(images['I1_data'], header=True, ignore_missing_end=True)
    I1_cov = fits.getdata(images['I1_cov'], ignore_missing_end=True)

    I2_data, I2_header = fits.getdata(images['I2_data'], header=True, ignore_missing_end=True)
    I2_cov = fits.getdata(images['I2_cov'], ignore_missing_end=True)

    I1_err = SkyError(I1_data, I1_header, I1_cov, bbox=(10, I1_data.shape[1] - 10, 10, I1_data.shape[0] - 10))
    I2_err = SkyError(I2_data, I2_header, I2_cov, bbox=(10, I2_data.shape[1] - 10, 10, I2_data.shape[0] - 10))

    # Compute flux errors
    I1_aper_flux_err, _ = I1_err.aperture_errors(aperture_diam=4 * u.arcsec, min_coverage=4.0,
                                              aperture_correction=10 ** (-0.4 * -0.38),
                                              nbins=60, hist_range=(-0.3, 0.3), hist_cutoff=0.05)
    I2_aper_flux_err, _ = I2_err.aperture_errors(aperture_diam=4 * u.arcsec, min_coverage=4.0,
                                              aperture_correction=10 ** (-0.4 * -0.4),
                                              nbins=60, hist_range=(-0.5, 0.5), hist_cutoff=0.02)

    # Make plots
    fig, ax = plt.subplots()
    I1_err.plot(ax=ax)
    ax.set(title=f'I1 Sky Error for {cluster_id}')
    plt.show()
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPT-SZ_2500d/plots/{cluster_id}_I1_sky_error.pdf')

    fig, ax = plt.subplots()
    I2_err.plot(ax=ax)
    ax.set(title=f'I2 Sky Error for {cluster_id}')
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPT-SZ_2500d/plots/{cluster_id}_I2_sky_error.pdf')

# Process the SPTpol 100d sky errors
for cluster_id, images in sptpol.items():
    I1_data, I1_header = fits.getdata(images['I1_data'], header=True)
    I1_cov = fits.getdata(images['I1_cov'])

    I2_data, I2_header = fits.getdata(images['I2_data'], header=True)
    I2_cov = fits.getdata(images['I2_cov'])

    I1_err = SkyError(I1_data, I1_header, I1_cov, bbox=(10, I1_data.shape[1] - 10, 10, I1_data.shape[0] - 10))
    I2_err = SkyError(I2_data, I2_header, I2_cov, bbox=(10, I2_data.shape[1] - 10, 10, I2_data.shape[0] - 10))

    # Compute flux errors
    I1_aper_flux_err = I1_err.aperture_errors(aperture_diam=4 * u.arcsec, min_coverage=4.0,
                                              aperture_correction=10 ** (-0.4 * -0.38),
                                              nbins=60, hist_range=(-0.2, 0.4), hist_cutoff=0.02)
    I2_aper_flux_err = I2_err.aperture_errors(aperture_diam=4 * u.arcsec, min_coverage=4.0,
                                              aperture_correction=10 ** (-0.4 * -0.4),
                                              nbins=60, hist_range=(-0.2, 0.4), hist_cutoff=0.02)

    # Make plots
    fig, ax = plt.subplots()
    I1_err.plot(ax=ax)
    ax.set(title=f'I1 Sky Error for {cluster_id}', xlim=[-1, 1])
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPTpol_100d/plots/{cluster_id}_I1_sky_error.pdf')

    fig, ax = plt.subplots()
    I2_err.plot(ax=ax)
    ax.set(title=f'I2 Sky Error for {cluster_id}', xlim=[-1, 1])
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPTpol_100d/plots/{cluster_id}_I2_sky_error.pdf')
