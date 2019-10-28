"""
SPT_Completeness_Simulation.py
Author: Benjamin Floyd

This script executes the functions written in Completeness_Simulation.Completeness_Simulation_Functions.py and generates
completeness curves for all the SPT clusters found in the Bleem survey.

The completeness simulation is preformed by using the IRAF tasks noao.artdata.starlists and noao.artdata.mkobjects.
The starlist task generates a list of random coordinates and magnitudes within the specified bounds where the artificial
stars will be placed. In all cases the spatial and magnitude values were drawn from a uniform distribution with the
spatial values bounded between a buffer of 5 pixels on either side in both axes. The magnitude values were bounded
between the specified minimum and maximum magnitudes which were iterated from the Vega magnitudes of 10.0 selection_band to
23.0 selection_band by 0.5 selection_band intervals. The mkobjects task created a new image based on the existing input image with the
addition of artificial stars with positions and magnitudes provided by the output of the starlist task using the
specified model and point spread function (psf) full-width at half-maximum in arcseconds.

Source Extractor (SExtractor) was then run on the image creating a catalog of all detected objects. Both the input list
created by the starlist task and the SExtractor catalog were read in using the astropy.io.ascii.read function, the
aperture magnitudes were corrected, and then a catalog matching procedure was preformed between the two catalogs to find
the placed artificial stars. Only objects that had a maximum separation of 1 psf and had an input - output magnitude
difference less than the specified value were reported as being recovered. The number of recovered objects was recorded
for each magnitude bin and stored in a dictionary with the image name used as the key.
"""

from __future__ import print_function, division

import glob
import json
import re
from time import time

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from schwimmbad import MPIPool

from Completeness_Simulation_Functions import *

try:
    import sys
    assert sys.version_info() == (2, 7)
except AssertionError:
    raise AssertionError('This script must be ran on Python 2.7')

hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'


def completeness(image_name, bins, nsteps, fwhm, mag_zero, aper_corr, mag_diff=0.2, model='gaussian'):
    """
    Generates a completeness simulation on one image.
    :param image_name:
        File name for the image to be processed.
    :param bins:
        Array of magnitude bins.
    :param nsteps:
        Number of star placements to preform per magnitude bin.
    :param fwhm:
        The PSF full-width at half maximum to be used in the modeling of the stars.
    :param mag_zero:
        The appropriate zero-point magnitude for the band.
    :param aper_corr:
        The aperture correction for the measurement magnitude.
    :param mag_diff:
        The magnitude difference tolerance between the "true" magnitudes and the measured magnitudes. Defaults to 0.2.
    :param model:
        The model to use to generate the artificial stars. Can be 'gaussian' or 'moffat'. Defaults to 'gaussian'.
    :return dict_rate:
        A dictionary the image name as the key and the recovery rates in a list as the value.
    :type image_name: str
    :type bins: numpy.ndarray
    :type nsteps: int
    :type fwhm: float
    :type aper_corr: float
    :type mag_diff: float
    :type model: str
    :rtype dict:
    """
    recovery_rate = []

    # Paths to files
    image = image_name
    sex_conf = '/work/mei/bfloyd/SPT_AGN/Data/Comp_Sim/SPTpol/sex_configs/default.sex'
    param_file = '/work/mei/bfloyd/SPT_AGN/Data/Comp_Sim/SPTpol/sex_configs/default.param'

    # Cluster Image ID
    image_id = cluster_image.search(image_name).group(0)
    cluster_id = cluster_image.search(image_name).group(1)

    # Image parameters
    output_image = hcc_prefix + 'Data/Comp_Sim/SPTpol/Images/{image_id}_stars.fits'.format(image_id=image_id)
    starlist_dir = hcc_prefix + 'Data/Comp_Sim/SPTpol/Starlists'

    # Altered image catalog
    alt_out_cat = hcc_prefix + 'Data/Comp_Sim/SPTpol/sex_catalogs/{image_id}_stars.cat'.format(image_id=image_id)

    for j in range(len(bins)-1):
        # Set magnitude range for bin
        min_mag = bins[j]
        max_mag = bins[j+1]

        # Store the number of objects that were placed and the number of objects that were recovered.
        placed = []
        recovered = []

        for i in range(nsteps):

            # Now generate the image with artificial stars.
            make_stars(image, output_image, starlist_dir, model=model, fwhm=fwhm, mag_zero=mag_zero, min_mag=min_mag,
                       max_mag=max_mag, nstars=10)

            # Run SExtractor again on the altered image
            run_sex(output_image, alt_out_cat, mag_zero=mag_zero, seeing_fwhm=fwhm, sex_config=sex_conf,
                    param_file=param_file)

            # Read in both the starlist as a truth catalog and the altered image catalog
            true_stars = Table.read('{starlist_dir}/{image_id}_stars.dat'.format(starlist_dir=starlist_dir, image_id=image_id),
                                    names=['x', 'y', 'selection_band'], format='ascii')
            altered_cat = Table.read(alt_out_cat, format='ascii.sextractor')

            # Preform aperture correction to the magnitude based on the published values in Ashby et al. 2009
            altered_cat['MAG_APER'] += aper_corr  # From mag_auto - mag_aper values.

            # Match the coordinates between the truth catalog and the SExtractor catalog.
            max_sep = fwhm * u.arcsec

            wcs = WCS(output_image)
            true_coord = SkyCoord.from_pixel(true_stars['x'], true_stars['y'], wcs=wcs)
            cat_coord = SkyCoord(altered_cat['ALPHA_J2000'], altered_cat['DELTA_J2000'], unit=u.degree)

            idx, sep, _ = true_coord.match_to_catalog_sky(cat_coord)

            # Only accept objects that are within the maximum separation.
            alt_cat_matched = altered_cat[idx][sep <= max_sep]
            true_stars_matched = true_stars[sep <= max_sep]

            # Require that the matched stars have magnitudes within 0.2 selection_band of the input magnitudes
            alt_cat_mag_matched = alt_cat_matched[
               np.abs(true_stars_matched['selection_band'] - alt_cat_matched['MAG_APER']) <= mag_diff]

            # Append the number of placed and recovered objects into their respective containers.
            placed.append(len(true_stars))
            recovered.append(len(alt_cat_mag_matched))

        if len(placed) != 0:
            recovery_rate.append(np.sum(recovered) / np.sum(placed))

    # Create a dictionary entry with the image name as the key and the recovery_rate list as the value.
    # This will allow for the rates to be identifiable to the image they were created from.
    dict_rate = {cluster_id: recovery_rate}

    return dict_rate


#  RegEx for finding the cluster/image ids
cluster_image = re.compile(r'I[12]_(SPT-CLJ\d+-\d+)')

# Magnitude bins
bins = np.arange(10.0, 22.5, 0.5)

# Number of iterations per magnitude bin
nsteps = 100

# Magnitude threshold
mag_diff = 0.2

# Model type
model = 'gaussian'

# Image directory
image_dir = hcc_prefix + 'Data/SPTPol/images/cluster_cutouts'

# Channel 1 science images
ch1_images = glob.glob(image_dir + '/I1*_mosaic.cutout.fits')

# Channel 2 science images
ch2_images = glob.glob(image_dir + '/I2*_mosaic.cutout.fits')

irac_data_sptsz = {1: {'psf_fwhm': 1.95, 'zeropt': 17.997, 'aper_corr': -0.1},
                   2: {'psf_fwhm': 2.02, 'zeropt': 17.538, 'aper_corr': -0.11}}

irac_data_sptpol = {1: {'psf_fwhm': 1.95, 'zeropt': 18.789, 'aper_corr': -0.05},
                    2: {'psf_fwhm': 2.02, 'zeropt': 18.316, 'aper_corr': -0.38}}


# Record start time
start_time = time()

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    args = [(image_name, bins, nsteps, irac_data_sptpol[2]['psf_fwhm'], irac_data_sptpol[2]['zeropt'],
             irac_data_sptpol[2]['aper_corr'], mag_diff, model) for image_name in ch2_images]
    pool_results = pool.map(completeness, args)

    if pool.is_master():
        completeness_results = {cluster_id: recovery_rates for result in pool_results
                                for cluster_id, recovery_rates in result.items()}
print('Simulation run time: {}'.format(time() - start_time))

# Add the magnitude values used to create the completeness rates.
completeness_results['magnitude_bins'] = bins

# Save results to disk
results_filename = hcc_prefix + 'Data/Comp_Sim/SPTpol/Results/SPT_I2_results_{model}_fwhm{fwhm}_corr{corr}_mag{mag_diff}'\
    .format(model=model, fwhm=irac_data_sptpol[2]['psf_fwhm'], corr=irac_data_sptpol[2]['aper_corr'], mag_diff=mag_diff)
with open(results_filename, 'w') as f:
    json.dump(completeness_results, f, ensure_ascii=False, indent=4)
