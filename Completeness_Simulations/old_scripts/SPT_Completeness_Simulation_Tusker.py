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

from multiprocessing import Pool, cpu_count
from time import time

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.table import Table

from Completeness_Simulation_Functions import *
from Pipeline_functions import file_pairing, catalog_image_match


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
    sex_conf = '/work/mei/bfloyd/SPT_AGN/Data/Comp_Sim/sex_configs/default.sex'
    param_file = '/work/mei/bfloyd/SPT_AGN/Data/Comp_Sim/sex_configs/default.param'

    # Image parameters
    output_image = '/work/mei/bfloyd/SPT_AGN/Data/Comp_Sim/Images/{image_name}_stars.fits'\
        .format(image_name=image_name[-38:-19])
    starlist_dir = '/work/mei/bfloyd/SPT_AGN/Data/Comp_Sim/Starlists'

    # Altered image catalog
    alt_out_cat = '/work/mei/bfloyd/SPT_AGN/Data/Comp_Sim/sex_catalogs/{image_name}_stars.cat'.format(image_name=image_name[-38:-19])

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
            true_stars = ascii.read('{starlist_dir}/{image_name}_stars.dat'
                                    .format(starlist_dir=starlist_dir, image_name=image_name[-38:-19]),
                                    names=['x', 'y', 'selection_band'])
            alt_cat = ascii.read(alt_out_cat, format='sextractor')

            # Preform aperture correction to the magnitude based on the published values in Ashby et al. 2009
            alt_cat['MAG_APER'] += aper_corr  # From mag_auto - mag_aper values.

            # Match the coordinates between the truth catalog and the SExtractor catalog.
            max_sep = fwhm * u.arcsec

            wcs = WCS(output_image)
            true_coord = SkyCoord.from_pixel(true_stars['x'], true_stars['y'], wcs=wcs)
            cat_coord = SkyCoord(alt_cat['ALPHA_J2000'], alt_cat['DELTA_J2000'], unit=u.degree)

            idx, sep, _ = true_coord.match_to_catalog_sky(cat_coord)

            # Only accept objects that are within the maximum separation.
            alt_cat_matched = alt_cat[idx][np.where(sep <= max_sep)]

            # Require that the matched stars have magnitudes within 0.2 selection_band of the input magnitudes
            alt_cat_mag_matched = alt_cat_matched[
                np.where(np.abs(true_stars[np.where(sep <= max_sep)]['selection_band'] - alt_cat_matched['MAG_APER']) <= mag_diff)]

            # Append the number of placed and recovered objects into their respective containers.
            placed.append(len(true_stars))
            recovered.append(len(alt_cat_mag_matched))

        if len(placed) != 0:
            recovery_rate.append(np.sum(recovered) / np.sum(placed))

    # Create a dictionary entry with the image name as the key and the recovery_rate list as the value.
    # This will allow for the rates to be identifiable to the image they were created from.
    dict_rate = {image_name[-35:-19]: recovery_rate}

    return dict_rate


# Magnitude bins
bins = np.arange(10.0, 22.5, 0.5)

# Number of iterations per magnitude bin
nsteps = 100

# Magnitude threshold
mag_diff = 0.2

# Model type
model = 'gaussian'

# First we need to grab the images and match them to the Bleem catalog. For this, we'll use the SPT_AGN_Pipeline
# functions file_pairing and catalog_image_match.

# Pair the files together
cluster_list = file_pairing('/work/mei/bfloyd/SPT_AGN/Data/Catalogs/', '/work/mei/bfloyd/SPT_AGN/Data/Images/')

# Read in the Bleem catalog
Bleem = Table(fits.getdata('/work/mei/bfloyd/SPT_AGN/Data/2500d_cluster_sample_fiducial_cosmology.fits'))

# Clean the table of the rows without mass data. These are unconfirmed cluster candidates.
Bleem = Bleem[np.where(Bleem['M500'] != 0.0)]

# Match the clusters to the catalog requiring the cluster centers be within 1 arcminute of the Bleem center.
matched_list = catalog_image_match(cluster_list, Bleem, cat_ra_col='RA', cat_dec_col='DEC', max_sep=1.0)

# Channel 1 science images
ch1_images = [matched_list[k]['ch1_sci_path'] for k in range(len(matched_list))]

# Channel 2 science images
ch2_images = [matched_list[k]['ch2_sci_path'] for k in range(len(matched_list))]

# Set up multiprocessing pool
# get number of cpus available to job
try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split('(')[0])
except KeyError:
    ncpus = cpu_count()

pool = Pool(processes=ncpus)

for channel in range(2):

    # Record start time
    start_time = time()

    if channel == 0:
        image_list = ch1_images  # Set the image list to channel 1 images
        psf_fwhm = 1.66          # PSF FWHM in arcseconds
        zeropt = 17.997          # Zero-point magnitude
        aper_corr = -0.05        # Aperture correction
    else:
        image_list = ch2_images  # Set the image list to channel 2 images
        psf_fwhm = 1.72          # PSF FWHM in arcseconds
        zeropt = 17.538          # Zero-point magnitude
        aper_corr = -0.05        # Aperture correction

    # Apply the completeness function to the multiprocessing pool for all the images in the image list.
    rates = [pool.apply_async(completeness, args=(image_name, bins, nsteps, psf_fwhm, zeropt, aper_corr))
             for image_name in image_list]

    # Retrieve the return radial from the multiprocessing object.
    rates = [p.get() for p in rates]

    # Convert the multiple dictionary radial in the rates array into a single dictionary.
    # If this script is updated to Python 3.x then the following three lines can be replaced with
    # rates_dict = {**d for d in rates}
    rates_dict = {}
    for d in rates:
        rates_dict.update(d)

    # Add the magnitude values used to create the completeness rates.
    rates_dict.update({'magnitude_bins': bins})

    # Save array to disk
    np.save('/work/mei/bfloyd/SPT_AGN/Data/Comp_Sim/Results/SPT_I{ch}_results_{model}_fwhm{fwhm}_corr{corr}_mag{mag_diff}'
            .format(ch=channel + 1, model=model, fwhm=str(psf_fwhm).replace('.', ''),
                    corr=str(np.abs(aper_corr)).replace('.', ''), mag_diff=str(mag_diff).replace('.', '')), rates_dict)
