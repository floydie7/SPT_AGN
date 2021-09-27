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

from multiprocessing import Pool
from time import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from Completeness_Simulation.Completeness_Simulation_Functions import *
from Pipeline_functions import file_pairing, catalog_image_match
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.ticker import AutoMinorLocator


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
    image = 'Data/Images/{image_name}'.format(image_name=image_name[12:])
    sex_conf = 'Data/Comp_Sim/sex_configs/default.sex'
    param_file = 'Data/Comp_Sim/sex_configs/default.param'

    # Image parameters
    output_image = 'Data/Comp_Sim/Images/{image_name}_stars.fits'.format(image_name=image_name[12:-5])
    starlist_dir = 'Data/Comp_Sim/Starlists'

    # Altered image catalog
    alt_out_cat = 'Data/Comp_Sim/sex_catalogs/{image_name}_stars.cat'.format(image_name=image_name[12:-5])

    # print(image)
    # print(output_image)
    # print(alt_out_cat)
    # raise SystemExit

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
                                    .format(starlist_dir=starlist_dir, image_name=image_name[12:-5]),
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
            # print("Objects meeting separation criterion: ", len(alt_cat_matched))

            # print(true_stars[np.where(sep <= max_sep)]['selection_band'])
            # print(alt_cat_matched['MAG_APER'])
            # raise SystemExit

            # Require that the matched stars have magnitudes within 0.2 selection_band of the input magnitudes
            alt_cat_mag_matched = alt_cat_matched[
                np.where(np.abs(true_stars[np.where(sep <= max_sep)]['selection_band'] - alt_cat_matched['MAG_APER']) <= mag_diff)]

            # print("Objects meeting magnitude criterion: ", len(alt_cat_mag_matched))
            # Append the number of placed and recovered objects into their respective containers.
            placed.append(len(true_stars))
            recovered.append(len(alt_cat_mag_matched))

            # print("Placed: {0} Recovered: {1}".format(placed[i], recovered[i]))

        if len(placed) != 0:
            recovery_rate.append(np.sum(recovered) / np.sum(placed))

        #     print('Recovery rate within {0} - {1} selection_band: {2}'.format(min_mag, max_mag, recovery_rate[j]))
        # raise SystemExit

    # Create a dictionary entry with the image name as the key and the recovery_rate list as the value.
    # This will allow for the rates to be identifiable to the image they were created from.
    dict_rate = {image_name[12:-20]: recovery_rate}

    return dict_rate

# Magnitude bins
bins = np.arange(10.0, 22.5, 0.5)

# Number of iterations per magnitude bin
nsteps = 100

# Magnitude threshold
mag_diff = 0.2

# Model type
model = 'gaussian'

start_time = time()

# First we need to grab the images and match them to the Bleem catalog. For this, we'll use the SPT_AGN_Pipeline
# functions file_pairing and catalog_image_match.

# Pair the files together
print('Pairing Files.')
cluster_list = file_pairing('Data/Catalogs/', 'Data/Images/')

# Read in the Bleem catalog
Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))

# Match the clusters to the catalog requiring the cluster centers be within 1 arcminute of the Bleem center.
print('Matching catalogs.')
matched_list = catalog_image_match(cluster_list, Bleem, cat_ra_col='RA', cat_dec_col='DEC')
end_match_time = time()
print('Matching completed.\nImages in directory: {0}\tClusters matched: {1}\tTime Spent Matching: {2:.1f} s\n'
      .format(len(cluster_list), len(matched_list), end_match_time - start_time))

# Channel 1 science images
ch1_images = [matched_list[k]['ch1_sci_path'] for k in range(len(matched_list))]

# Channel 2 science images
ch2_images = [matched_list[k]['ch2_sci_path'] for k in range(len(matched_list))]
ch2_images = [ch2_images[0]]

# Set up multiprocessing pool
pool = Pool(processes=1)

# for channel in range(2):
for channel in [1]:

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
    np.save('Data/Comp_Sim/Results/SPT_I{ch}_results_{model}_fwhm{fwhm}_corr{corr}_mag{mag_diff}'
            .format(ch=channel + 1, model=model, fwhm=str(psf_fwhm).replace('.', ''),
                    corr=str(np.abs(aper_corr)).replace('.', ''), mag_diff=str(mag_diff).replace('.', '')), rates_dict)

    # Remove the magnitudes_bin entry to the dictionary for plot making.
    rates_dict.pop('magnitude_bins', None)

    # Find the median of the recovery rates.
    med_rates = [np.median(e) for e in zip(*rates_dict.values())]

    # Print the computation time
    end_time = time()
    print('Computation run time: {0} s'.format(end_time - start_time))

    # Make the plot
    # Add 0.25 to bins so that the data point is centered on the bin
    bins += 0.25

    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for curve in rates_dict:
        ax.plot(bins[:-1], rates_dict[curve], 'k-', alpha=0.4)
    ax.plot(bins[:-1], med_rates, 'r-', alpha=1.0, linewidth=2)

    ax.set(xlim=[10.0, 23.0], ylim=[0.0, 1.0], xlabel='Vega Magnitude', ylabel='Recovery Rate',
           title='Completeness Simulation for Channel {ch} SPT Clusters'.format(ch=channel + 1))
    fig.savefig('Data/Comp_Sim/Plots/SPT_Comp_Sim_I{ch}_{model}_fwhm{fwhm}_corr{corr}_mag{mag_diff}.pdf'
                .format(ch=channel + 1, model=model, fwhm=str(psf_fwhm).replace('.', ''),
                        corr=str(np.abs(aper_corr)).replace('.', ''), mag_diff=str(mag_diff).replace('.', '')),
                format='pdf')
