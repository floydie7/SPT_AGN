"""
Completeness_mag_scatter.py
Author: Benjamin Floyd
Test script to determine the scatter in input versus output magnitude using the functions in 
Completeness_Simulation_Functions.py
"""

from __future__ import print_function, division

import glob

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Completeness_Simulation_Functions import *
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.wcs import WCS
from matplotlib.ticker import AutoMinorLocator

matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Magnitude bins
bins = np.arange(10.0, 23.0, 0.5)

# Image directory
image_dir = 'Data/SPTPol/images/cluster_cutouts'

# Channel 1 science images
ch1_images = glob.glob(image_dir + '/I1*_mosaic.cutout.fits')

# Channel 2 science images
ch2_images = glob.glob(image_dir + '/I2*_mosaic.cutout.fits')

# Select a number of images randomly from the I2_img_list up to the number of magnitude bins.
# image_list = [ch1_images[i] for i in np.random.random_integers(len(ch1_images)-1, size=len(bins))]
image_list = [ch2_images[i] for i in np.random.random_integers(len(ch2_images)-1, size=len(bins))]

print(image_list)

mag_zero = 18.316

# for fwhm in [1.95]:  # Warm mission Ch1 psf
for fwhm in [2.02]:  # Warm mission Ch2 psf

    mag_aper_diff = []
    mag_auto_diff = []
    mag_auto_aper = []
    mag_auto_aper_corr = []
    true_mag = []
    mag_auto = []

    for j in range(len(bins)-1):
        # Set magnitude range for bin
        min_mag = bins[j]
        max_mag = bins[j + 1]

        # Only use one image per magnitude bin
        image_name = image_list[j]

        # Paths to files
        image = image_name
        sex_conf = 'Data/Comp_Sim/SPTpol/sex_configs/default.sex'
        param_file = 'Data/Comp_Sim/SPTpol/sex_configs/default.param'

        # Image parameters
        output_image = 'Data/Comp_Sim/SPTpol/Images/{image_name}_stars.fits'.format(image_name=image_name[-38:-19])
        starlist_dir = 'Data/Comp_Sim/SPTpol/Starlists'

        # Altered image catalog
        alt_out_cat = 'Data/Comp_Sim/SPTpol/sex_catalogs/{image_name}_stars.cat'.format(image_name=image_name[-38:-19])

        # Generate the image with artificial stars.
        make_stars(image, output_image, starlist_dir, model='gaussian', fwhm=fwhm, mag_zero=mag_zero, min_mag=min_mag,
                   max_mag=max_mag, nstars=10)

        # Run SExtractor again on the altered image
        run_sex(output_image, alt_out_cat, mag_zero=mag_zero, seeing_fwhm=fwhm, sex_config=sex_conf,
                param_file=param_file)

        # Read in both the starlist as a truth catalog and the altered image catalog
        true_stars = ascii.read(starlist_dir + '/{image_name}_stars.dat'.format(image_name=image_name[-38:-19]),
                                names=['x', 'y', 'selection_band'])
        alt_cat = ascii.read(alt_out_cat, format='sextractor')

        # Match the coordinates between the truth catalog and the SExtractor catalog.
        max_sep = fwhm * u.arcsec
        wcs = WCS(output_image)
        true_coord = SkyCoord.from_pixel(true_stars['x'], true_stars['y'], wcs=wcs)
        cat_coord = SkyCoord(alt_cat['ALPHA_J2000'], alt_cat['DELTA_J2000'], unit=u.degree)

        idx, sep, _ = true_coord.match_to_catalog_sky(cat_coord)

        # Only accept objects that are within the maximum separation.
        alt_cat_matched = alt_cat[idx][np.where(sep <= max_sep)]

        # Define a magnitude difference
        true_mag.append(true_stars[np.where(sep <= max_sep)]['selection_band'])
        mag_auto.append(alt_cat_matched['MAG_AUTO'])
        mag_auto_aper.append(alt_cat_matched['MAG_AUTO'] - alt_cat_matched['MAG_APER'])
        # Preform aperture correction to the magnitude based on the published values in Ashby et al. 2009
        # alt_cat_matched['MAG_APER'] += -0.40
        # mag_auto_aper_corr.append(alt_cat_matched['MAG_AUTO'] - alt_cat_matched['MAG_APER'])
        mag_aper_diff.append(true_stars[np.where(sep <= max_sep)]['selection_band'] - alt_cat_matched['MAG_APER'])
        mag_auto_diff.append(true_stars[np.where(sep <= max_sep)]['selection_band'] - alt_cat_matched['MAG_AUTO'])

    # Make the plot
    fig, ax = plt.subplots()
    ax.grid()
    for j in range(len(bins)-1):
        ax.scatter(true_mag[j], mag_aper_diff[j], c='k', edgecolors='face')
    ax.set(title='Randomly selected Ch2 stamps', xlabel='true_mag (Vega)', ylabel='true_mag - mag_aper(4",uncorrected)')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig('Data/Comp_Sim/SPTpol/Plots/I2_true_mag_aper_scatter_fwhm{fwhm}_gauss.pdf'.format(fwhm=fwhm), format='pdf')
    # plt.show()
    #
    fig, ax = plt.subplots()
    ax.grid()
    for j in range(len(bins)-1):
        ax.scatter(true_mag[j], mag_auto_diff[j], c='k', edgecolors='face')
    ax.set(title='Randomly selected Ch2 stamps', xlabel='true_mag (Vega)', ylabel='true_mag - mag_auto')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig('Data/Comp_Sim/SPTpol/Plots/I2_true_mag_auto_scatter_fwhm{fwhm}_gauss.pdf'.format(fwhm=fwhm), format='pdf')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.grid()
    # for j in range(len(bins)-1):
    #     ax.scatter(mag_auto[j], mag_auto_aper[j], c='k', edgecolors='face')
    # ax.plot([10.0,23.0], [-0.38, -0.38], 'r-')
    # ax.set(title='Randomly selected Ch1 stamps', xlabel='mag_auto (Vega)', ylabel='mag_auto - mag_aper(4",uncorrected)')
    # fig.savefig('Data/Comp_Sim/Plots/I2_mag_auto_aper_scatter_fwhm'+str(fwhm)+'_gauss.pdf', format='pdf')
    # # plt.show()

    fig, ax = plt.subplots()
    ax.grid()
    for j in range(len(bins)-1):
        ax.scatter(mag_auto[j], mag_auto_aper[j], c='k', edgecolors='face')
    # ax.plot([10.0,23.0], [-0.38, -0.38], 'r-')
    ax.plot([10.0,23.0], [-0.40, -0.40], 'b-')
    ax.set(title='Randomly selected Ch2 stamps', xlabel='mag_auto (Vega)', ylabel='mag_auto - mag_aper(4",uncorrected)',
           xlim=[10.0, 14.0], ylim=[-0.45,0.0])
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig('Data/Comp_Sim/SPTpol/Plots/I2_mag_auto_aper_scatter_fwhm{fwhm}_gauss_zoom.pdf'.format(fwhm=fwhm), format='pdf')
    # plt.show()

    # Corrected aperture photometry
    # fig, ax = plt.subplots()
    # ax.grid()
    # for j in range(len(bins)-1):
    #     ax.scatter(mag_auto[j], mag_auto_aper_corr[j], c='k', edgecolors='face')
    # ax.plot([10.0,23.0], [-0.2, -0.2], 'r-')
    # ax.plot([10.0,23.0], [0.2, 0.2], 'r-')
    # ax.set(title='Randomly selected Ch2 stamps', xlabel='mag_auto (Vega)', ylabel='mag_auto - mag_aper(4",corrected)')
    # fig.savefig('Data/Comp_Sim/Plots/I2_mag_auto_aper_corr_scatter_fwhm'+str(fwhm)+'_gauss.pdf', format='pdf')
    # # plt.show()

    # Give a printout of the aproximate correction needed
    mag_auto_arr = np.concatenate([mag_auto[j].data for j in range(len(bins) - 1)])
    mag_auto_aper_arr = np.concatenate([mag_auto_aper[j].data for j in range(len(bins) - 1)])
    mean_corr = np.mean(mag_auto_aper_arr[np.logical_or(mag_auto_arr >= 10.0, mag_auto_arr <= 11.0)])
    print('Aperture correction needed: {:.2f}'.format(mean_corr))
