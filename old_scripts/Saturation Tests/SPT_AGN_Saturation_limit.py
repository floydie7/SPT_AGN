"""
SPT_AGN_Saturation_limit.py
Author: Benjamin Floyd

This script will run SExtractor on all the cluster images and produce a plot of FWHM vs. MAG_APER (4") to determine the
saturation magnitude of objects in our sample.
"""

from __future__ import print_function, division

import os

import matplotlib
import numpy as np
from SExtractor import run_sex
from astropy.io import ascii, fits
from astropy.table import Table, vstack

from Pipeline_functions import file_pairing, catalog_image_match

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# Read in the Bleem catalog and remove unconfirmed candidates.
Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))
Bleem = Bleem[np.where(Bleem['M500'] != 0.0)]

# Collate the files and match against Bleem.
cluster_list = file_pairing('Data/Catalogs/', 'Data/Images/')
cluster_list = catalog_image_match(cluster_list, Bleem, cat_ra_col='RA', cat_dec_col='DEC', max_sep=1.0)

# Group the clusters by their observation cycle.
program_flag = []
for cluster in cluster_list:
    if fits.getval(cluster['ch1_sci_path'], 'PROGID') == 60099:
        program_flag.append(6)
    elif fits.getval(cluster['ch1_sci_path'], 'PROGID') == 70053:
        program_flag.append(7)
    elif fits.getval(cluster['ch1_sci_path'], 'PROGID') == 80012:
        program_flag.append(8)
    elif fits.getval(cluster['ch1_sci_path'], 'PROGID') == 10101:
        program_flag.append(10)
    elif fits.getval(cluster['ch1_sci_path'], 'PROGID') == 11096:
        program_flag.append(11)


# Cycles 6--8 had Channel 1 exposure times of 8x100s.
cycles_678 = [cluster_list[i] for i in range(len(cluster_list)) if program_flag[i] in [6, 7, 8]]

# Cycles 10 & 11 had Channel 2 exposure times of 12x30s.
cycles_1011 = [cluster_list[i] for i in range(len(cluster_list)) if program_flag[i] in [10, 11]]

# for group in range(2):
for group in [0]:
    if group == 0:
        cluster_group = cycles_678
        image = 'ch1_sci_path'
        psf_fwhm = 1.95
        zpt_mag = 17.997
        aper_corr = -0.38
    else:
        cluster_group = cluster_list
        image = 'ch2_sci_path'
        psf_fwhm = 2.02
        zpt_mag = 17.538
        aper_corr = -0.4

    # Find the sources and preform the photometry.
    tables = []
    for cluster in cluster_group:

        # print(cluster[image])

        # Get the SExtractor file paths
        sex_config = 'Data/Saturation/sex_configs/default.sex'
        sex_param = 'Data/Saturation/sex_configs/default.param'
        sex_in_image = '{det_img},{meas_img}'.format(det_img=os.path.abspath(cluster['ch2_sci_path']),
                                                     meas_img=os.path.abspath(cluster[image]))
        sex_out_cat = 'Data/Saturation/sex_catalogs/{img_name}.cat'.format(img_name=cluster[image][-38:-19])

        # Run SExtractor in dual-image mode. Using the single-image method for now because I only need a single band
        # catalog. Eventually I'll fix the real dual-image method.
        run_sex(sex_in_image, sex_out_cat, mag_zero=zpt_mag, seeing_fwhm=psf_fwhm, sex_config=sex_config,
                param_file=sex_param)

        # Read in the catalog and store it in the tables list
        catalog = ascii.read(sex_out_cat)
        catalog['SPT_ID'] = cluster['SPT_ID']
        catalog = catalog[np.where(catalog['FLAGS'] < 4)]
        catalog['MAG_APER'] += aper_corr
        catalog = catalog[np.where(catalog['MAG_APER'] < 17.0)]
        tables.append(catalog)

    # For each group, stack all the tables
    stacked_catalog = vstack(tables)

    # Make plots
    # fig, ax = plt.subplots()
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.scatter(stacked_catalog['MAG_APER'], stacked_catalog['FWHM_WORLD']*3600, marker='.', alpha=0.2)
    # if group == 0:
    #     ax.plot([7, 25], [1.95, 1.95], 'r-', alpha=0.6)
    # else:
    #     ax.plot([7, 25], [2.02, 2.02], 'r-', alpha=0.6)
    # ax.annotate('PSF FWHM: {fwhm}"'.format(fwhm=psf_fwhm), xy=(7.5, psf_fwhm+0.05), color='r', alpha=0.6)
    # ax.set(title='SPT Cluster Objects Channel {ch}, Cycles {cycle}'
    #        .format(ch=group+1, cycle='6-8' if group == 0 else '6-8, 10, 11'),
    #        xlabel='Vega Aperture Magnitude (4", corrected)', ylabel='FWHM (arcsec)', xlim=[7, 16], ylim=[0, 5])
    # fig.savefig('Data/Saturation/Plots/SPT_Cycles{cycle}_Saturation.pdf'
    #             .format(cycle='678' if group == 0 else '6781011'), format='pdf')

    if group == 0:
        stars = stacked_catalog[np.where(stacked_catalog['FWHM_WORLD']*3600. >= 2.5) and
                                np.where(stacked_catalog['FWHM_WORLD']*3600. <= 4.0)]
        stars = stars[np.where(stars['MAG_APER'] <= 12.5)]
        stars['FWHM_WORLD'] = stars['FWHM_WORLD'] * 3600.
        stars.sort('MAG_APER')
        stars.pprint(max_width=-1, max_lines=-1)
        stars.sort('FWHM_WORLD')
        stars.reverse()
        stars.pprint(max_width=-1, max_lines=-1)
