"""
Saturation_imexam.py
Author: Benjamin Floyd

This script uses the python packages Photutils and imexam to indentify stars in the images then generate the 1D Gaussian
profile plots to determine the saturation limit.
"""

from __future__ import print_function, division

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from Pipeline_functions import file_pairing, catalog_image_match
from imexam.imexamine import Imexamine

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

for group in range(2):
    if group == 0:
        cluster_group = cycles_678
        image = 'ch1_sci_path'
        psf_fwhm = 1.95
        zpt_mag = 17.997
    else:
        cluster_group = cycles_1011
        image = 'ch2_sci_path'
        psf_fwhm = 2.02
        zpt_mag = 17.538

    # Find the sources and preform the photometry.
    tables = []
    for cluster in cluster_group:
        print(cluster[image])
        # Load in the data
        data = fits.getdata(cluster[image])

        # Get the pixel scale
        pix_scale = fits.getval(cluster[image], 'PXSCAL2')

        # Collect the sigma-clipped statistics
        mean, median, stddev = sigma_clipped_stats(data, sigma=3.0, iters=5)

        # Use the source finder to identify the stars
        daofinder = DAOStarFinder(fwhm=psf_fwhm*pix_scale, threshold=5.*stddev, exclude_border=True)
        sources = daofinder(data - median)  # Run the star finder on background subtracted data.
        sources = sources[np.where(sources['peak'] >= 10)]  # To match wcsTools' imstar functionality.
        sources.pprint(max_width=-1)

        if len(sources) != 0:
            # Collect the positions from the source catalog
            # positions = zip(sources['xcentroid'], sources['ycentroid'])
            # print(positions)
            # raise SystemExit

            # Load the image in imexam
            plots = Imexamine()

            # For each object create a column fit 1D Gaussian to the data
            for row in sources:
                print('{cluster}_{star}'.format(cluster=cluster['SPT_ID'],star=row['id']))
                plots.line_fit_pars['title'][0] = '{cluster}_({x:d}, {y:d})'\
                    .format(cluster=cluster['SPT_ID'], x=int(row['xcentroid']), y=int(row['ycentroid']))
                plots.line_fit_pars['rplot'][0] = 8.
                # plots.line_fit_pars['background'][0] = True
                plots.line_fit(row['xcentroid'], row['ycentroid'], data=data, form='Gaussian1D')
                plots.save('Data/Saturation/Plots/imexam_plots/{cluster}_{star}.pdf'.format(cluster=cluster['SPT_ID'],
                                                                                            star=row['id']))
                plots.close()
                raise SystemExit
