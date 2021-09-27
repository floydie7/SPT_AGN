"""
SPT_AGN_Saturation_limit.py
Author: Benjamin Floyd

This script will find the stars in the cluster images and preform aperture photometry on them using Photutils library.
The top ten brightest stars per exposure program will be reported.
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, DAOStarFinder
from Pipeline_functions import file_pairing, catalog_image_match

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

        if len(sources) != 0:
            # Collect the positions from the source catalog
            positions = zip(sources['xcentroid'], sources['ycentroid'])

            # Create the apertures and sky annuli
            apertures = CircularAperture(positions, r=4.*pix_scale)
            sky_annuli = CircularAnnulus(positions, r_in=10., r_out=13.)
            apers = [apertures, sky_annuli]

            # Preform the photometery
            phot_table = aperture_photometry(data, apers)

            # Background subtract the flux
            bkg_mean = phot_table['aperture_sum_1'] / sky_annuli.area()
            bkg_sum = bkg_mean * apertures.area()
            phot_table['residual_aperture_sum'] = phot_table['aperture_sum_0'] - bkg_sum

            # Convert the flux into a magnitude.
            phot_table['selection_band'] = -2.5 * np.log10(phot_table['residual_aperture_sum']) + zpt_mag

            # Store the photometric table in the list for later.
            tables.append(phot_table)

    # For each group stack all the catalogs.
    stacked_tables = vstack(tables)

    # Sort by magnitude (brightest to dimmest)
    stacked_tables.sort('selection_band')

    # Print the brightest ten stars.
    # stacked_tables[:len(stacked_tables)//2].pprint(max_width=-1)

    # Make plots
    fig, ax = plt.subplots()
    ax.hist(stacked_tables['selection_band'])
    ax.set(title='Channel {ch} Stars in SPT Clusters'.format(ch=group+1), xlabel='Vega Magnitudes')
    fig.savefig('Data/SPT_Ch{ch}_Stars.pdf'.format(ch=group+1), format='pdf')
