"""
SPT_ch2_5sigma_mags.py
Author: Benjamin Floyd

Tiny script to generate the 5 sigma limits on the SPT clusters.
"""

from __future__ import print_function, division
from astropy.io import ascii,fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from Pipeline_functions import file_pairing, catalog_image_match

# Pair the files together
print('Pairing Files.')
cluster_list = file_pairing('Data/Catalogs/', 'Data/Images/')

# Read in the Bleem catalog
Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))

# Match the clusters to the catalog requiring the cluster centers be within 1 arcminute of the Bleem center.
print('Matching catalogs.')
matched_list = catalog_image_match(cluster_list, Bleem, cat_ra_col='RA', cat_dec_col='DEC')
matched_list = [matched_list[l] for l in range(len(matched_list)) if matched_list[l][6] <= 60.0]

# Channel 1 science images and coverage maps
ch1_images = [[matched_list[j][1], matched_list[j][2]] for j in range(len(matched_list))]

program_flag = []

for i in range(len(ch1_images)):
    # Add a flag based on which observation cycle the images were taken.
    if fits.getval(ch1_images[i][0], 'PROGID') == 60099:
        program_flag.append(6)
    elif fits.getval(ch1_images[i][0], 'PROGID') == 70053:
        program_flag.append(7)
    elif fits.getval(ch1_images[i][0], 'PROGID') == 80012:
        program_flag.append(8)
    elif fits.getval(ch1_images[i][0], 'PROGID') == 10101:
        program_flag.append(10)
    elif fits.getval(ch1_images[i][0], 'PROGID') == 11096:
        program_flag.append(11)

# Read in the sky error catalog
error_cat = ascii.read('Data/sky_errors/SPT_sky_errors.cat')

error_cat['Program_Flag'] = program_flag

# Flux conversion from uJy to MJy/sr
flux_conv = 1.0/(23.5045 * fits.getval(ch1_images[0][0], 'PXSCAL2')**2)
# 5 sigma mags
# error_cat['Ch2_5sigma_Mag'] = -2.5 * np.log10(error_cat['Ch2_Sky_Error'] * 5.0 * flux_conv) + 17.538
#
# ascii.write(error_cat, 'Data/sky_errors/SPT_sky_errors.cat', overwrite=True)
#
# # Split the catalog based on exposure times
# prog_6_cat = error_cat[np.where(error_cat['Program_Flag'] == 6)]
# prog_7_cat = error_cat[np.where(error_cat['Program_Flag'] == 7)]
# prog_8_cat = error_cat[np.where(error_cat['Program_Flag'] == 8)]
# prog_10_cat = error_cat[np.where(error_cat['Program_Flag'] == 10)]
# prog_11_cat = error_cat[np.where(error_cat['Program_Flag'] == 11)]
#
# prog_cats = [prog_6_cat, prog_7_cat, prog_8_cat, prog_10_cat, prog_11_cat]

# print('Mean Channel 1 sky error for cycles 6, 7, and 8: {0:.3f} uJy'.format(np.mean(prog_678_cat['Ch1_Sky_Error'])))
# print('Mean Channel 2 sky error for cycles 6, 7, and 8: {0:.3f} uJy'.format(np.mean(prog_678_cat['Ch2_Sky_Error'])))
# print('Mean Channel 2 5-sigma magnitude for cycles 6, 7, and 8: {0:.2f} selection_band'.format(np.mean(prog_678_cat['Ch2_5sigma_Mag'])))
# print('Mean Channel 1 sky error for cycles 10 and 11: {0:.3f} uJy'.format(np.mean(prog_1011_cat['Ch1_Sky_Error'])))
# print('Mean Channel 2 sky error for cycles 10 and 11: {0:.3f} uJy'.format(np.mean(prog_1011_cat['Ch2_Sky_Error'])))
# print('Mean Channel 2 5-sigma magnitude for cycles 10 and 11: {0:.2f} selection_band'.format(np.mean(prog_1011_cat['Ch2_5sigma_Mag'])))

# Make plots of scatter
# for i in range(len(prog_cats)):
#     fig, ax = plt.subplots()
#     ax.hist(prog_cats[i]['Ch1_Sky_Error'], bins=len(prog_cats[i]), color='lightblue')
#     ax.set(title='SPT Ch1 Sky Errors for Cycle {0}'.format(prog_cats[i]['Program_Flag'][0]), xlabel='$\mu$Jy')
#     fig.savefig('Data/sky_errors/plots/SPT_Ch1_Sky_Error_Cycle_{0}.pdf'.format(prog_cats[i]['Program_Flag'][0]), format='pdf')
#
#     fig, ax = plt.subplots()
#     ax.hist(prog_cats[i]['Ch2_Sky_Error'], bins=len(prog_cats[i]), color='lightblue')
#     ax.set(title='SPT Ch2 Sky Errors for Cycle {0}'.format(prog_cats[i]['Program_Flag'][0]), xlabel='$\mu$Jy')
#     fig.savefig('Data/sky_errors/plots/SPT_Ch2_Sky_Error_Cycle_{0}.pdf'.format(prog_cats[i]['Program_Flag'][0]), format='pdf')

# Convert all flux errors into magnitude errors
error_cat['Ch1_Mag_Error'] = -2.5 * np.log10(error_cat['Ch1_Sky_Error'] * flux_conv) + 17.997
error_cat['Ch2_Mag_Error'] = -2.5 * np.log10(error_cat['Ch2_Sky_Error'] * flux_conv) + 17.538

# Plots for all cycles combined in magnitude units
fig, ax = plt.subplots()
ax.hist(error_cat['Ch1_Mag_Error'], bins=len(error_cat), color='lightblue')
ax.set(title='SPT Ch1 Sky Errors for All Cycles', xlabel='Vega Magnitude')
fig.savefig('Data/sky_errors/plots/SPT_Ch1_Sky_Error_All_Cycles.pdf', format='pdf')

fig, ax = plt.subplots()
ax.hist(error_cat['Ch2_Mag_Error'], bins=len(error_cat), color='lightblue')
ax.set(title='SPT Ch2 Sky Errors for All Cycles', xlabel='Vega Magnitude')
fig.savefig('Data/sky_errors/plots/SPT_Ch2_Sky_Error_All_Cycles.pdf', format='pdf')