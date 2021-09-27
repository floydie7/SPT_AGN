"""
SPT_Sky_Errors.py
author: Benjamin Floyd

This script will process sky errors for all the SPT clusters in the Bleem 2015 survey.
"""

from __future__ import print_function, division

from time import time

from Pipeline_functions import file_pairing, catalog_image_match
from Sky_Errors.Sky_Error_Functions import *

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
matched_list = [matched_list[l] for l in range(len(matched_list)) if matched_list[l][6] <= 60.0]
end_match_time = time()
print('Matching completed.\nImages in directory: {0}\tClusters matched: {1}\tTime Spent Matching: {2:.1f} s\n'
      .format(len(cluster_list), len(matched_list), end_match_time - start_time))

# Channel 1 science images and coverage maps
ch1_images = [[matched_list[j][1], matched_list[j][2]] for j in range(len(matched_list))]

# Channel 2 science images and coverage maps
ch2_images = [[matched_list[j][3], matched_list[j][4]] for j in range(len(matched_list))]

# Containers to store the sky errors in.
ch1_sky_errors = []
ch2_sky_errors = []

# Flag for programs 6,7,8 or 10,11
program_flag = []

for channel in range(2):

    if channel == 0:
        image_list = ch1_images
        zpt_mag = 17.997
        aper_corr = 10**(-(-0.38)/2.5)
        sky_error_list = ch1_sky_errors
    else:
        print('Finished Ch1 errors, now processing Ch2 errors.')
        image_list = ch2_images
        zpt_mag = 17.538
        aper_corr = 10**(-(-0.40)/2.5)
        sky_error_list = ch2_sky_errors

    # Iterate though the images to find the sky errors
    for i in range(len(image_list)):

        # Define the image and coverage paths
        image = image_list[i][0]
        coverage = image_list[i][1]

        print('Processing image: {0}'.format(image))

        # Define .coo and .selection_band paths
        image_coo = 'Data/sky_errors/coordinates/'+image[-38:-19]+'.coo'
        image_mag = 'Data/sky_errors/qphot_mag_cats/'+image[-38:-19]+'.selection_band'

        # Convert the channel 1 4" aperture to pixels
        aper_pix = 4 / fits.getval(image, 'PXSCAL2')
        print('Using an aperture diameter of {0} pixels'.format(aper_pix))

        # Generate the aperture coordinate files.
        generate_apertures(image, output_coo=image_coo, aper_size=aper_pix,
                           xmin=2*aper_pix, xmax=fits.getval(image, 'NAXIS1')-2*aper_pix,
                           ymin=2*aper_pix, ymax=fits.getval(image, 'NAXIS2')-2*aper_pix)

        # Run qphot on the apertures.
        run_qphot(image, coord_list=image_coo, output_file=image_mag, aper_size=aper_pix, zpt_mag=zpt_mag)

        # Clean up the qphot catalog and output as an Astropy table
        flux_catalog = catalog_management(image_mag, coverage, min_cov=4.0)

        # Preform the curve fitting on the data and output the bin midpoints, the histogram values, and the best fit
        # values.
        mid_pts, hist, opt_params = fit_gaussian(flux_catalog, nbins=60, hist_range=(-0.2, 0.4), cutoff=0.02)

        # The best-fit flux error parameter
        bf_sigma = opt_params[1]

        # Convert the native image units of MJy/sr to uJy.
        flux_conv = 23.5045 * fits.getval(image, 'PXSCAL2')**2  # 23.5045 (uJy/arcsec^2)/(MJy/sr) * (arcsec/pix)^2

        # Find the corrected flux in uJy.
        sky_error = np.abs(flux_error(bf_sigma, aper_corr=1., flux_conv=flux_conv))

        # Store the sky errors with the appropriate cluster name.
        sky_error_list.append(sky_error)

        # Make the plot.
        make_plot(mid_pts, hist, opt_params, sky_error, image[-38:-19], cutoff=0.02, flux_conv=flux_conv)

# for i in range(len(ch1_images)):
#     # Add a flag if the image was collected in programs 6, 7, or 8. 1 if true, 0 if false (programs 10 or 11)
#     if fits.getval(ch1_images[i][0], 'PROGID') in [60099, 70053, 80012]:
#         program_flag.append(1)
#     else:
#         program_flag.append(0)
# Write the sky error catalog to disk.
cluster_ids = [ch1_images[i][0][-35:-19] for i in range(len(ch1_images))]
sky_error_cat = Table([cluster_ids, ch1_sky_errors, ch2_sky_errors, program_flag],
                      names=['SPT_ID', 'Ch1_Sky_Error', 'Ch2_Sky_Error'])
# sky_error_cat['Ch1_Sky_Error'].unit, sky_error_cat['Ch2_Sky_Error'].unit = ['uJy', 'uJy']
ascii.write(sky_error_cat, 'Data/sky_errors/SPT_sky_errors.cat', overwrite=True)

end_time = time()

print('Total runtime: {0} s'.format(end_time - start_time))
