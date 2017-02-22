"""
AGN_Surf_Den.py
Author: Benjamin Floyd
This script is designed to automate the process of calculating the AGN surface density for the SPT clusters.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, system, chmod
from astropy.io import fits, ascii
from astropy.wcs import WCS  # For converting RA/DEC coords in catalog to pixel coords for coverage map.
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy import units as u
from tools import area, m500Tor500
from time import time


def Object_Selection(catalog, coverage, mincov, aperdiam, cat_ra='ra', cat_dec='dec'):
    """
    A rewriting and improvement of Bandon Decker's coverageCut method. This method will select objects from the
    specified catalog dependent on the object having enough "good" pixels surrounding it above an inputted threshold.

    :param catalog: Astropy table object.
        This is the coordinate catalog from which objects will be selected.
    :param cat_ra: String.
        Catalog RA column name label. Defaults to 'ra'.
    :param cat_dec: String.
        Catalog Dec column name label. Defaults to 'dec'.
    :param coverage: Fits image file name.
        Coverage map describing the number of exposures for each pixel.
    :param mincov: Integer.
        Specifies the minimum number of exposures allowed for a pixel to be allowed in the cut.
    :param aperdiam: Float.
        The aperture diameter in arcseconds centered on the catalog object's coordinate within which we will check the
        values of the pixels in the coverage map.
    :param areathresh: Float.
        Specifies the minimum ratio of pixels within the aperture required for the object to be included in the cut.
    :return selection_cat: Astropy table object. The final catalog of selected objects.
    """

    # Read in the WCS from the coverage map
    w = WCS(coverage)

    # Load in the data from the coverage map
    cov = fits.getdata(coverage)
    selection_cat = []
    for obj in catalog:
        Xpix, Ypix = w.wcs_world2pix(obj[cat_ra], obj[cat_dec], 1)
        if Xpix**2 + Ypix**2 <= aperdiam**2:
            cov_value = cov[int(round(Ypix))][int(round(Xpix))]
            if cov_value >= mincov:
                if len(selection_cat) == 0:
                    selection_cat = Table(obj)
                else:
                    selection_cat.add_row(obj)
    return selection_cat
    # TODO Add the AGN selection here (objects in catalog above Ch1 - Ch2 color).
    # TODO Change output to be table of number of AGN for each cluster within three radius bins + total radius.


def file_pairing(cat_dir, img_dir):
    """
    Given directories of catalogs and images this method will pair the filenames corresponding to a cluster based
    on the coordinate listed in the filenames and then return an array containing the paths to the files.

    :param cat_dir: String.
        Path to directory with catalogs.
    :param img_dir: String.
        Path to directory with image files.
    :return clusters: Array.
        An array consisting of lists containing the cluster's filenames.
        0: SExtractor catalog path name.
        1: IRAC Ch1 science image path name.
        2: IRAC Ch1 coverage map path name.
        3: IRAC Ch2 science map path name.
        4: IRAC Ch2 coverage map path name.
    """

    # Generate lists of all the catalog files and all the image files.
    cat_files = [f for f in listdir(cat_dir) if not f.startswith('.')]
    image_files = [f for f in listdir(img_dir) if not f.startswith('.')]

    # Pair the catalogs with their corresponding images.
    clusts = [[] for j in range(len(cat_files))]
    for i in range(len(cat_files)):
        clusts[i].append(cat_dir + cat_files[i])
        cat_coord = cat_files[i][7:16]
        for image in image_files:
            img_coord = image[10:19]
            if cat_coord == img_coord:
                clusts[i].append(img_dir + image)

    return [x for x in clusts if len(x) == 5]


def catalog_image_match(catalog, cluster_list, cat_ra_col='ra', cat_dec_col='dec'):
    """
    Matches the center coordinate of a fits image to a catalog of objects then returns an array containing the paths to
    the files that have a match in the catalog.
    :param catalog: Astropy table object.
        The catalog we wish to check our images against.
    :param cluster_list: Array.
        An array of lists containing the paths to the clusters' files.
    :param cat_ra_col: String.
        RA column name in catalog. Defaults to 'ra'.
    :param cat_dec_col: String.
        Dec column name in catalog. Defaults to 'dec'.
    :return matched_clusters: Array.
        An array in the same format as the input cluster_list for the matched clusters with the addition of the catalog
        id value for the cluster and the separation between the image coordinate and the catalog coordinate.
        0: SExtractor catalog path name.
        1: IRAC Ch1 science image path name.
        2: IRAC Ch1 coverage map path name.
        3: IRAC Ch2 science map path name.
        4: IRAC Ch2 coverage map path name.
        5: Index in catalog corresponding to the match.
        6: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
    """

    # Create astropy skycoord object from the catalog columns.
    cat_coords = SkyCoord(catalog[cat_ra_col], catalog[cat_dec_col], unit=u.degree)

    # Initialize the output array.
    matched_clusters = []

    for i in range(len(cluster_list)):
        # Get the RA and Dec of the center pixel in the image.
        img_ra = fits.getval(cluster_list[i][1], 'CRVAL1')
        img_dec = fits.getval(cluster_list[i][1], 'CRVAL2')

        # Create astropy skycoord object for the center pixel of the image.
        img_coord = SkyCoord(img_ra, img_dec, unit=u.degree)

        # Preform the catalog matching.
        idx, d2d, d3d = img_coord.match_to_catalog_sky(cat_coords)

        # Add the (nearest) catalog id and separation (in arcsec) to the output array.
        cluster_list[i].extend([idx, d2d.arcsec.item()])
        matched_clusters.append(cluster_list[i])

    return matched_clusters


def mask_flag(cluster_list, mask_file):
    """
    Appends the masking flag to the catalog list array.
    :param cluster_list: Array.
        An array of lists containing the paths to the clusters' files.
    :param mask_file:
    :return matched_clusters: Array.
        An array in the same format as the input cluster_list for the matched clusters with the addition of the catalog
        id value for the cluster and the separation between the image coordinate and the catalog coordinate.
        0: SExtractor catalog path name.
        1: IRAC Ch1 science image path name.
        2: IRAC Ch1 coverage map path name.
        3: IRAC Ch2 science map path name.
        4: IRAC Ch2 coverage map path name.
        5: Catalog ID.
        6: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
        7: Masking Flag.
            0: No additional masking required, 1: Object masking needed, have regions file, 2: Further attention needed.
    """

    # Open the masking notes file
    with open(mask_file) as mask:
        mask_lines = mask.readlines()
        # Pair the clusters in the list with those in the masking notes file.
        for i in range(len(cluster_list)):
            for j in range(len(mask_lines)):
                mask_notes = mask_lines[j].strip("\n").split("\t")
                # if cluster_list[i][0] == mask_notes[0]:
                if cluster_list[i][0] == 'Data/test/' + mask_notes[0][14:]:
                    cluster_list[i].append(int(mask_notes[1]))

    return cluster_list


def catalog_match(cluster_list, master_catalog, catalog_cols, cat_ra_col='ra', cat_dec_col='dec'):
    """
    Pairs the SExtractor catalogs with the master catalog.
    :param cluster_list: Array.
        An array of lists containing the paths to the clusters' files.
    :param master_catalog: Astropy table object.
        Catalog containing information about the cluster as a whole.
    :param catalog_cols: List of strings.
        List of column names in master catalog that we wish to incorporate into the cluster-specific catalogs.
    :param cat_ra_col: String.
        RA column name in catalog. Defaults to 'ra'.
    :param cat_dec_col: String.
        Dec column name in catalog. Defaults to 'dec'.
    :return cluster_list: Array.
        An array in the same format as the input cluster_list for the matched clusters with the addition of the catalog
        id value for the cluster and the separation between the image coordinate and the catalog coordinate.
        0: SExtractor catalog path name.
        1: IRAC Ch1 science image path name.
        2: IRAC Ch1 coverage map path name.
        3: IRAC Ch2 science map path name.
        4: IRAC Ch2 coverage map path name.
        5: Catalog ID.
        6: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
        7: Masking Flag.
            0: No additional masking required, 1: Object masking needed, have regions file, 2: Further attention needed.
        8: SExtractor catalog with added columns from the master catalog and radial distances for all objects.
    """

    # List to hold our catalogs
    catalogs = []

    for i in range(len(cluster_list)):
        # Read in all the catalogs.
        catalogs.append(ascii.read(cluster_list[i][0]))

        # We already matched our SExtractor catalogs to the master catalog so we only need to pull the correct row.
        # The master catalog index is stored in cluster_list[i][5].
        # Create astropy skycoord object from the catalog columns.
        cat_coords = SkyCoord(master_catalog[cat_ra_col][cluster_list[i][5]],
                              master_catalog[cat_dec_col][cluster_list[i][5]], unit=u.degree)

        # List to hold the separations
        separation = []

        # For all objects in catalog find the angular separation between the object's coordinate and the
        # cluster-centered coordinate.
        # Note: Astropy .separation() calculates the great-circle distance via the Vincenty formula *not* the small
        # angle approximation.
        for j in range(len(catalogs[i])):
            sexcat_coords = SkyCoord(catalogs[i]['ALPHA_J2000'][j], catalogs[i]['DELTA_J2000'][j], unit=u.degree)
            separation.append(sexcat_coords.separation(cat_coords).arcsec)

        # For all requested columns from the master catalog add the value to all columns in the SExtractor catalog.
        for col_name in catalog_cols:
            catalogs[i][col_name] = master_catalog[col_name][cluster_list[i][5]]

        # Store all the separations in as a column in the catalog.
        # catalogs[i]['rad_dist'] = separation
        catalogs[i].add_column(Column(np.array(separation), name='rad_dist'))

        # Append the catalog to the cluster_list array.
        cluster_list.extend([catalogs[i]])

    return cluster_list


def visualizer(cluster_list):
    """
    Creates a script to view all cluster images in ds9.
    :param cluster_list: Array.
        An array of lists containing the paths to the clusters' files.
    :return:
    """
    script = open('ds9viz', mode='w')
    script.write('#!/bin/tcsh\n')
    script.write('ds9 -single ')
    for i in range(len(cluster_list)):
        script.write(cluster_list[i][1])
        script.write(' ')
        script.write(cluster_list[i][3])
        script.write(' ')
    script.close()

    chmod('ds9viz', 0o755)
    system('./ds9viz')


def coverage_mask(cluster_list, Ch1_min_cov, Ch2_min_cov):
    """
    Generates a pixel mask of good pixels where both IRAC Ch1 and Ch2 coverage values are above a specified value.
    :param cluster_list: Array.
        An array of lists containing the paths to the clusters' files.
    :param min_cov: Float.
        Minimum coverage value allowed.
    :return cov_cat_flag_list: Array.
        A restructured version of the cluster_list array.
        0: Coverage good/bad pixel map path name.
        1: Modified SExtractor catalog generated by cluster_match method.
        2: Masking Flag.
            0: No additional masking required, 1: Object masking needed, have regions file, 2: Further attention needed.
    """

    # Generate a new output list
    cov_cat_flag_list = []

    for i in range(len(cluster_list)):

        # Read in the two coverage maps, also grabbing the header from the Ch1 map.
        IRAC_Ch1_cover, head = fits.getdata(cluster_list[i][2], header=True)
        IRAC_Ch2_cover = fits.getdata(cluster_list[i][4])

        # Initialize the mask.
        combined_cov = np.zeros((fits.getval(cluster_list[i][2], 'NAXIS2'), fits.getval(cluster_list[i][2], 'NAXIS1')))

        # Create the mask by setting pixel value to 1 if the pixel has coverage above the minimum coverage value in
        # both IRAC bands.
        for j in range(len(IRAC_Ch1_cover)):
            for k in range(len(IRAC_Ch1_cover[j])):
                if (IRAC_Ch1_cover[j][k] >= Ch1_min_cov) and (IRAC_Ch2_cover[j][k] >= Ch2_min_cov):
                    combined_cov[j][k] = 1

        # Write out the coverage mask.
        combined_cov_hdu = fits.PrimaryHDU(combined_cov, header=head)
        combined_cov_hdu.writeto('Data/Masks/' + cluster_list[i][0][14:30] + '_cov_mask'
                                 + str(Ch1_min_cov) + '_' + str(Ch2_min_cov) + '.fits', clobber=True)

        # Append the new coverage mask path name and both the catalog and the masking flagfrom cluster_list
        # to the new output list.
        cov_cat_flag_list.extend(['Data/Masks/' + cluster_list[i][0][14:30] + '_cov_mask'
                                  + str(Ch1_min_cov) + '_' + str(Ch2_min_cov) + '.fits',
                                  cluster_list[i][8], cluster_list[i][7]])

    return cov_cat_flag_list


def object_mask(cov_cat_flag_list, reg_file_dir):
    """
    For the clusters that have objects that require additional masking apply the masks based on the previously generated
    ds9 regions files.
    :param cov_cat_flag_list: Array.
        An array of lists containing the coverage mask, the cluster catalog, and the masking flag.
    :param reg_file_dir: String.
        Path to directory containing the ds9 regions files.
    :return cov_cat_list: Array.
        An array with just the final good/bad pixel map and the cluster catalog.
        0: Coverage (Masked) pixel map path name.
        1: Cluster SExtractor catalog (with Bleem columns).
    """

    # Output Array
    cov_cat_list = []
    for i in range(len(cov_cat_flag_list)):

        # If the masking flag == 0 then no additional masking is required. Append to output list
        if cov_cat_flag_list[i][2] == 0:
            cov_cat_list.append(cov_cat_flag_list[i])

        # Otherwise go get the appropriate regions file and append the mask to the coverage pixel map.
        #else:

    return cov_cat_list



def surface_density(cluster_list):
    """
    Computes the AGN surface density of a cluster.
    :param cluster_list: Array.
        An array consisting of lists containing the cluster's filenames.
    :return:
    """

    for i in range(len(cluster_list)):
        # Calculate the area (in deg^2) of the combined coverage mask.
        I1I2_area = area(cluster_list[i][7])/(60.0 ** 2)
        # TODO Make sure the index corresponds to the right filename.
        # TODO "area" needs to be rewritten to specify radius.

        # Read in the catalog from the cluster list.
        catalog = ascii.read(cluster_list[i][1], format='SExtractor')
        # TODO The catalog is already in memory.

        # Generate a catalog of objects subject to the coverage mask.
        cov_cat = Object_Selection(catalog, cluster_list[i][7], mincov=mincov, aperdiam=r500)

        surf_den = float(len(cov_cat))/I1I2_area

    return surf_den


def final_catalogs(cluster_list, catalog_cols):
    """
    Writes the final catalogs to disk.
    :param cluster_list:
    :param catalog_cols:
    :return:
    """

    for i in range(len(cluster_list)):
        final_cat = cluster_list[i][1][catalog_cols]
        # ascii.write(final_cat, 'Data/Output/' + cluster_list[i][0][])


# Run the pipeline.
start_time = time()
print("Beginning Pipeline")
clusters = file_pairing('Data/test/', 'Data/Images/')
print("File pairing complete, Clusters in directory: ", len(clusters))

Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))

print("Matching Images against Bleem Catalog.")
matched_list = catalog_image_match(Bleem, clusters, cat_ra_col='RA', cat_dec_col='DEC')

# fig, ax = plt.subplots()
# ax.hist([matched_list[i][6] for i in range(len(matched_list))], bins=1e4)
# ax.set(title='Separation between Bleem and center pixel', xlabel='separation (arcsec)')
# ax.set_xlim([0,120])
# plt.show()

matched_list = [matched_list[i] for i in range(len(matched_list)) if matched_list[i][6] <= 60.0]
print("Matched clusters (within 1 arcmin): ", len(matched_list))

print("Applying mask flags.")
cluster_matched_flagged = mask_flag(matched_list, 'Data/mask_notes.txt')
# manual_mask = [cluster_matched_flagged[i] for i in range(len(cluster_matched_flagged))
#                if cluster_matched_flagged[i][7] == 2]
# print("Clusters needing manual attention: ", len(manual_mask))

print("Matching catalogs.")
match_time_start = time()
cat_matched_list = catalog_match(cluster_matched_flagged, Bleem, ['REDSHIFT', 'REDSHIFT_UNC', 'M500', 'DM500'],
                                 cat_ra_col='RA', cat_dec_col='DEC')
match_time_end = time()
print("Time taken calculating separtations: ", match_time_end - match_time_start, " s")
print(cat_matched_list[0])
raise SystemExit

print("Generating coverage level masks.")
cov_list = coverage_mask(cat_matched_list, Ch1_min_cov=5, Ch2_min_cov=4)

end_time = time()
print("Total runtime: ", end_time - start_time, " s.")
