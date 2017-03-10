"""
AGN_Surf_Den.py
Author: Benjamin Floyd
This script is designed to automate the process of calculating the AGN surface density for the SPT clusters.
"""
from __future__ import print_function
import numpy as np
import warnings  # For suppressing the astropy warnings that pop up when reading headers.
from os import listdir, system, chmod
from astropy.io import fits, ascii
from astropy.wcs import WCS  # For converting RA/DEC coords in catalog to pixel coords for coverage map.
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning  # For suppressing the astropy warnings.
from itertools import islice
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

# Suppress Astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)


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
    clusters = [[] for _ in range(len(cat_files))]
    for i in range(len(cat_files)):
        clusters[i].append(cat_dir + cat_files[i])
        cat_coord = cat_files[i][7:16]
        for image in image_files:
            img_coord = image[10:19]
            if cat_coord == img_coord:
                clusters[i].append(img_dir + image)

    return [x for x in clusters if len(x) == 5]


def catalog_image_match(cluster_info, catalog, cat_ra_col='ra', cat_dec_col='dec'):
    """
    Matches the center coordinate of a fits image to a catalog of objects then returns an array containing the paths to
    the files that have a match in the catalog.
    :param cluster_info: Array.
        An array of lists containing the paths to the clusters' files.
    :param catalog: Astropy table object.
        The catalog we wish to check our images against.
    :param cat_ra_col: String.
        RA column name in catalog. Defaults to 'ra'.
    :param cat_dec_col: String.
        Dec column name in catalog. Defaults to 'dec'.
    :return cluster_info: Array.
        An array in the same format as the input cluster_info for the matched clusters with the addition of the catalog
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

    # Array element names
    irac_ch1_sci = cluster_info[1]

    # Get the RA and Dec of the center pixel in the image.
    img_ra = fits.getval(irac_ch1_sci, 'CRVAL1')
    img_dec = fits.getval(irac_ch1_sci, 'CRVAL2')

    # Create astropy skycoord object for the center pixel of the image.
    img_coord = SkyCoord(img_ra, img_dec, unit=u.degree)

    # Preform the catalog matching.
    idx, d2d, d3d = img_coord.match_to_catalog_sky(cat_coords)

    # Add the (nearest) catalog id and separation (in arcsec) to the output array.
    cluster_info.extend([idx, d2d.arcsec.item()])

    return cluster_info


def mask_flag(cluster_info, mask_file):
    """
    Appends the masking flag to the catalog list array.
    :param cluster_info: Array.
        An array of lists containing the paths to the clusters' files.
    :param mask_file:
    :return cluster_info: Array.
        An array in the same format as the input cluster_info for the matched clusters with the addition of the catalog
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

        # Array element names
        sex_cat_path = cluster_info[0]

        # Go through the masking notes file and match the flag value with the correct cluster.
        # The cluster catalog name will be stored in mask_notes[0], the flag will be in mask_notes[1], there will be
        # notes stored in mask_notes[2] but we'll ignore those since all clusters that need object masking will have
        # ds9 region files with the objects specified there.
        for j in range(len(mask_lines)):
            mask_notes = mask_lines[j].strip("\n").split("\t")
            if sex_cat_path.endswith(mask_notes[0][-23:]):
                cluster_info.append(int(mask_notes[1]))

    return cluster_info


def coverage_mask(cluster_info, ch1_min_cov, ch2_min_cov):
    """
    Generates a pixel mask of good pixels where both IRAC Ch1 and Ch2 coverage values are above a specified value.
    :param cluster_info: Array.
        An array of lists containing the paths to the clusters' files.
    :param ch1_min_cov: Float.
        Minimum coverage value allowed in IRAC Ch1.
    :param ch2_min_cov: Float.
        Minimum coverage value allowed in IRAC Ch2.
    :return cluster_info: Array.
        An array in the same format as the input cluster_info for the matched clusters with the addition of the catalog
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
        8: Coverage good/bad pixel map path name.
    """

    # Array element names
    sex_cat_path = cluster_info[0]
    irac_ch1_cov_path = cluster_info[2]
    irac_ch2_cov_path = cluster_info[4]

    # Read in the two coverage maps, also grabbing the header from the Ch1 map.
    irac_ch1_cover, head = fits.getdata(irac_ch1_cov_path, header=True, ignore_missing_end=True)
    irac_ch2_cover = fits.getdata(irac_ch2_cov_path, ignore_missing_end=True)

    # Initialize the mask.
    combined_cov = np.zeros((fits.getval(irac_ch1_cov_path, 'NAXIS2', ignore_missing_end=True),
                             fits.getval(irac_ch1_cov_path, 'NAXIS1', ignore_missing_end=True)))

    # Create the mask by setting pixel value to 1 if the pixel has coverage above the minimum coverage value in
    # both IRAC bands.
    for j in range(len(irac_ch1_cover)):
        for k in range(len(irac_ch1_cover[j])):
            if (irac_ch1_cover[j][k] >= ch1_min_cov) and (irac_ch2_cover[j][k] >= ch2_min_cov):
                combined_cov[j][k] = 1

    # Write out the coverage mask.
    mask_pathname = 'Data/Masks/' + sex_cat_path[-23:-7] + '_cov_mask' + str(ch1_min_cov) \
                    + '_' + str(ch2_min_cov) + '.fits'
    combined_cov_hdu = fits.PrimaryHDU(combined_cov, header=head)
    combined_cov_hdu.writeto(mask_pathname, overwrite=True)

    # Append the new coverage mask path name and both the catalog and the masking flag from cluster_info
    # to the new output list.
    cluster_info.append(mask_pathname)

    return cluster_info


def object_mask(cluster_info, reg_file_dir):
    """
    For the clusters that have objects that require additional masking apply the masks based on the previously generated
    ds9 regions files.
    :param cluster_info: Array.
        An array of lists containing the coverage mask, the cluster catalog, and the masking flag.
    :param reg_file_dir: String.
        Path to directory containing the ds9 regions files.
    :return cluster_info: Array.
        An array in the same format as the input cluster_info for the matched clusters with the addition of the catalog
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
        8: Coverage good/bad pixel map path name.
    """

    # Region file directory files
    reg_files = [f for f in listdir(reg_file_dir) if not f.startswith('.')]

    # Array element names
    sex_cat_path = cluster_info[0]
    flag = cluster_info[7]
    pixel_map_path = cluster_info[8]

    # If the masking flag == 0 then no additional masking is required.
    # Otherwise go get the appropriate regions file and append the mask to the coverage pixel map.
    if flag != 0:

        # Read in the WCS from the coverage mask we made earlier.
        w = WCS(pixel_map_path)

        # Get the pixel scale as well for single value conversions.
        try:
            pix_scale = fits.getval(pixel_map_path, 'PXSCAL2')
        except KeyError:    # Just in case the file doesn't have 'PXSCAL2'
            try:
                pix_scale = fits.getval(pixel_map_path, 'CDELT2') * 3600
            except KeyError:  # If both cases fail report the cluster and the problem
                print("Header is missing both 'PXSCAL2' and 'CDELT2'. Please check the header of: ",
                      pixel_map_path)
                raise

        # Read in the coverage mask data and header.
        coverage, head = fits.getdata(pixel_map_path, header=True, ignore_missing_end=True)

        for j in range(len(reg_files)):

            # Find the correct regions file corresponding to the cluster.
            if reg_files[j].startswith(sex_cat_path[-23:-7]):

                # Open the regions file and get the lines containing the shapes.
                with open(reg_file_dir + reg_files[j]) as lines:
                    for line in islice(lines, 3):
                        pass
                    objs = list(lines)

                # For each shape extract the defining parameters.
                for item in objs:
                    mask = item.strip()

                    # For circle shapes we need the center coordinate and the radius.
                    if mask.startswith('circle'):
                        params = mask[7:-1].split(',')

                        params[0] = float(params[0])        # center RA degrees
                        params[1] = float(params[1])        # center Dec degrees
                        params[2] = float(params[2][:-1])   # radius arcsec

                        # Convert the center coordinates into pixel system.
                        # "0" is to correct the pixel coordinates to the right origin for the data.
                        cent_x, cent_y = w.wcs_world2pix(params[0], params[1], 0)

                        # Generate the mask shape.
                        shape = Path.circle(center=(cent_x, cent_y), radius=params[2] / pix_scale)

                    # For the box we'll need...
                    elif mask.startswith('box'):
                        params = mask[4:-1].split(',')

                        params[0] = float(params[0])        # center RA degrees
                        params[1] = float(params[1])        # center Dec degrees
                        params[2] = float(params[2][:-1])   # width arcsec
                        params[3] = float(params[3][:-1])   # height arcsec
                        params[4] = float(params[4])        # rotation degrees

                        # Convert the center coordinates into pixel system.
                        cent_x, cent_y = w.wcs_world2pix(params[0], params[1], 0)

                        # Vertices of the box are needed for the path object to work.
                        verts = [[cent_x - 0.5 * (params[2] / pix_scale), cent_y + 0.5 * (params[3] / pix_scale)],
                                 [cent_x + 0.5 * (params[2] / pix_scale), cent_y + 0.5 * (params[3] / pix_scale)],
                                 [cent_x + 0.5 * (params[2] / pix_scale), cent_y - 0.5 * (params[3] / pix_scale)],
                                 [cent_x - 0.5 * (params[2] / pix_scale), cent_y - 0.5 * (params[3] / pix_scale)]]

                        # For rotations of the box.
                        rot = Affine2D().rotate_deg_around(cent_x, cent_y, degrees=params[4])

                        # Generate the mask shape.
                        shape = Path(verts).transformed(rot)

                    # Return error if mask shape isn't known.
                    else:
                        raise KeyError('Mask shape is unknown, please check the region file of cluster:',
                                       reg_files[j])

                    # Check if the pixel values are within the shape we defined earlier.
                    # If true, set the pixel value to 0.
                    for y in range(fits.getval(pixel_map_path, 'NAXIS2')):
                        for x in range(fits.getval(pixel_map_path, 'NAXIS1')):
                            if shape.contains_point([x, y]):
                                coverage[y, x] = 0

                # Write the new mask to disk overwriting the old mask.
                new_mask_hdu = fits.PrimaryHDU(coverage, header=head)
                new_mask_hdu.writeto(pixel_map_path, overwrite=True)

    return cluster_info


def object_selection(cluster_info, mag, cat_ra='RA', cat_dec='DEC',
                     sex_flag_cut=4, snr_cut=5.0, mag_cut=18.0, ch1_ch2_color_cut=0.7):
    """
    Reads in the SExtractor catalogs and performs all necessary cuts to select the AGN in the cluster. First, a cut is
    made on the SExtractor flag, the SNR is calculated, a flat magnitude cut is made to keep the completeness correction
    small, then a IRAC Ch1 - Ch2 color cut preformed to select the AGN, finally the AGN candidates are checked against
    the mask to verify the objects lie on good pixels in the original images.

    :param cluster_info: Array.
        An array of lists containing the paths to the clusters' files.
    :param mag: String.
        Specifies which IRAC band the magnitude should be computed on.
    :param cat_ra: String.
        Catalog RA column name label. Defaults to 'RA'.
    :param cat_dec: String.
        Catalog Dec column name label. Defaults to 'DEC'.
    :param sex_flag_cut: Integer.
        SExtractor flag value to cut on. Values less than the entered value are accepted. Defaults to < 4.
    :param snr_cut: Float.
        Signal to Noise Ratio value to cut on. Values greater than the entered value are accepted. Defaults to > 5.
    :param mag_cut: Float.
        Flat magnitude value in IRAC band specified by the 'mag' parameter. Values less than the entered value are
        accepted. This should be chosen so that the completeness correction is kept small. Defaults to < 18.0.
    :param ch1_ch2_color_cut: Float.
        IRAC Ch1 - Ch2 color value to cut on. Values greater than the entered value are accepted. This is chosen as a
        flat color cut based on the Stern+05 AGN wedge. Defaults to > 0.7.
    :return cluster_info: Array.
        An array in the same format as the input cluster_info for the matched clusters with the addition of the
        SExtractor catalog that is read in and had selection cuts preformed on it.
        0: SExtractor catalog path name.
        1: IRAC Ch1 science image path name.
        2: IRAC Ch1 coverage map path name.
        3: IRAC Ch2 science map path name.
        4: IRAC Ch2 coverage map path name.
        5: Catalog ID.
        6: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
        7: Masking Flag.
            0: No additional masking required, 1: Object masking needed, have regions file, 2: Further attention needed.
        8: Coverage good/bad pixel map path name.
        9: SExtractor catalog.
    """

    # Array element names
    sex_cat_path = cluster_info[0]
    pixel_mask_path = cluster_info[8]

    # Read in the catalog
    catalog = ascii.read(sex_cat_path)

    # Preform SExtractor Flag cut
    catalog = catalog[np.where(catalog['FLAG'] < sex_flag_cut)]

    # Calculate SNR in both bands (4" apertures)
    catalog['SNR_ch1'] = catalog['I1_FLUX_APER4'] / catalog['I1_FLUXERR_APER4']
    catalog['SNR_ch2'] = catalog['I2_FLUX_APER4'] / catalog['I2_FLUXERR_APER4']

    # Preform SNR cut
    catalog = catalog[np.where(catalog['SNR_ch1'] >= snr_cut)]
    catalog = catalog[np.where(catalog['SNR_ch2'] >= snr_cut)]

    # Preform Magnitude cut
    catalog = catalog[np.where(catalog[mag] <= mag_cut)]

    # Calculate the IRAC Ch1 - Ch2 color (4" apertures)
    catalog['color'] = catalog['I1_MAG_APER4'] - catalog['I2_MAG_APER4']

    # Preform the color cut
    catalog = catalog[np.where(catalog['color'] >= ch1_ch2_color_cut)]

    # For the mask cut we need to check the pixel value for each object's centroid.
    # Read in the mask file
    mask = fits.getdata(pixel_mask_path)

    # Read in the WCS from the mask
    w = WCS(pixel_mask_path)

    # Initialize the output catalog
    final_catalog = []

    for obj in catalog:
        # Get the object's pixel coordinates
        x_pix, y_pix = w.wcs_world2pix(obj[cat_ra], obj[cat_dec], 0)

        # Get the pixel value in the mask for the object's pixel coordinate
        mask_value = mask[int(round(y_pix)), int(round(x_pix))]

        # Check if the pixel value is good and if so add that row to the final catalog
        if mask_value == 1:
            if len(final_catalog) == 0:
                final_catalog = Table(obj)
            else:
                final_catalog.add_row(obj)

    # Clean up the catalog
    del final_catalog['SNR_ch1']
    del final_catalog['SNR_ch2']
    del final_catalog['color']

    # Append the final catalog to the output array
    cluster_info.append(final_catalog)

    return cluster_info


def catalog_match(cluster_info, master_catalog, catalog_cols, sex_ra_col='RA', sex_dec_col='DEC',
                  master_ra_col='RA', master_dec_col='DEC'):
    """
    Pairs the SExtractor catalogs with the master catalog.
    :param cluster_info: Array.
        An array of lists containing the paths to the clusters' files.
    :param master_catalog: Astropy table object.
        Catalog containing information about the cluster as a whole.
    :param catalog_cols: List of strings.
        List of column names in master catalog that we wish to incorporate into the cluster-specific catalogs.
    :param sex_ra_col: String.
        RA column name in master catalog. Defaults to 'RA'.
    :param sex_dec_col: String.
        Dec column name in SExtractor catalog. Defaults to 'DEC'.
    :param master_ra_col: String.
        RA column name in SExtractor catalog. Defaults to 'RA'.
    :param master_dec_col: String.
        Dec column name in master catalog. Defaults to 'DEC'.
    :return cluster_info: Array.
        An array in the same format as the input cluster_info for the matched clusters with the addition of the catalog
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
        8: Coverage good/bad pixel map path name.
        9: SExtractor catalog.
    """

    # Array element names
    catalog_id = cluster_info[5]
    sex_catalog = cluster_info[9]

    # We already matched our SExtractor catalogs to the master catalog so we only need to pull the correct row.
    # The master catalog index is stored in cluster_info[5].
    # Create astropy skycoord object from the catalog columns.
    cat_coords = SkyCoord(master_catalog[master_ra_col][catalog_id],
                          master_catalog[master_dec_col][catalog_id], unit=u.degree)

    # List to hold the separations
    separation = []

    # For all objects in catalog find the angular separation between the object's coordinate and the
    # cluster-centered coordinate.
    # Note: Astropy .separation() calculates the great-circle distance via the Vincenty formula *not* the small
    # angle approximation.
    for j in range(len(sex_catalog)):
        sexcat_coords = SkyCoord(sex_catalog[sex_ra_col][j], sex_catalog[sex_dec_col][j], unit=u.degree)
        separation.append(sexcat_coords.separation(cat_coords).arcsec)

    # For all requested columns from the master catalog add the value to all columns in the SExtractor catalog.
    for col_name in catalog_cols:
        sex_catalog[col_name] = master_catalog[col_name][catalog_id]

    # Store all the separations in as a column in the catalog.
    sex_catalog['rad_dist'] = separation

    return cluster_info


def final_catalogs(cluster_info, catalog_cols):
    """
    Writes the final catalogs to disk.
    :param cluster_info: Array.
        An array of lists containing the paths to the clusters' files and the SExtractor catalog stored in memory.
    :param catalog_cols: List of strings.
        List of column names in the SExtractor catalog that we wish to include in the final version of the catalog.
    :return:
    """

    # Array element names
    sex_cat_path = cluster_info[0]
    sex_catalog = cluster_info[9]

    final_cat = sex_catalog[catalog_cols]

    final_cat_path = 'Data/Output/' + sex_cat_path[-23:-7] + '_AGN.cat'

    ascii.write(final_cat, final_cat_path)


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



