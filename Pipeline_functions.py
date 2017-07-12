"""
.. Pipeline_functions.py
.. Author: Benjamin Floyd

This script is designed to automate the process of selecting the AGN in the SPT clusters and generating the proper
masks needed for determining the feasible area for calculating a surface density.
"""
from __future__ import print_function, division

import warnings  # For suppressing the astropy warnings that pop up when reading headers.
from itertools import ifilter
from os import listdir, system, chmod

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from astropy.table import Table, unique
from astropy.utils.exceptions import AstropyWarning  # For suppressing the astropy warnings.
from astropy.wcs import WCS
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from scipy.interpolate import interp1d

# Suppress Astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)


def file_pairing(cat_dir, img_dir):
    """
    Given directories of catalogs and images this method will pair the filenames corresponding to a cluster based
    on the coordinate listed in the filenames and then return an array containing the paths to the files.

    :param cat_dir:
        Path to directory with catalogs.
    :param img_dir:
        Path to directory with image files.
    :type cat_dir: str
    :type img_dir: str
    :return clusters: 
        A list consisting of dictionaries containing the cluster's filenames with the following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
    :rtype: list
    """

    # Add a trailing '/' to the directory paths if not present.
    if not cat_dir.endswith('/'):
        cat_dir = cat_dir + '/'
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'

    # Generate lists of all the catalog files and all the image files.
    cat_files = [f for f in listdir(cat_dir) if not f.startswith('.')]
    image_files = [f for f in listdir(img_dir) if not f.startswith('.')]

    # Pair the catalogs with their corresponding images.
    clusters = [{} for _ in range(len(cat_files))]
    for i in range(len(cat_files)):
        clusters[i]['sex_cat_path'] = cat_dir + cat_files[i]
        cat_coord = cat_files[i][7:16]
        for image in image_files:
            img_coord = image[10:19]
            if cat_coord == img_coord:
                if image.startswith('I1') and '_cov' not in image:
                    clusters[i]['ch1_sci_path'] = img_dir + image
                elif image.startswith('I1') and '_cov' in image:
                    clusters[i]['ch1_cov_path'] = img_dir + image
                elif image.startswith('I2') and '_cov' not in image:
                    clusters[i]['ch2_sci_path'] = img_dir + image
                elif image.startswith('I2') and '_cov' in image:
                    clusters[i]['ch2_cov_path'] = img_dir + image

    return [x for x in clusters if len(x) == 5]


def catalog_image_match(cluster_list, catalog, cat_ra_col='RA', cat_dec_col='DEC', max_sep=1.0):
    """
    Matches the center coordinate of a fits image to a catalog of objects then returns an array containing the paths to
    the files that have a match in the catalog.

    :param cluster_list:
        A list of dictionaries containing the paths to the clusters' files.
    :param catalog:
        The catalog we wish to check our images against.
    :param cat_ra_col:
        RA column name in catalog. Defaults to 'RA'.
    :param cat_dec_col:
        Dec column name in catalog. Defaults to 'DEC'.
    :param max_sep:
        Maximum separation (in arcminutes) allowed between the image center coordinate and the Bleem cluster center
        coordinate. Defaults to 1.0 arcminute.
    :type cluster_list: list
    :type catalog: Astropy table object
    :type cat_ra_col: str
    :type cat_dec_col: str
    :return cluster_list:
        A list of dictionaries for the matched clusters with the path names to the files and the index value in the
        catalog for the cluster and the separation between the image coordinate and the catalog coordinate with the
        following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
            :Bleem_idx: Index in catalog corresponding to the match.
            :center_sep: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
    :rtype: list
    """

    # Create astropy skycoord object from the catalog columns.
    cat_coords = SkyCoord(catalog[cat_ra_col], catalog[cat_dec_col], unit=u.degree)

    for cluster in cluster_list:
        # Array element names
        irac_ch1_sci = cluster['ch1_sci_path']

        # Get the RA and Dec of the center pixel in the image.
        img_ra = fits.getval(irac_ch1_sci, 'CRVAL1')
        img_dec = fits.getval(irac_ch1_sci, 'CRVAL2')

        # Create astropy skycoord object for the center pixel of the image.
        img_coord = SkyCoord(img_ra, img_dec, unit=u.degree)

        # Preform the catalog matching.
        idx, sep, _ = img_coord.match_to_catalog_sky(cat_coords)

        # Add the (nearest) catalog id and separation (in arcsec) to the output array.
        cluster.update({'Bleem_idx': idx, 'center_sep': sep.arcmin.item(), 'SPT_ID': catalog[idx]['SPT_ID']})

    # Reject any match with a separation larger than 1 arcminute.
    cluster_list = [cluster for cluster in cluster_list if cluster['center_sep'] <= max_sep]

    # If there are any duplicate matches in the sample remaining we need to remove the match that is the poorer
    # match. We will only keep the closest matches.
    # First set up a table of the index of the cluster dictionaries in cluster_list, the recorded Bleem index, and
    # the recorded separation.
    match_info = Table(names=['list_idx', 'Bleem_idx', 'center_sep'], dtype=['i8', 'i8', 'f8'])
    for i in range(len(cluster_list)):
        match_info.add_row([i, cluster_list[i]['Bleem_idx'], cluster_list[i]['center_sep']])

    # Sort the table by the Bleem index.
    match_info.sort(['Bleem_idx', 'center_sep'])

    # Use Astropy's unique function to remove the duplicate rows. Because the table rows will be subsorted by the
    # separation column we only need to keep the first incidence of the Bleem index as our best match.
    match_info = unique(match_info, keys='Bleem_idx', keep='first')

    # Resort the table by the list index (not sure if this is necessary).
    match_info.sort('list_idx')

    # Generate the output list using the remaining indices in the table.
    cluster_list = [cluster_list[i] for i in match_info['list_idx']]

    return cluster_list


def mask_flag(cluster_list, mask_file):
    """
    Appends the masking flag to the catalog list array.

    :param cluster_list:
        A list of dictionaries containing the paths to the clusters' files and other information about the cluster.
    :param mask_file:
        An external text file with at least two tab delimited columns. The first must be the path name of the SExtractor
        catalog, the second should be an integer [0, 1, 2, 3] indicating the degree of severity of masking issues in the
        field.
    :type cluster_list: list
    :type mask_file: str
    :return cluster_info:
        A list of dictionaries for the matched clusters with the path names to the files, the index value in the
        catalog for the cluster, the separation between the image coordinate and the catalog coordinate, and the masking
        flag from the external masking catalog with the following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
            :Bleem_idx: Index in catalog corresponding to the match.
            :center_sep: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
            :mask_flag: Masking Flag with one of the following values.

                * 0: No additional masking required,
                * 1: Object masking needed, have regions file,
                * 2: Further attention needed,
                * 3: Remove cluster from sample (these should not show up).

    :rtype: list
    """

    # Open the masking notes file
    mask_notes = ascii.read(mask_file, names=['catalog', 'flag'])

    # Pair the clusters in the list with those in the masking notes file.
    for cluster in cluster_list:
        # Array element names
        cutout_id = cluster['SPT_ID']

        # Go through the masking notes file and match the flag value with the correct cluster.
        for row in mask_notes:
            if cutout_id in row['catalog']:
                cluster.update({'mask_flag': row['flag']})

    return cluster_list


def coverage_mask(cluster_info, ch1_min_cov, ch2_min_cov):
    """
    Generates a pixel mask of good pixels where both IRAC Ch1 and Ch2 coverage values are above a specified value.

    :param cluster_info:
        A dictionary containing the paths to the clusters' files and other information about the cluster.
    :param ch1_min_cov:
        Minimum coverage value allowed in IRAC Ch1.
    :param ch2_min_cov:
        Minimum coverage value allowed in IRAC Ch2.
    :type cluster_info: dict
    :type ch1_min_cov: float
    :type ch2_min_cov: float
    :return cluster_info: 
        A dictionary for the cluster with the path names to the files, the index value in the master catalog for the
        cluster, the separation between the image coordinate and the catalog coordinate, the masking flag from the
        external masking catalog, and the path name of the coverage pixel mask with the following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
            :Bleem_idx: Index in catalog corresponding to the match.
            :center_sep: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
            :mask_flag: Masking Flag with one of the following values.

                * 0: No additional masking required,
                * 1: Object masking needed, have regions file,
                * 2: Further attention needed,
                * 3: Remove cluster from sample (these should not show up).

            :cov_mask_path: Coverage good/bad pixel map path name.
    :rtype: dict
    """

    # Array element names
    spt_id = cluster_info['SPT_ID']
    irac_ch1_cov_path = cluster_info['ch1_cov_path']
    irac_ch2_cov_path = cluster_info['ch2_cov_path']

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
    mask_pathname = 'Data/Masks/{cluster_id}_cov_mask{ch1_cov}_{ch2_cov}.fits'\
        .format(cluster_id=spt_id, ch1_cov=ch1_min_cov, ch2_cov=ch2_min_cov)
    combined_cov_hdu = fits.PrimaryHDU(combined_cov, header=head)
    combined_cov_hdu.writeto(mask_pathname, overwrite=True)

    # Append the new coverage mask path name and both the catalog and the masking flag from cluster_info
    # to the new output list.
    cluster_info.update({'cov_mask_path': mask_pathname})

    return cluster_info


def object_mask(cluster_info, reg_file_dir):
    """
    For the clusters that have objects that require additional masking apply the masks based on the previously generated
    ds9 regions files.

    :param cluster_info:
        A dictionary containing the paths to the clusters' files and other information about the cluster.
    :param reg_file_dir:
        Path to directory containing the ds9 regions files.
    :type cluster_info: dict
    :type reg_file_dir: str
    :return cluster_info:
        A dictionary for the cluster with the path names to the files, the index value in the master catalog for the
        cluster, the separation between the image coordinate and the catalog coordinate, the masking flag from the
        external masking catalog, and the path name of the coverage pixel mask with the following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
            :Bleem_idx: Index in catalog corresponding to the match.
            :center_sep: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
            :mask_flag: Masking Flag with one of the following values.

                * 0: No additional masking required,
                * 1: Object masking needed, have regions file,
                * 2: Further attention needed,
                * 3: Remove cluster from sample (these should not show up).

            :cov_mask_path: Coverage good/bad pixel map path name.
    :rtype: dict
    :raises:
        :KeyError: 
            If the pixel mask file is missing a pixel scale value in its header.
        :KeyError: 
            If the object masking shape is unknown.
    """

    # Add a trailing '/' to the directory paths if not present.
    if not reg_file_dir.endswith('/'):
        reg_file_dir = reg_file_dir + '/'

    # Region file directory files
    reg_files = [f for f in listdir(reg_file_dir) if not f.startswith('.')]

    # Array element names
    spt_id = cluster_info['SPT_ID']
    flag = cluster_info['mask_flag']
    pixel_map_path = cluster_info['cov_mask_path']

    # If the masking flag == 0 then no additional masking is required.
    # Otherwise go get the appropriate regions file and append the mask to the coverage pixel map.
    if flag != 0:

        # Read in the WCS from the coverage mask we made earlier.
        w = WCS(pixel_map_path)

        # Get the pixel scale as well for single value conversions.
        try:
            pix_scale = fits.getval(pixel_map_path, 'PXSCAL2')
        except KeyError:  # Just in case the file doesn't have 'PXSCAL2'
            try:
                pix_scale = fits.getval(pixel_map_path, 'CDELT2') * 3600
            except KeyError:  # If both cases fail report the cluster and the problem
                print("Header is missing both 'PXSCAL2' and 'CDELT2'. Please check the header of: {file}"
                      .format(file=pixel_map_path))
                raise

        # Read in the coverage mask data and header.
        coverage, head = fits.getdata(pixel_map_path, header=True, ignore_missing_end=True, memmap=False)

        for j in range(len(reg_files)):

            # Find the correct regions file corresponding to the cluster.
            if spt_id in reg_files[j]:

                # Open the regions file and get the lines containing the shapes.
                with open(reg_file_dir + reg_files[j]) as region:
                    objs = []
                    for line in ifilter(lambda _: _.startswith('circle') or _.startswith('box'), region):
                        objs.append(line.strip())

                # For each shape extract the defining parameters.
                for mask in objs:

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
                        raise KeyError('Mask shape is unknown, please check the region file of cluster: {region} {mask}'
                                       .format(region=reg_files[j], mask=mask))

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


def object_selection(cluster_info, mag, cat_ra='RA', cat_dec='DEC', sex_flag_cut=4, mag_cut=18., ch1_ch2_color_cut=0.7):
    """
    Reads in the SExtractor catalogs and performs all necessary cuts to select the AGN in the cluster. First, a cut is
    made on the SExtractor flag, the SNR is calculated, a flat magnitude cut is made to keep the completeness correction
    small, then a IRAC Ch1 - Ch2 color cut preformed to select the AGN, finally the AGN candidates are checked against
    the mask to verify the objects lie on good pixels in the original images.

    :param cluster_info:
        A dictionary containing the paths to the clusters' files and other information about the cluster.
    :param mag:
        Specifies which IRAC band the magnitude should be computed on.
    :param cat_ra:
        Catalog RA column name label. Defaults to 'RA'.
    :param cat_dec:
        Catalog Dec column name label. Defaults to 'DEC'.
    :param sex_flag_cut:
        SExtractor flag value to cut on. Values less than the entered value are accepted. Defaults to < 4.
    :param mag_cut:
        Flat magnitude value in IRAC band specified by the 'mag' parameter. Values less than the entered value are
        accepted. This should be chosen so that the completeness correction is kept small. Defaults to < 18.0.
    :param ch1_ch2_color_cut:
        IRAC Ch1 - Ch2 color value to cut on. Values greater than the entered value are accepted. This is chosen as a
        flat color cut based on the Stern+05 AGN wedge. Defaults to > 0.7.
    :type cluster_info: dict
    :type mag: str
    :type cat_ra: str
    :type cat_dec: str
    :type sex_flag_cut: int
    :type mag_cut: float
    :type ch1_ch2_color_cut: float
    :return cluster_info:
        A dictionary for the cluster with the path names to the files, the index value in the master catalog for the
        cluster, the separation between the image coordinate and the catalog coordinate, the masking flag from the
        external masking catalog, the path name of the coverage pixel mask, and the SExtractor catalog (loaded into
        memory) with the following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
            :Bleem_idx: Index in catalog corresponding to the match.
            :center_sep: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
            :mask_flag: Masking Flag with one of the following values.

                * 0: No additional masking required,
                * 1: Object masking needed, have regions file,
                * 2: Further attention needed,
                * 3: Remove cluster from sample (these should not show up).

            :cov_mask_path: Coverage good/bad pixel map path name.
            :catalog: SExtractor catalog.
    :rtype: dict
    """

    # Array element names
    sex_cat_path = cluster_info['sex_cat_path']
    pixel_mask_path = cluster_info['cov_mask_path']

    # Read in the catalog
    catalog = ascii.read(sex_cat_path)

    # Preform SExtractor Flag cut
    catalog = catalog[np.where(catalog['FLAGS'] < sex_flag_cut)]

    # Preform Magnitude cut
    catalog = catalog[np.where(catalog[mag] <= mag_cut)]

    # Preform saturation cuts using limits from Eisenhardt et al. 2004
    catalog = catalog[np.where(catalog['I1_MAG_APER4'] > 10.0)]  # [3.6] saturation limit
    catalog = catalog[np.where(catalog['I2_MAG_APER4'] > 9.8)]   # [4.5] saturation limit

    # Calculate the IRAC Ch1 - Ch2 color (4" apertures) and preform the color cut
    catalog = catalog[np.where(catalog['I1_MAG_APER4'] - catalog['I2_MAG_APER4'] >= ch1_ch2_color_cut)]

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

    # # Clean up the catalog
    # del final_catalog['SNR_ch1']
    # del final_catalog['SNR_ch2']
    # del final_catalog['color']

    # Append the final catalog to the output array
    cluster_info.update({'catalog': final_catalog})

    return cluster_info


def catalog_match(cluster_info, master_catalog, catalog_cols, sex_ra_col='RA', sex_dec_col='DEC',
                  master_ra_col='RA', master_dec_col='DEC'):
    """
    Pairs the SExtractor catalogs with the master catalog.

    :param cluster_info:
        A dictionary containing the paths to the clusters' files and other information about the cluster.
    :param master_catalog:
        Catalog containing information about the cluster as a whole.
    :param catalog_cols:
        List of column names in master catalog that we wish to incorporate into the cluster-specific catalogs.
    :param sex_ra_col:
        RA column name in SExtractor catalog. Defaults to 'RA'.
    :param sex_dec_col:
        Dec column name in SExtractor catalog. Defaults to 'DEC'.
    :param master_ra_col:
        RA column name in master catalog. Defaults to 'RA'.
    :param master_dec_col:
        Dec column name in master catalog. Defaults to 'DEC'.
    :type cluster_info: dict
    :type master_catalog: Astropy table object
    :type catalog_cols: list of strings
    :type sex_ra_col: str
    :type sex_dec_col: str
    :type master_ra_col: str
    :type master_dec_col: str
    :return cluster_info:
        A dictionary for the cluster with the path names to the files, the index value in the master catalog for the
        cluster, the separation between the image coordinate and the catalog coordinate, the masking flag from the
        external masking catalog, the path name of the coverage pixel mask, and the SExtractor catalog (loaded into
        memory) with the following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
            :Bleem_idx: Index in catalog corresponding to the match.
            :center_sep: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
            :mask_flag: Masking Flag with one of the following values.

                * 0: No additional masking required,
                * 1: Object masking needed, have regions file,
                * 2: Further attention needed,
                * 3: Remove cluster from sample (these should not show up).

            :cov_mask_path: Coverage good/bad pixel map path name.
            :catalog: SExtractor catalog.
    :rtype: dict
    """

    # Array element names
    catalog_idx = cluster_info['Bleem_idx']
    sex_catalog = cluster_info['catalog']
    spt_id = cluster_info['SPT_ID']

    # We already matched our SExtractor catalogs to the master catalog so we only need to pull the correct row.
    # The master catalog index is stored in cluster_info['Bleem_idx'].
    # Create astropy skycoord object from the catalog columns.
    cat_coords = SkyCoord(master_catalog[master_ra_col][catalog_idx],
                          master_catalog[master_dec_col][catalog_idx], unit=u.degree)

    # List to hold the separations
    separation = []

    # For all objects in catalog find the angular separation between the object's coordinate and the
    # cluster-centered coordinate.
    # Note: Astropy .separation() calculates the great-circle distance via the Vincenty formula *not* the small
    # angle approximation.
    for j in range(len(sex_catalog)):
        sexcat_coords = SkyCoord(sex_catalog[sex_ra_col][j], sex_catalog[sex_dec_col][j], unit=u.degree)
        separation.append(sexcat_coords.separation(cat_coords).arcmin)

    # Replace the existing SPT_ID in the SExtractor catalog with the official one from Bleem+15.
    # First change the data type of the column to str16 so the ID can fit in the column
    sex_catalog['SPT_ID'] = sex_catalog['SPT_ID'].astype('S16')

    # Then replace the column values with the one we have stored in the dictionary.
    sex_catalog['SPT_ID'] = spt_id

    # For all requested columns from the master catalog add the value to all columns in the SExtractor catalog.
    for col_name in catalog_cols:
        sex_catalog[col_name] = master_catalog[col_name][catalog_idx]

    # Store all the separations in as a column in the catalog.
    sex_catalog['RADIAL_DIST'] = separation

    return cluster_info


def image_area(cluster_info, units='arcmin'):
    """
    Uses the pixel mask of the image and calculates the total area of all good pixels then add this value to the catalog
    as a column.

    :param cluster_info:
        A dictionary containing the paths to the clusters' files and other information about the cluster.
    :param units:
        The units we wish the area to be in. Defaults to `arcmin`.
    :type cluster_info: dict
    :type units: str
    :return cluster_info:
        A dictionary for the cluster with the path names to the files, the index value in the master catalog for the
        cluster, the separation between the image coordinate and the catalog coordinate, the masking flag from the
        external masking catalog, the path name of the coverage pixel mask, and the SExtractor catalog (loaded into
        memory) with the following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
            :Bleem_idx: Index in catalog corresponding to the match.
            :center_sep: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
            :mask_flag: Masking Flag with one of the following values.

                * 0: No additional masking required,
                * 1: Object masking needed, have regions file,
                * 2: Further attention needed,
                * 3: Remove cluster from sample (these should not show up).

            :cov_mask_path: Coverage good/bad pixel map path name.
            :catalog: SExtractor catalog.
    :rtype: dict
    """

    # Dictionary element names.
    catalog = cluster_info['catalog']
    pixel_mask_path = cluster_info['cov_mask_path']

    # Read in the pixel mask.
    pixel_mask = fits.getdata(pixel_mask_path)

    # The pixel mask has values of only 0 (bad) or 1 (good) so a simple sum of all the values will give us the total
    # number of good pixels in the mask.
    good_pixels = pixel_mask.flatten().sum()

    # Get the pixel scale.
    try:
        pix_scale = fits.getval(pixel_mask_path, 'PXSCAL2') * u.arcsec
    except KeyError:  # Just in case the file doesn't have 'PXSCAL2'
        try:
            pix_scale = fits.getval(pixel_mask_path, 'CDELT2') * u.deg
        except KeyError:  # If both cases fail report the cluster and the problem
            print("Header is missing both 'PXSCAL2' and 'CDELT2'. Please check the header of: {file}"
                  .format(file=pixel_mask_path))
            raise

    # Convert the pixel scale to whatever units were requested.
    pix_scale = pix_scale.to(u.Unit(units))

    # Now simply convert our number of pixels into a sky area using the pixel scale.
    total_area = good_pixels * pix_scale.value * pix_scale.value

    # Add the area to the catalog as a column
    catalog['IMAGE_AREA'] = total_area

    return cluster_info


def completeness_value(cluster_info, mag, completeness_dict):
    """
    Takes the completeness curve values for the cluster, interpolates a function between the discrete values, then
    queries the specified magnitude of the selected objects in the SExtractor catalog and adds two columns: The
    completeness value of that object at its magnitude and the completeness correction value (1/[completeness value]).

    :param cluster_info:
        A dictionary containing the paths to the clusters' files and other information about the cluster.
    :param mag:
        Specifies which IRAC magnitude corresponds to the magnitude used to generate the completeness curve values.
    :param completeness_dict:
        A dictionary containing completeness curve values of a specific magnitude for all clusters in the sample.
    :type cluster_info: dict
    :type mag: str
    :type completeness_dict: dict
    :return cluster_info:
        A dictionary for the cluster with the path names to the files, the index value in the master catalog for the
        cluster, the separation between the image coordinate and the catalog coordinate, the masking flag from the
        external masking catalog, the path name of the coverage pixel mask, and the SExtractor catalog (loaded into
        memory) with the following keynames.
            :sex_cat_path: SExtractor catalog path name.
            :ch1_sci_path: IRAC Ch1 science image path name.
            :ch1_cov_path: IRAC Ch1 coverage map path name.
            :ch2_sci_path: IRAC Ch2 science map path name.
            :ch2_cov_path: IRAC Ch2 coverage map path name.
            :Bleem_idx: Index in catalog corresponding to the match.
            :center_sep: Separation (in arcsec) between catalog RA/Dec and image center pixel RA/Dec.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
            :mask_flag: Masking Flag with one of the following values.

                * 0: No additional masking required,
                * 1: Object masking needed, have regions file,
                * 2: Further attention needed,
                * 3: Remove cluster from sample (these should not show up).

            :cov_mask_path: Coverage good/bad pixel map path name.
            :catalog: SExtractor catalog.
    :rtype: dict
    """

    # Array element names
    spt_id = cluster_info['SPT_ID']
    sex_catalog = cluster_info['catalog']

    # Select the correct entry in the dictionary corresponding to our cluster.
    completeness_data = [value for key, value in completeness_dict.items() if spt_id in key][0]

    # Also grab the magnitude bins used to create the completeness data
    mag_bins = completeness_dict['magnitude_bins'][:-1]

    # Interpolate the completeness data into a functional form using linear interpolation
    completeness_funct = interp1d(mag_bins, completeness_data, kind='linear')

    # For the objects' magnitude specified by `mag` query the completeness function to find the completeness value.
    completeness_values = completeness_funct(sex_catalog[mag])

    # The completeness correction values are defined as 1/[completeness value]
    completeness_corrections = 1 / completeness_values

    # Add the completeness values and corrections to the SExtractor catalog.
    sex_catalog['completeness_value'] = completeness_values
    sex_catalog['completeness_correction'] = completeness_corrections

    return cluster_info


def final_catalogs(cluster_info, catalog_cols):
    """
    Writes the final catalogs to disk.

    :param cluster_info:
        A dictionary for the cluster with the path names to the files, the index value in the master catalog for the
        cluster, the separation between the image coordinate and the catalog coordinate, the masking flag from the
        external masking catalog, the path name of the coverage pixel mask, and the SExtractor catalog (loaded into
        memory) with the following required keynames.
            :SPT_ID: String containing the official Bleem SPT ID for the cluster.
            :catalog: SExtractor catalog.
    :param catalog_cols:
        List of column names in the SExtractor catalog that we wish to include in the final version of the catalog.
    :type cluster_info: dict
    :type catalog_cols: list of str
    """

    # Array element names
    spt_id = cluster_info['SPT_ID']
    sex_catalog = cluster_info['catalog']

    final_cat = sex_catalog[catalog_cols]

    final_cat_path = 'Data/Output/{cluster_id}_AGN.cat'.format(cluster_id=spt_id)

    ascii.write(final_cat, final_cat_path)


def visualizer(cluster_list):
    """
    Creates a script to view all cluster images in ds9.

    :param cluster_list:
        An array of lists containing the paths to the clusters' files.
    :type cluster_list: list
    """

    with open('ds9viz', mode='w') as script:
        script.write('#!/bin/tcsh\n')
        script.write('ds9 -single ')
        for cluster in cluster_list:
            script.write(cluster['ch1_sci_path'])
            script.write(' ')
            script.write(cluster['ch2_sci_path'])
            script.write(' ')

    chmod('ds9viz', 0o755)
    system('./ds9viz')
