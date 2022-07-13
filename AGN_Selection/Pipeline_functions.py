"""
.. Pipeline_functions.py
.. Author: Benjamin Floyd

This script is designed to automate the process of selecting the AGN in the SPT clusters and generating the proper
masks needed for determining the feasible area for calculating a surface density.
"""
import glob
import json
import re
import warnings
from collections import ChainMap
from itertools import groupby, product, chain

import numpy as np
from astro_compendium.utils.k_correction import k_correction, k_corr_abs_mag
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, unique, vstack
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyWarning  # For suppressing the astropy warnings.
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from numpy.random import default_rng
from scipy.integrate import quad_vec, quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
from synphot import SourceSpectrum, SpectralElement
from schwimmbad import MultiPool

# Suppress Astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)


class SelectIRAGN:
    """
    Pipeline to select for IR-Bright AGN in the SPT cluster surveys.

    Parameters
    ----------
    sextractor_cat_dir : str or list of str
        Directory for the SExtractor photometric catalogs.
    irac_image_dir : str or list of str
        Directory for the IRAC 3.6 um and 4.5 um science and coverage images.
    region_file_dir : str or list of str
        Directory for DS9 regions files describing areas to mask manually.
    mask_dir : str
        Directory to write good pixel masks into
    spt_catalog : Table
        Official SPT cluster catalog.
    completeness_file : str or list of str
        File name of the completeness simulation results.
    field_number_dist_file : str
        Filename containing the number count histogram of all galaxies as a function of [3.6] - [4.5] color
        using the SDWFS field sample and the associated color bins.
    purity_color_threshold_file : str or None
        Filename containing the function values of the purity-based color selection thresholds and the associated
        redshift bins.
    sed : SourceSpectrum
        Spectral energy distribution template of the source object, used for K-correction/absolute magnitude.
    irac_filter : str
        Filename of the IRAC filter being used for absolute magnitude calculations.
    j_band_filter : str
        Filename of the FLAMINGOS J-band filter being used for absolute magnitude calculations.

    """

    def __init__(self, sextractor_cat_dir, irac_image_dir, region_file_dir, mask_dir, spt_catalog, completeness_file,
                 field_number_dist_file, purity_color_threshold_file, sed, irac_filter, j_band_filter):

        # Directory paths to files
        self._sextractor_cat_dir = sextractor_cat_dir
        self._irac_image_dir = irac_image_dir
        self._region_file_dir = region_file_dir
        self._mask_dir = mask_dir

        # Official SPT cluster catalog table
        self._spt_catalog = spt_catalog

        # File name of completeness simulation results
        self._completeness_results = completeness_file

        # Initialization of catalog dictionary data structure
        self._catalog_dictionary = {}

        # Set the cosmology to a standard concordance cosmology
        self._cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        # K-correction parameters
        self._sed = sed
        self._irac_filter = irac_filter
        self._j_band_filter = j_band_filter

        # Number count distribution from SDWFS used to remove Eddington bias
        self._field_number_dist = field_number_dist_file

        # Purity-based color threshold function. This is used in place of a flat color cut in `purify_selection`.
        self._purity_color_funct = purity_color_threshold_file

    def file_pairing(self, include=None, exclude=None):
        """
        Collates the different files for each cluster into a dictionary.

        This will also remove any clusters we request and perform a check that all clusters have the required files.

        Parameters
        ----------
        include : list_like, optional
            A list containing the names of the clusters we wish to filter for from the original sample.

        exclude : list_like, optional
            A list containing the names of the clusters we wish to remove from our sample.

        Raises
        ------
        UserWarning
            A warning is printed if a cluster is missing any necessary files. These clusters will be automatically
            removed from the sample.

        Notes
        -----
        The list provided by `include` is filtered for first before the `exclude` list is processed. This allows the
        sample to be filtered for a set of clusters we want to run the selection on while still allowing us to mark
        certain clusters to be removed from the sample.

        """

        # List the file names for both the images and the catalogs
        if isinstance(self._irac_image_dir, list):
            image_files = list(chain.from_iterable(glob.glob(f'{img_dir}/*.fits') for img_dir in self._irac_image_dir))
        else:
            image_files = glob.glob(f'{self._irac_image_dir}/*.fits')
        if isinstance(self._sextractor_cat_dir, list):
            cat_files = list(
                chain.from_iterable(glob.glob(f'{cat_dir}/*.cat') for cat_dir in self._sextractor_cat_dir))
        else:
            cat_files = glob.glob(f'{self._sextractor_cat_dir}/*.cat')

        # Combine and sort both file lists
        cat_image_files = sorted(cat_files + image_files, key=self._keyfunct)

        # Group the file names together
        self._catalog_dictionary = {cluster_id: list(files)
                                    for cluster_id, files in groupby(cat_image_files, key=self._keyfunct)}

        # If we want to only run on a set of clusters we can filter for them now
        if include is not None:
            self._catalog_dictionary = {cluster_id: files for cluster_id, files in self._catalog_dictionary.items()
                                        if cluster_id in include}

        # If we want to exclude some clusters manually we can remove them now
        if exclude is not None:
            for cluster_id in exclude:
                self._catalog_dictionary.pop(cluster_id, None)

        # Sort the files into a dictionary according to the type of file
        for cluster_id, files in self._catalog_dictionary.items():
            self._catalog_dictionary[cluster_id] = {}
            for f in files:
                if f.endswith('.cat'):
                    self._catalog_dictionary[cluster_id]['se_cat_path'] = f
                elif 'I1' in f and '_cov' not in f:
                    self._catalog_dictionary[cluster_id]['ch1_sci_path'] = f
                elif 'I1' in f and '_cov' in f:
                    self._catalog_dictionary[cluster_id]['ch1_cov_path'] = f
                elif 'I2' in f and '_cov' not in f:
                    self._catalog_dictionary[cluster_id]['ch2_sci_path'] = f
                elif 'I2' in f and '_cov' in f:
                    self._catalog_dictionary[cluster_id]['ch2_cov_path'] = f

        # Verify that all the clusters in our sample have all the necessary files
        problem_clusters = []
        for cluster_id, cluster_files in self._catalog_dictionary.items():
            file_keys = {'ch1_sci_path', 'ch1_cov_path', 'ch2_sci_path', 'ch2_cov_path', 'se_cat_path'}
            try:
                assert file_keys == cluster_files.keys()
            except AssertionError:
                message = f'Cluster {cluster_id} is missing files {file_keys - cluster_files.keys()}'
                warnings.warn(message)
                problem_clusters.append(cluster_id)

        # For now, remove the clusters missing files
        for cluster_id in problem_clusters:
            self._catalog_dictionary.pop(cluster_id, None)

    def image_to_catalog_match(self, max_image_catalog_sep):
        """
        Matches the science images to the official SPT catalog.

        Uses the center pixel coordinate of the 3.6 um science image to match against the SZ center of the
        clusters in the official SPT catalog. Clusters are kept only if their images match an SZ center within the given
        maximum separation. If multiple images match to the same SZ center within our max separation then only the
        closest match is kept.

        Parameters
        ----------
        max_image_catalog_sep : Quantity
            Maximum separation allowed between the image center pixel and the SZ center matched in the official SPT
            catalog.

        """

        catalog = self._spt_catalog

        # Create astropy skycoord object of the SZ centers.
        sz_centers = SkyCoord(catalog['RA'], catalog['DEC'], unit=u.degree)

        for cluster in self._catalog_dictionary.values():
            # Get the RA and Dec of the center pixel in the image.
            w = WCS(cluster['ch1_sci_path'])
            center_pixel = np.array(w.array_shape) // 2

            # Create astropy skycoord object for the reference pixel of the image.
            img_coord = SkyCoord.from_pixel(center_pixel[1], center_pixel[0], wcs=w, origin=0)

            # Match the reference pixel to the SZ centers
            idx, sep, _ = img_coord.match_to_catalog_sky(sz_centers)

            # Add the (nearest) catalog id and separation (in arcsec) to the output array.
            cluster.update({'SPT_cat_idx': idx, 'center_sep': sep})

        # Reject any match with a separation larger than 1 arcminute.
        large_sep_clusters = [cluster_id for cluster_id, cluster_info in self._catalog_dictionary.items()
                              if cluster_info['center_sep'].to(u.arcmin) > max_image_catalog_sep]
        for cluster_id in large_sep_clusters:
            self._catalog_dictionary.pop(cluster_id, None)

        # If there are any duplicate matches in the sample remaining we need to remove the match that is the poorer
        # match. We will only keep the closest matches.
        match_info = Table(rows=[[cluster['SPT_cat_idx'], cluster['center_sep'], cluster_id]
                                 for cluster_id, cluster in self._catalog_dictionary.items()],
                           names=['SPT_cat_idx', 'center_sep', 'cluster_id'])

        # Sort the table by the catalog index.
        match_info.sort(['SPT_cat_idx', 'center_sep'])

        # Use Astropy's unique function to remove the duplicate rows. Because the table rows will be subsorted by the
        # separation column we only need to keep the first incidence of the catalog index as our best match.
        match_info = unique(match_info, keys='SPT_cat_idx', keep='first')

        # Remove the duplicate clusters
        duplicate_clusters = set(match_info['cluster_id']).symmetric_difference(self._catalog_dictionary.keys())
        for cluster_id in duplicate_clusters:
            self._catalog_dictionary.pop(cluster_id, None)

    def coverage_mask(self, ch1_min_cov, ch2_min_cov):
        """
        Generates a binary good pixel map using the coverage maps.

        Creates a new fits image where every pixel has values of `1` if the coverage values in both IRAC bands are above
        the given thresholds or `0` otherwise.

        Parameters
        ----------
        ch1_min_cov : int or float
            Minimum coverage value allowed in 3.6 um band.
        ch2_min_cov : int or float
            Minimum coverage value allowed in 4.5 um band.

        Notes
        -----
        This method writes fits images into the directory specified as `mask_dir` in initialization.

        """

        for cluster_id, cluster_info in self._catalog_dictionary.items():
            # Array element names
            irac_ch1_cov_path = cluster_info['ch1_cov_path']
            irac_ch2_cov_path = cluster_info['ch2_cov_path']

            # Read in the two coverage maps, also grabbing the header from the Ch1 map.
            irac_ch1_cover, header = fits.getdata(irac_ch1_cov_path, header=True, ignore_missing_end=True)
            irac_ch2_cover = fits.getdata(irac_ch2_cov_path, ignore_missing_end=True)

            # Create the mask by setting pixel value to 1 if the pixel has coverage above the minimum coverage value in
            # both IRAC bands.
            combined_cov = np.logical_and((irac_ch1_cover >= ch1_min_cov), (irac_ch2_cover >= ch2_min_cov)).astype(int)

            # For naming, we will use the official SPT ID name for the cluster
            spt_id = self._spt_catalog['SPT_ID'][cluster_info['SPT_cat_idx']]

            # Write out the coverage mask.
            mask_pathname = f'{self._mask_dir}/{spt_id}_cov_mask{ch1_min_cov}_{ch2_min_cov}.fits'
            combined_cov_hdu = fits.PrimaryHDU(combined_cov, header=header)
            combined_cov_hdu.writeto(mask_pathname, overwrite=True)

            # Append the new coverage mask path name and both the catalog and the masking flag from cluster_info
            # to the new output list.
            cluster_info['cov_mask_path'] = mask_pathname

    def object_mask(self):
        """
        Performs additional masking on the good pixel maps for requested clusters.

        If a cluster has a DS9 regions file present in the directory specified as `region_file_dir` in initialization
        then we will read in the file, and set pixels within the shapes present in the file to `0`.

        Notes
        -----
        The allowable shapes in the regions file are `circle`, `box`, and `ellipse`. An unexpected shape will raise a
        KeyError.

        Raises
        ------
        KeyError
            An error is raised if the shape in the regions file is not one of the allowable shapes.

        """

        # Region file directory files
        if isinstance(self._region_file_dir, list):
            reg_files = {self._keyfunct(f): f for f in chain.from_iterable(glob.glob(f'{reg_dir}/*.reg')
                                                                           for reg_dir in self._region_file_dir)}
        else:
            reg_files = {self._keyfunct(f): f for f in glob.glob(f'{self._region_file_dir}/*.reg')}

        # Select out the IDs of the clusters needing additional masking
        clusters_to_mask = set(reg_files).intersection(self._catalog_dictionary)

        for cluster_id in clusters_to_mask:
            cluster_info = self._catalog_dictionary.get(cluster_id, None)
            region_file = reg_files.get(cluster_id, None)

            pixel_map_path = cluster_info['cov_mask_path']

            # Read in the coverage mask data and header.
            good_pix_mask, header = fits.getdata(pixel_map_path, header=True, ignore_missing_end=True, memmap=False)

            # Read in the WCS from the coverage mask we made earlier.
            w = WCS(header)

            # Get the pixel scale from the WCS
            pix_scale = w.proj_plane_pixel_scales()[0]

            # Open the regions file and get the lines containing the shapes.
            with open(region_file, 'r') as region:
                objs = [ln.strip() for ln in region
                        if ln.startswith('circle') or ln.startswith('box') or ln.startswith('ellipse')]

            # For each shape extract the defining parameters and define a path region.
            shapes_to_mask = []
            for mask in objs:

                # For circle shapes we need the center coordinate and the radius.
                if mask.startswith('circle'):
                    # Parameters of circle shape are as follows:
                    # params[0] : region center RA in degrees
                    # params[1] : region center Dec in degrees
                    # params[2] : region radius in arcseconds
                    params = np.array(re.findall(r'[+-]?\d+(?:\.\d+)?', mask), dtype=np.float64)

                    # Convert the center coordinates into pixel system.
                    # "0" is to correct the pixel coordinates to the right origin for the data.
                    cent_xy = w.wcs_world2pix(params[0], params[1], 0)

                    # Generate the mask shape.
                    shape = Path.circle(center=cent_xy, radius=params[2] / pix_scale)

                # For the box we'll need...
                elif mask.startswith('box'):
                    # Parameters for box shape are as follows:
                    # params[0] : region center RA in degrees
                    # params[1] : region center Dec in degrees
                    # params[2] : region width in arcseconds
                    # params[3] : region height in arcseconds
                    # params[4] : rotation of region about the center in degrees
                    params = np.array(re.findall(r'[+-]?\d+(?:\.\d+)?', mask), dtype=np.float64)

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

                elif mask.startswith('ellipse'):
                    # Parameters for ellipse shape are as follows
                    # params[0] : region center RA in degrees
                    # params[1] : region center Dec in degrees
                    # params[2] : region semi-major axis in arcseconds
                    # params[3] : region semi-minor axis in arcseconds
                    # params[4] : rotation of region about the center in degrees
                    # Note: For consistency, the semi-major axis should always be aligned along the horizontal axis
                    # before rotation
                    params = np.array(re.findall(r'[+-]?\d+(?:\.\d+)?', mask), dtype=np.float64)

                    # Convert the center coordinates into pixel system
                    cent_xy = w.wcs_world2pix(params[0], params[1], 0)

                    # Generate the mask shape
                    shape = Ellipse(cent_xy, width=params[2] / pix_scale, height=params[3] / pix_scale, angle=params[4])
                    shape = shape.get_path()

                # Return error if mask shape isn't known.
                else:
                    raise KeyError(
                        f'Mask shape is unknown, please check the region file of cluster: {region_file} {mask}')

                shapes_to_mask.append(shape)

            # Check if the pixel values are within the shape we defined earlier.
            # If true, set the pixel value to 0.
            pts = list(product(range(w.pixel_shape[0]), range(w.pixel_shape[1])))

            shape_masks = np.array(
                [shape.contains_points(pts).reshape(good_pix_mask.shape) for shape in shapes_to_mask])

            # Combine all the shape masks into a final object mask, inverting the boolean values so we can multiply
            # our mask with our existing good pixel mask
            total_obj_mask = ~np.logical_or.reduce(shape_masks)

            # Apply the object mask to the existing good pixel mask
            good_pix_mask *= total_obj_mask.astype(int)

            # Write the new mask to disk overwriting the old mask.
            new_mask_hdu = fits.PrimaryHDU(good_pix_mask, header=header)
            new_mask_hdu.writeto(pixel_map_path, overwrite=True)

    def j_band_abs_mag(self):
        """
        Computes the J-band absolute magnitudes for use in the Assef et al. (2011) luminosity function.

        To compute the J-band absolute magnitudes, we will use the observed apparent 3.6 um magnitude and assume a
        Polleta  QSO2 SED for all objects to K-correct to the absolute FLAMINGOS J-band magnitude.
        """

        # Load in the IRAC 3.6 um filter as the observed filter
        irac_36 = SpectralElement.from_file(self._irac_filter, wave_unit=u.um)
        flamingos_j = SpectralElement.from_file(self._j_band_filter, wave_unit=u.nm)

        # We will use the official IRAC 3.6 um zero-point flux
        irac_36_zp = 280.9 * u.Jy

        for cluster_id, cluster_info in self._catalog_dictionary.items():
            # Retrieve the cluster redshift from the SPT catalog
            catalog_idx = cluster_info['SPT_cat_idx']
            cluster_z = self._spt_catalog['REDSHIFT'][catalog_idx]

            # Get the 3.6 um apparent magnitudes from the catalog
            se_catalog = cluster_info['catalog']
            irac_36_mag = se_catalog['I1_MAG_APER4']

            # Given the observed IRAC 3.6 um photometry, compute the rest-frame J-band absolute (Vega) magnitude.
            j_abs_mag = k_corr_abs_mag(apparent_mag=irac_36_mag, z=cluster_z, f_lambda_sed=self._sed,
                                       zero_pt_obs_band=irac_36_zp, zero_pt_em_band='vega', obs_filter=irac_36,
                                       em_filter=flamingos_j, cosmo=self._cosmo)

            # Store the J-band absolute magnitude in the catalog and update the data structure
            se_catalog['J_ABS_MAG'] = j_abs_mag
            cluster_info['catalog'] = se_catalog

    def object_selection(self, ch1_bright_mag, ch2_bright_mag, selection_band_faint_mag, selection_band='I2_MAG_APER4'):
        """
        Selects the objects in the clusters as AGN subject to a color cut.

        Reads in the SExtractor catalogs and performs all necessary cuts to select the AGN in the cluster.
         - First, a cut is made on the SExtractor flag requiring an extraction flag of `< 4`.
         - A magnitude cut is applied on the faint-end of the selection band. This is determined such that the
           completeness limit in the selection band is kept above 80%.
         - A magnitude cut is applied to the bright-end of both bands to remain under the saturation limit.
         - Finally, the surviving objects' positions are checked against the good pixel map to ensure that the object
           lies on an acceptable location.
         - Further selection refinement is handled by :method:`selection_membership`.

        Parameters
        ----------
        ch1_bright_mag : float
            Bright-end magnitude threshold for 3.6 um band.
        ch2_bright_mag : float
            Bright-end magnitude threshold for 4.5 um band.
        selection_band_faint_mag : float
            Faint-end magnitude threshold for the specified selection band.
        selection_band : str, optional
            Column name in SExtractor catalog specifying the selection band to use. Defaults to the 4.5 um, 4" aperture
            magnitude photometry.

        """

        clusters_to_remove = []
        for cluster_id, cluster_info in self._catalog_dictionary.items():
            # Read in the catalog
            se_catalog = Table.read(cluster_info['se_cat_path'], format='ascii')

            # Add the mask name to the catalog. Extracting only the system agnostic portion of the path
            se_catalog['MASK_NAME'] = re.search(r'Data_Repository/.*?\Z', cluster_info['cov_mask_path']).group(0)

            # Preform SExtractor Flag cut. A value of under 4 should indicate the object was extracted well.
            se_catalog = se_catalog[se_catalog['FLAGS'] < 4]

            # Preform a faint-end magnitude cut in selection band.
            se_catalog = se_catalog[se_catalog[selection_band] <= selection_band_faint_mag]

            # Preform bright-end cuts
            # Limits from Eisenhardt+04 for ch1 = 10.0 and ch2 = 9.8
            se_catalog = se_catalog[se_catalog['I1_MAG_APER4'] > ch1_bright_mag]  # [3.6] saturation limit
            se_catalog = se_catalog[se_catalog['I2_MAG_APER4'] > ch2_bright_mag]  # [4.5] saturation limit

            # For the mask cut we need to check the pixel value for each object's centroid.
            # Read in the mask file
            mask, header = fits.getdata(cluster_info['cov_mask_path'], header=True)

            # Recast the mask image as a boolean array so we can use it as a check on the catalog entries
            mask = mask.astype(bool)

            # Read in the WCS from the mask
            w = WCS(header)

            # Get the objects pixel coordinates
            xy_data = np.array(w.wcs_world2pix(se_catalog['ALPHA_J2000'], se_catalog['DELTA_J2000'], 0))

            # Floor the values and cast as integers so we have the pixel indices into the mask
            xy_pix_idxs = np.floor(xy_data).astype(int)

            # Filter the catalog according to the boolean value in the mask at the objects' locations.
            se_catalog = se_catalog[mask[xy_pix_idxs[1], xy_pix_idxs[0]]]

            # If we have completely exhausted the cluster of any object, we should mark it for removal otherwise add it
            # to the data structure
            if se_catalog:
                cluster_info['catalog'] = se_catalog
            else:
                clusters_to_remove.append(cluster_id)

        # Remove any cluster that has no objects surviving our selection cuts
        for cluster_id in clusters_to_remove:
            self._catalog_dictionary.pop(cluster_id, None)

    def selection_membership(self, color_threshold=None):
        """
        Calculates fuzzy degree of membership of each object to be an AGN.

        Computes the AGN selection membership for each object in the sample by first incorporating the color error to
        remove the Eddington bias---where we are likely to have truly blue objects scatter into our sample. We then
        calculate a fuzzy degree of membership for the AGN sample.

        Parameters
        ----------
        color_threshold : float, optional
            Manually specify a [3.6] - [4.5] color threshold to be used. If given, the selection membership function
            will not use any catalog redshift information to determine the color threshold to be used. Instead, a flat
            color threshold will be applied to the entire catalog sample.

        Notes
        -----
        The selection membership acts as effectively a degree of belief that an object should be included in a sample of
        AGN. It is computed as an alpha-cut on a fuzzy set membership function given by,

        .. math::
            \\mu_{\\mathrm{AGN}} = \\frac{\\int_{\\alpha}^{\\infty} \\mathcal{N}(\\mathcal{C}_{12},
            \\delta\\mathcal{C}_{12}) \\frac{dN}{d(\\mathcal{C}_{12})} d\\mathcal{C}_{12}}{\\int_{-\infty}^{\infty}
            \\mathcal{N}(\\mathcal{C}_{12}, \\delta\\mathcal{C}_{12}) \\frac{dN}{d(\\mathcal{C}_{12})}
            d\\mathcal{C}_{12}

        where :math:`\\mathcal{C}_{12}` and :math:`\\delta\\mathcal{C}_{12}` are the [3.6] - [4.5] color and color
        errors respectively of each object, :math:`\\frac{dN}{d\\mathcal{C}_{12}}` is the standard field (SDWFS)
        number--color distribution, and :math:`\\alpha` is either a redshift-dependent sample purity-based color
        threshold or a flat color threshold depending on the user input.
        """

        # Read in the number count distribution file
        with open(self._field_number_dist, 'r') as f:
            field_number_distribution = json.load(f)
        field_number_counts = field_number_distribution['normalized_number_counts']
        color_bins = field_number_distribution['color_bins']
        color_bin_min, color_bin_max = np.min(color_bins), np.max(color_bins)

        # Create an interpolation of our number count distribution
        color_probability_distribution = interp1d(color_bins, field_number_counts)

        color_redshift_threshold_function = None
        if color_threshold is None:
            # Read in the purity color-redshift threshold file
            with open(self._purity_color_funct, 'r') as f:
                color_threshold_data = json.load(f)
            color_thresholds = color_threshold_data['purity_90_colors']
            redshift_bins = color_threshold_data['redshift_bins']

            # Create a step function interpolation of the color-redshift function
            color_redshift_threshold_function = interp1d(redshift_bins, color_thresholds, kind='previous')

        clusters_to_remove = []
        for cluster_id, cluster_info in self._catalog_dictionary.items():
            # Retrieve the cluster redshift from the SPT catalog
            catalog_idx = cluster_info['SPT_cat_idx']
            cluster_z = self._spt_catalog['REDSHIFT'][catalog_idx]

            if color_threshold is None:
                # Set the color threshold according to the cluster's redshift
                ch1_ch2_color_cut = color_redshift_threshold_function(cluster_z)
            else:
                ch1_ch2_color_cut = color_threshold

            # Get the photometric catalog for the cluster
            se_catalog = cluster_info['catalog']

            # Compute the color and color errors for each object
            I1_I2_color = se_catalog['I1_MAG_APER4'] - se_catalog['I2_MAG_APER4']
            I1_I2_color_err = np.sqrt((2.5 * se_catalog['I1_FLUXERR_APER4'] /
                                       (se_catalog['I1_FLUX_APER4'] * np.log(10))) ** 2 +
                                      (2.5 * se_catalog['I2_FLUXERR_APER4'] /
                                       (se_catalog['I2_FLUX_APER4'] * np.log(10))) ** 2)

            # Convolve the error distribution for each object with the overall number count distribution
            def object_integrand(x):
                return norm(loc=I1_I2_color, scale=I1_I2_color_err).pdf(x) * color_probability_distribution(x)

            selection_membership_numer = quad_vec(object_integrand, a=ch1_ch2_color_cut, b=color_bin_max)[0]
            selection_membership_denom = quad_vec(object_integrand, a=color_bin_min, b=color_bin_max)[0]
            selection_membership = selection_membership_numer / selection_membership_denom

            # Store the degree of membership into the catalog
            se_catalog['SELECTION_MEMBERSHIP'] = selection_membership

            # As objects with degrees of membership of 0 do not contribute to the sample, we can safely remove them.
            se_catalog = se_catalog[se_catalog['SELECTION_MEMBERSHIP'] > 0]

            # If we have exhausted all objects from the catalog mark the cluster for removal otherwise update the
            # photometric catalog in our database
            if se_catalog:
                cluster_info['catalog'] = se_catalog
            else:
                clusters_to_remove.append(cluster_id)

        # Remove any cluster that has no objects surviving our selection cuts
        for cluster_id in clusters_to_remove:
            self._catalog_dictionary.pop(cluster_id, None)

    def catalog_merge(self, catalog_cols=None):
        """
        Merges the SExtractor photometry catalog with information about the cluster in the official SPT catalog.


        Parameters
        ----------
        catalog_cols : list_like, optional
            A list of column names to include from the official SPT catalog. If not specified, only the official SPT ID
            and the SZ center RA and Dec will be added to the photometric catalog.

        """

        for cluster_info in self._catalog_dictionary.values():
            # Array element names
            catalog_idx = cluster_info['SPT_cat_idx']
            se_catalog = cluster_info['catalog']

            # Replace the existing SPT_ID in the SExtractor catalog with the official cluster ID.
            # se_catalog.columns[0].name = 'SPT_ID'
            # del se_catalog['SPT_ID']

            # Then replace the column values with the official ID.
            se_catalog['SPT_ID'] = self._spt_catalog['SPT_ID'][catalog_idx]

            # Add the SZ center coordinates to the catalog
            se_catalog['SZ_RA'] = self._spt_catalog['RA'][catalog_idx]
            se_catalog['SZ_DEC'] = self._spt_catalog['DEC'][catalog_idx]

            # For all requested columns from the master catalog add the value to all columns in the SExtractor catalog.
            if catalog_cols is not None:
                for col_name in catalog_cols:
                    se_catalog[col_name] = self._spt_catalog[col_name][catalog_idx]

            cluster_info['catalog'] = se_catalog

    def object_separations(self):
        """
        Calculates the separations of each object relative to the SZ center.

        Finds both the angular separations and physical separations relative to the cluster's :math:`r_{500}` radius.

        """

        for cluster_info in self._catalog_dictionary.values():
            catalog = cluster_info['catalog']

            # Create SkyCoord objects for all objects in the catalog as well as the SZ center
            object_coords = SkyCoord(catalog['ALPHA_J2000'], catalog['DELTA_J2000'], unit=u.degree)
            sz_center = SkyCoord(catalog['SZ_RA'][0], catalog['SZ_DEC'][0], unit=u.degree)

            # Calculate the angular separations between the objects and the SZ center in arcminutes
            separations_arcmin = object_coords.separation(sz_center).to(u.arcmin)

            # Compute the r500 radius for the cluster
            r500 = (3 * catalog['M500'][0] * u.Msun /
                    (4 * np.pi * 500 * self._cosmo.critical_density(catalog['REDSHIFT'][0]).to(
                        u.Msun / u.Mpc ** 3))) ** (1 / 3)

            # Convert the angular separations into physical separations relative to the cluster's r500 radius
            separations_r500 = (separations_arcmin / r500
                                * self._cosmo.kpc_proper_per_arcmin(catalog['REDSHIFT'][0]).to(u.Mpc / u.arcmin))

            # Add our new columns to the catalog
            catalog['R500'] = r500
            catalog['RADIAL_SEP_R500'] = separations_r500
            catalog['RADIAL_SEP_ARCMIN'] = separations_arcmin

            # Update the catalog in the data structure
            cluster_info['catalog'] = catalog

    def completeness_value(self, selection_band='I2_MAG_APER4'):
        """
        Adds completeness simulation data to the catalog.

        Takes the completeness curve values for the cluster, interpolates a function between the discrete values, then
        queries the specified magnitude of the selected objects in the SExtractor catalog and adds two columns: The
        completeness value of that object at its magnitude and the completeness correction value
        `(1/[completeness value])`.

        Parameters
        ----------
        selection_band : str, optional
            Column name in SExtractor catalog specifying the selection band to use. Must be the same band used in
            :method:`object_selection`. Defaults to the 4.5 um, 4" aperture magnitude photometry.

        """

        # Load in the completeness simulation data from the file
        if isinstance(self._completeness_results, list):
            json_dicts = []
            for comp_results in self._completeness_results:
                with open(comp_results, 'r') as f:
                    json_dicts.append(json.load(f))
            completeness_dict = dict(ChainMap(*json_dicts))
        else:
            with open(self._completeness_results, 'r') as f:
                completeness_dict = json.load(f)

        for cluster_id, cluster_info in self._catalog_dictionary.items():
            # Array element names
            se_catalog = cluster_info['catalog']

            # Select the correct entry in the dictionary corresponding to our cluster.
            completeness_data = completeness_dict[cluster_id]

            # Also grab the magnitude bins used to create the completeness data (removing the last entry so we can
            # broadcast our arrays correctly)
            mag_bins = completeness_dict['magnitude_bins'][:-1]

            # Interpolate the completeness data into a functional form using linear interpolation
            completeness_funct = interp1d(mag_bins, completeness_data, kind='linear')

            # For the objects' magnitude specified by `selection_band` query the completeness function to find the
            # completeness value.
            completeness_values = completeness_funct(se_catalog[selection_band])

            # The completeness correction values are defined as 1/[completeness value]
            completeness_corrections = 1 / completeness_values

            # Add the completeness values and corrections to the SExtractor catalog.
            se_catalog['COMPLETENESS_VALUE'] = completeness_values
            se_catalog['COMPLETENESS_CORRECTION'] = completeness_corrections

            cluster_info['catalog'] = se_catalog

    def final_catalogs(self, filename=None, catalog_cols=None):
        """
        Collates all catalogs into one table then writes the catalog to disk.

        Parameters
        ----------
        filename : str, optional
            File name of the output catalog file. If not specified, the function will return the final catalog instead
            of writing to disk.
        catalog_cols : list_like, optional
            List of column names in the catalog which we wish to keep in our output file. If not specified, all columns
            present in the catalog are kept in the output catalog.

        Returns
        -------
        final_catalog : Table or None
            The final catalog of the survey AGN. If `filename` is specified, the function will write the catalog to disk
            instead of returning.
        """

        final_catalog = vstack([cluster_info['catalog'] for cluster_info in self._catalog_dictionary.values()])

        # If we request to keep only certain columns in our output
        if catalog_cols is not None:
            final_catalog.keep_columns(catalog_cols)

        if filename is None:
            return final_catalog
        else:
            if filename.endswith('.cat'):
                final_catalog.write(filename, format='ascii', overwrite=True)
            else:
                final_catalog.write(filename, overwrite=True)

    def run_selection(self, included_clusters, excluded_clusters, max_image_catalog_sep, ch1_min_cov, ch2_min_cov,
                      ch1_bright_mag, ch2_bright_mag, selection_band_faint_mag, spt_colnames,
                      output_name, output_colnames, ch1_ch2_color=None):
        """
        Executes full selection pipeline using default values.

        Parameters
        ----------
        included_clusters : list_like
            List of clusters to
        excluded_clusters : list_like
            Excluded clusters for  :method:`file_pairing`.
        max_image_catalog_sep : astropy.quantity
            Maximum separation for :method:`image_to_catalog_match`.
        ch1_min_cov : int or float
            Minimum 3.6 um coverage for :method:`coverage_mask`.
        ch2_min_cov : int or float
            Minimum 4.5 um coverage for :method:`coverage_mask`.
        ch1_bright_mag : float
            Bright-end 3.6 um magnitude for :method:`object_selection`
        ch2_bright_mag : float
            Bright-end 4.5 um magnitude for :method:`object_selection`
        selection_band_faint_mag : float
            Faint-end selection band magnitude for :method:`object_selection`
        spt_colnames : list_like
            Column names in SPT catalog for :method:`catalog_merge`
        output_name : str or None
            File name of output catalog for :method:`final_catalogs`
        output_colnames : list_like
            Column names to be kept in output catalog for :method:`final_catalogs`
        ch1_ch2_color : float or None
            Flat [3.6] -[4.5] color threshold to use when computing selection memberships in :method:`purify_selection`.
            If `None` then the purity based, redshift dependent color threshold relation will be used.

        Returns
        -------
        final_catalog : Table or None
            The pipeline will return the final catalog if `output_name` is given as None. Otherwise, the pipeline will
            write the final catalog to disk automatically.

        """
        self.file_pairing(include=included_clusters, exclude=excluded_clusters)
        self.image_to_catalog_match(max_image_catalog_sep=max_image_catalog_sep)
        self.coverage_mask(ch1_min_cov=ch1_min_cov, ch2_min_cov=ch2_min_cov)
        self.object_mask()
        self.object_selection(ch1_bright_mag=ch1_bright_mag, ch2_bright_mag=ch2_bright_mag,
                              selection_band_faint_mag=selection_band_faint_mag)
        self.selection_membership(color_threshold=ch1_ch2_color)
        self.j_band_abs_mag()
        self.catalog_merge(catalog_cols=spt_colnames)
        self.object_separations()
        self.completeness_value()
        final_catalog = self.final_catalogs(filename=output_name, catalog_cols=output_colnames)
        if final_catalog is not None:
            return final_catalog

    @classmethod
    def _keyfunct(cls, f):
        """Generate a key function that isolates the cluster ID for sorting and grouping"""
        return re.search(r'SPT-CLJ\d+[+-]?\d+[-\d+]?', f).group(0)


class SelectSDWFS(SelectIRAGN):
    def __init__(self, sextractor_cat_dir, irac_image_dir, region_file_dir, mask_dir, sdwfs_master_catalog,
                 completeness_file, field_number_dist_file, sed, irac_filter, j_band_filter):
        super().__init__(sextractor_cat_dir=sextractor_cat_dir,
                         irac_image_dir=irac_image_dir,
                         region_file_dir=region_file_dir,
                         mask_dir=mask_dir,
                         spt_catalog=sdwfs_master_catalog,
                         completeness_file=completeness_file,
                         field_number_dist_file=field_number_dist_file,
                         purity_color_threshold_file=None,
                         sed=sed,
                         irac_filter=irac_filter,
                         j_band_filter=j_band_filter)

    def object_separations(self):

        for cutout_info in self._catalog_dictionary.values():
            catalog = cutout_info['catalog']

            # Create SkyCoord objects for all objects in the catalog as well as the image center
            object_coords = SkyCoord(catalog['ALPHA_J2000'], catalog['DELTA_J2000'], unit=u.deg)
            center_coord = SkyCoord(catalog['SZ_RA'][0], catalog['SZ_DEC'][0], unit=u.deg)

            # Calculate the angular separations between the objects and the image center in arcminutes
            separations_arcmin = object_coords.separation(center_coord).to(u.arcmin)

            # Add our new column to the catalog
            catalog['RADIAL_SEP_ARCMIN'] = separations_arcmin

            # Update the catalog in the data structure
            cutout_info['catalog'] = catalog

    def j_band_abs_mag(self):

        # Load in the IRAC 3.6 um filter as the observed filter
        irac_36 = SpectralElement.from_file(self._irac_filter, wave_unit=u.um)
        flamingos_j = SpectralElement.from_file(self._j_band_filter, wave_unit=u.nm)

        # We will use the official IRAC 3.6 um zero-point flux
        irac_36_zp = 280.9 * u.Jy

        for cutout_id, cutout_info in self._catalog_dictionary.items():
            # Get the 3.6 um apparent magnitudes and photometric redshifts from the catalog
            se_catalog = cutout_info['catalog']
            irac_36_mag = se_catalog['I1_MAG_APER4']
            galaxy_z = se_catalog['REDSHIFT']

            # Given the observed IRAC 3.6 um photometry, compute the rest-frame J-band absolute (Vega) magnitude.
            j_abs_mag = k_corr_abs_mag(apparent_mag=irac_36_mag, z=galaxy_z, f_lambda_sed=self._sed,
                                       zero_pt_obs_band=irac_36_zp, zero_pt_em_band='vega', obs_filter=irac_36,
                                       em_filter=flamingos_j, cosmo=self._cosmo)

            # Store the J-band absolute magnitude in the catalog and update the data structure
            se_catalog['J_ABS_MAG'] = j_abs_mag
            cutout_info['catalog'] = se_catalog

    @classmethod
    def _keyfunct(cls, f):
        return re.search(r'SDWFS_cutout_\d+', f).group(0)
