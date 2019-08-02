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
from itertools import groupby, product

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table, unique, vstack
from astropy.utils.exceptions import AstropyWarning  # For suppressing the astropy warnings.
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from scipy.interpolate import interp1d

# Suppress Astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)


class SelectIRAGN:
    """
    Pipeline to select for IR-Bright AGN in the SPT cluster surveys.

    Parameters
    ----------
    sextractor_cat_dir : str
        Directory for the SExtractor photometric catalogs.
    irac_image_dir : str
        Directory for the IRAC 3.6 um and 4.5 um science and coverage images.
    region_file_dir : str
        Directory for DS9 regions files describing areas to mask manually.
    mask_dir : str
        Directory to write good pixel masks into
    spt_catalog : astropy table_like
        Official SPT cluster catalog.
    completeness_file : str
        File name of the completeness simulation results.
    """

    def __init__(self, sextractor_cat_dir, irac_image_dir, region_file_dir, mask_dir, spt_catalog, completeness_file):

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

    def file_pairing(self, exclude=None):
        """
        Collates the different files for each cluster into a dictionary.

        This will also remove any clusters we request and perform a check that all clusters have the required files.

        Parameters
        ----------
        exclude : list_like, optional
            A list containing the names of the clusters we wish to remove from our sample.

        Raises
        ------
        UserWarning
            A warning is printed if a cluster is missing any necessary files. These clusters will be automatically
            removed from the sample.

        """

        # List the file names for both the images and the catalogs
        image_files = glob.glob(self._irac_image_dir + '/*.fits')
        cat_files = glob.glob(self._sextractor_cat_dir + '/*.cat')

        # Combine and sort both file lists
        cat_image_files = sorted(cat_files + image_files, key=self.__keyfunct)

        # Group the file names together
        self._catalog_dictionary = {cluster_id: list(files)
                                    for cluster_id, files in groupby(cat_image_files, key=self.__keyfunct)}

        # If we want to exclude some clusters manually we can remove them now
        if exclude is not None:
            for cluster_id in exclude:
                self._catalog_dictionary.pop(cluster_id, None)

        # Sort the files into a dictionary according to the type of file
        for cluster_id, files in self._catalog_dictionary.items():
            self._catalog_dictionary[cluster_id] = {}
            for f in files:
                if f.endswith('.cat'):
                    self._catalog_dictionary[cluster_id]['sex_cat_path'] = f
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
            file_keys = {'ch1_sci_path', 'ch1_cov_path', 'ch2_sci_path', 'ch2_cov_path', 'sex_cat_path'}
            try:
                assert file_keys == cluster_files.keys()
            except AssertionError:
                message = 'Cluster {id} is missing files {k}'.format(id=cluster_id, k=file_keys - cluster_files.keys())
                warnings.warn(message)
                problem_clusters.append(cluster_id)

        # For now, remove the clusters missing files
        for cluster_id in problem_clusters:
            self._catalog_dictionary.pop(cluster_id, None)

    def image_to_catalog_match(self, max_image_catalog_sep):
        """
        Matches the science images to the official SPT catalog.

        Uses the reference pixel coordinate (CRVAL1/2) of the 3.6 um science image to match against the SZ center of the
        clusters in the official SPT catalog. Clusters are kept only if their images match an SZ center within the given
        maximum separation. If multiple images match to the same SZ center within our max separation then only the
        closest match is kept.

        Parameters
        ----------
        max_image_catalog_sep : astropy.quantity
            Maximum separation allowed between the image reference pixel and the SZ center matched in the official SPT
            catalog.

        """

        catalog = self._spt_catalog

        # Create astropy skycoord object of the SZ centers.
        sz_centers = SkyCoord(catalog['RA'], catalog['DEC'], unit=u.degree)

        for cluster in self._catalog_dictionary.values():
            # Get the RA and Dec of the reference pixel in the image.
            w = WCS(cluster['ch1_sci_path'])
            crval = w.wcs.crval

            # Create astropy skycoord object for the reference pixel of the image.
            img_coord = SkyCoord(crval[0], crval[1], unit=w.wcs.cunit)

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
        match_info = Table(np.array([[cluster['SPT_cat_idx'], cluster['center_sep'], cluster_id]
                                     for cluster_id, cluster in self._catalog_dictionary.items()]),
                           names=['SPT_cat_idx', 'center_sep', 'cluster_id'])

        # Sort the table by the Bleem index.
        match_info.sort(['SPT_cat_idx', 'center_sep'])

        # Use Astropy's unique function to remove the duplicate rows. Because the table rows will be subsorted by the
        # separation column we only need to keep the first incidence of the Bleem index as our best match.
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

        for cluster_info in self._catalog_dictionary.values():
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
            mask_fname = '{cluster_id}_cov_mask{ch1_cov}_{ch2_cov}.fits'.format(cluster_id=spt_id,
                                                                                ch1_cov=ch1_min_cov,
                                                                                ch2_cov=ch2_min_cov)
            mask_pathname = self._mask_dir + '/' + mask_fname
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
        reg_files = {self.__keyfunct(f): f for f in glob.glob(self._region_file_dir + '/*.reg')}

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

            pix_scale = (w.pixel_scale_matrix[1, 1] * w.wcs.cunit[1]).to(u.arcsec).value

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

                # Return error if mask shape isn't known.
                else:
                    raise KeyError('Mask shape is unknown, please check the region file of cluster: {region} {mask}'
                                   .format(region=region_file, mask=mask))

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

    def object_selection(self, ch1_bright_mag, ch2_bright_mag, selection_band_faint_mag, ch1_ch2_color_cut,
                         selection_band='I2_MAG_APER4'):
        """
        Selects the objects in the clusters as AGN subject to a color cut.

        Reads in the SExtractor catalogs and performs all necessary cuts to select the AGN in the cluster.
         - First, a cut is made on the SExtractor flag requiring an extraction flag of `< 4`.
         - A magnitude cut is applied on the faint-end of the selection band. This is determined such that the
           completeness limit in the selection band is kept above 80%.
         - A magnitude cut is applied to the bright-end of both bands to remain under the saturation limit.
         - The [3.6] - [4.5] color cut is applied to select for red objects above which we define as IR-bright AGN.
         - Finally, the surviving objects' positions are checked against the good pixel map to ensure that the object
           lies on an acceptable location.

        Parameters
        ----------
        ch1_bright_mag : float
            Bright-end magnitude threshold for 3.6 um band.
        ch2_bright_mag : float
            Bright-end magnitude threshold for 4.5 um band.
        selection_band_faint_mag : float
            Faint-end magnitude threshold for the specified selection band.
        ch1_ch2_color_cut : float
            [3.6] - [4.5] color cut above which objects will be selected as AGN.
        selection_band : str, optional
            Column name in SExtractor catalog specifying the selection band to use. Defaults to the 4.5 um, 4" aperture
            magnitude photometry.

        """

        clusters_to_remove = []
        for cluster_id, cluster_info in self._catalog_dictionary.items():
            # Read in the catalog
            sex_catalog = Table.read(cluster_info['sex_cat_path'], format='ascii')

            # Preform SExtractor Flag cut. A value of under 4 should indicate the object was extracted well.
            sex_catalog = sex_catalog[sex_catalog['FLAGS'] < 4]

            # Preform a faint-end magnitude cut in selection band.
            sex_catalog = sex_catalog[sex_catalog[selection_band] <= selection_band_faint_mag]

            # Preform bright-end cuts
            # Limits from Eisenhardt+04 for ch1 = 10.0 and ch2 = 9.8
            sex_catalog = sex_catalog[sex_catalog['I1_MAG_APER4'] > ch1_bright_mag]  # [3.6] saturation limit
            sex_catalog = sex_catalog[sex_catalog['I2_MAG_APER4'] > ch2_bright_mag]  # [4.5] saturation limit

            # Calculate the IRAC Ch1 - Ch2 color (4" apertures) and preform the color cut
            sex_catalog = sex_catalog[sex_catalog['I1_MAG_APER4'] - sex_catalog['I2_MAG_APER4'] >= ch1_ch2_color_cut]

            # For the mask cut we need to check the pixel value for each object's centroid.
            # Read in the mask file
            mask, header = fits.getdata(cluster_info['cov_mask_path'], header=True)

            # Recast the mask image as a boolean array so we can use it as a check on the catalog entries
            mask = mask.astype(bool)

            # Read in the WCS from the mask
            w = WCS(header)

            # Get the objects pixel coordinates
            xy_data = np.array(w.wcs_world2pix(sex_catalog['ALPHA_J2000'], sex_catalog['DELTA_J2000'], 0))

            # Floor the values and cast as integers so we have the pixel indices into the mask
            xy_pix_idxs = np.floor(xy_data).astype(int)

            # Filter the catalog according to the boolean value in the mask at the objects' locations.
            sex_catalog = sex_catalog[mask[xy_pix_idxs[1], xy_pix_idxs[0]]]

            # If we have completely exhausted the cluster of any object, we should mark it for removal otherwise add it
            # to the data structure
            if sex_catalog:
                cluster_info['catalog'] = sex_catalog
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
            sex_catalog = cluster_info['catalog']

            # Replace the existing SPT_ID in the SExtractor catalog with the official cluster ID.
            sex_catalog.columns[0].name = 'SPT_ID'
            del sex_catalog['SPT_ID']

            # Then replace the column values with the official ID.
            sex_catalog['SPT_ID'] = self._spt_catalog['SPT_ID'][catalog_idx]

            # Add the SZ center coordinates to the catalog
            sex_catalog['SZ_RA'] = self._spt_catalog['RA'][catalog_idx]
            sex_catalog['SZ_DEC'] = self._spt_catalog['DEC'][catalog_idx]

            # For all requested columns from the master catalog add the value to all columns in the SExtractor catalog.
            if catalog_cols is not None:
                for col_name in catalog_cols:
                    sex_catalog[col_name] = self._spt_catalog[col_name][catalog_idx]

            cluster_info['catalog'] = sex_catalog

    def object_separations(self):
        """
        Calculates the separations of each object relative to the SZ center.

        Finds both the angular separations and physical separations relative to the cluster's r500 radius.

        """

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        for cluster_info in self._catalog_dictionary.values():
            catalog = cluster_info['catalog']

            # Create SkyCoord objects for all objects in the catalog as well as the SZ center
            object_coords = SkyCoord(catalog['ALPHA_J2000'], catalog['DELTA_J2000'], unit=u.degree)
            sz_center = SkyCoord(catalog['SZ_RA'][0], catalog['SZ_DEC'][0], unit=u.degree)

            # Calculate the angular separations between the objects and the SZ center in arcminutes
            separations_arcmin = object_coords.separation(sz_center).to(u.arcmin)

            # Compute the r500 radius for the cluster
            r500 = (3 * catalog['M500'][0]*u.Msun /
                    (4*np.pi * 500 * cosmo.critical_density(catalog['REDSHIFT'][0]).to(u.Msun/u.Mpc**3)))**(1/3)

            # Convert the angular separations into physical separations relative to the cluster's r500 radius
            separations_r500 = (separations_arcmin / r500
                                * cosmo.kpc_proper_per_arcmin(catalog['REDSHIFT'][0]).to(u.Mpc/u.arcmin))

            # Add our new columns to the catalog
            catalog['R500'] = r500
            catalog['RADIAL_SEP_ARCMIN'] = separations_arcmin
            catalog['RADIAL_SEP_R500'] = separations_r500

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
        with open(self._completeness_results, 'r') as f:
            completeness_dict = json.load(f)

        for cluster_id, cluster_info in self._catalog_dictionary.items():
            # Array element names
            sex_catalog = cluster_info['catalog']

            # Select the correct entry in the dictionary corresponding to our cluster.
            completeness_data = completeness_dict[cluster_id]

            # Also grab the magnitude bins used to create the completeness data (removing the last entry so we can
            # broadcast our arrays correctly)
            mag_bins = completeness_dict['magnitude_bins'][:-1]

            # Interpolate the completeness data into a functional form using linear interpolation
            completeness_funct = interp1d(mag_bins, completeness_data, kind='linear')

            # For the objects' magnitude specified by `selection_band` query the completeness function to find the
            # completeness value.
            completeness_values = completeness_funct(sex_catalog[selection_band])

            # The completeness correction values are defined as 1/[completeness value]
            completeness_corrections = 1 / completeness_values

            # Add the completeness values and corrections to the SExtractor catalog.
            sex_catalog['COMPLETENESS_VALUE'] = completeness_values
            sex_catalog['COMPLETENESS_CORRECTION'] = completeness_corrections

            cluster_info.update({'catalog': sex_catalog})

    def final_catalogs(self, filename, catalog_cols=None):
        """
        Collates all catalogs into one table then writes the catalog to disk.

        Parameters
        ----------
        filename : str
            File name of the output catalog file.
        catalog_cols : list_like, optional
            List of column names in the catalog which we wish to keep in our output file. If not specified all columns
            present in the catalog are kept in the output file.

        """

        final_catalog = vstack([cluster_info['catalog'] for cluster_info in self._catalog_dictionary.values()])

        # If we request to keep only certain columns in our output
        if catalog_cols is not None:
            final_catalog.keep_columns(catalog_cols)

        final_cat_path = filename

        if filename.endswith(".cat"):
            final_catalog.write(final_cat_path, format='ascii', overwrite=True)
        else:
            final_catalog.write(final_cat_path, overwrite=True)

    def run_selection(self, excluded_clusters, max_separation, ch1_min_cov, ch2_min_cov, ch1_bright_mag, ch2_bright_mag,
                      selection_band_faint_mag, ch1_ch2_color, spt_colnames, output_name, output_colnames):
        """
        Executes full selection pipeline using default values.

        Parameters
        ----------
        excluded_clusters : list_like
            Excluded clusters for  :method:`file_pairing`.
        max_separation : astropy.quantity
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
        ch1_ch2_color : float
            [3.6] - [4.5] color cut for :method:`object_selection`
        spt_colnames : list_like
            Column names in SPT catalog for :method:`catalog_merge`
        output_name : str
            File name of output catalog for :method:`final_catalogs`
        output_colnames : list_like
            Column names to be kept in output catalog for :method:`final_catalogs`

        """
        self.file_pairing(exclude=excluded_clusters)
        self.image_to_catalog_match(max_image_catalog_sep=max_separation)
        self.coverage_mask(ch1_min_cov=ch1_min_cov, ch2_min_cov=ch2_min_cov)
        self.object_mask()
        self.object_selection(ch1_bright_mag=ch1_bright_mag, ch2_bright_mag=ch2_bright_mag,
                              selection_band_faint_mag=selection_band_faint_mag, ch1_ch2_color_cut=ch1_ch2_color)
        self.catalog_merge(catalog_cols=spt_colnames)
        self.object_separations()
        self.completeness_value()
        self.final_catalogs(filename=output_name, catalog_cols=output_colnames)

    @staticmethod
    def __keyfunct(f):
        """Generate a key function that isolates the cluster ID for sorting and grouping"""
        return re.search(r'SPT-CLJ\d+[+-]?\d+[-\d+]?', f).group(0)
