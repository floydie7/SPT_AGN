"""
SPT_Completeness_Simulation.py
Author: Benjamin Floyd

This script executes the functions written in Completeness_Simulation.Completeness_Simulation_Functions.py and generates
completeness curves for all the SPT clusters found in the Bleem survey.

The completeness simulation is preformed by using the IRAF tasks noao.artdata.starlists and noao.artdata.mkobjects.
The starlist task generates a list of random coordinates and magnitudes within the specified bounds where the
artificial stars will be placed. In all cases the spatial and magnitude values were drawn from a uniform distribution
with the spatial values bounded between a buffer of 5 pixels on either side in both axes. The magnitude values were
bounded between the specified minimum and maximum magnitudes which were iterated from the Vega magnitudes of 10.0
selection_band to 23.0 selection_band by 0.5 selection_band intervals. The mkobjects task created a new image based
on the existing input image with the addition of artificial stars with positions and magnitudes provided by the
output of the starlist task using the specified model and point spread function (psf) full-width at half-maximum in
arcseconds.

Source Extractor (SExtractor) was then run on the image creating a catalog of all detected objects. Both the input list
created by the starlist task and the SExtractor catalog were read in using the astropy.io.ascii.read function, the
aperture magnitudes were corrected, and then a catalog matching procedure was preformed between the two catalogs to find
the placed artificial stars. Only objects that had a maximum separation of 1 psf and had an input - output magnitude
difference less than the specified value were reported as being recovered. The number of recovered objects was recorded
for each magnitude bin and stored in a dictionary with the image name used as the key.
"""

from __future__ import print_function, division

import glob
import json
import re
import sys
from functools import partial
from time import time

import astropy.units as u
import numpy as np
from Completeness_Simulation_Functions import run_sex, make_stars
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack, vstack
from astropy.wcs import WCS
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from schwimmbad import MPIPool

# try:
#     import sys
#     assert sys.version_info() == (2, 7)
# except AssertionError:
#     raise AssertionError('This script must be ran on Python 2.7')

hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'


# hcc_prefix = '/Users/btfkwd/Documents/SPT_AGN/'


def completeness(image_name, bins, nsteps, fwhm, mag_zero, aper_corr, config_file, param_file, root_dir, mag_diff,
                 model, reg_file=False):
    """
    Generates a completeness simulation on one image.

    Parameters
    ----------
    reg_file : bool
        Flag to indicate if an additional region restriction is required for placement.
    image_name : str
        File name for the image to be processed.
    bins : list_like
        Array of magnitude bins.
    nsteps : int
        Number of star placements to preform per magnitude bin.
    fwhm : float
        The PSF full-width at half maximum to be used in the modeling of the stars.
    mag_zero : float
        The appropriate zero-point magnitude for the band.
    aper_corr : float
        The aperture correction for the measurement magnitude.
    config_file : str
        SExtractor configuration file.
    param_file : str
        SExtractor parameter file.
    root_dir : str
        Root working directory for simulation.
    mag_diff : float
        The magnitude difference tolerance between the "true" magnitudes and the measured magnitudes. Defaults to 0.2.
    model : str
        The model to use to generate the artificial stars. Can be 'gaussian' or 'moffat'. Defaults to 'gaussian'.


    Returns
    -------
    dict_rate : dict
        A dictionary the image name as the key and the recovery rates in a list as the value.
    """

    # Cluster Image ID
    image_id = cluster_image.search(image_name).group(0)
    cluster_id = cluster_image.search(image_name).group(1)

    # Image parameters
    output_image = root_dir + '/Images/{image_id}_stars.fits'.format(image_id=image_id)
    starlist_dir = root_dir + '/Starlists'

    # Altered image catalog
    alt_out_cat = root_dir + '/sex_catalogs/{image_id}_stars.cat'.format(image_id=image_id)

    if reg_file:
        region_file = hcc_prefix + 'Data/SPTPol/images/rereduced_images/regions/{image_id}.reg'.format(image_id=image_id)
        with open(region_file, 'r') as region:
            objs = [ln.strip() for ln in region
                    if ln.startswith('circle') or ln.startswith('box') or ln.startswith('ellipse')]
        w = WCS(image_name)
        try:
            assert w.pixel_scale_matrix[0, 1] == 0.
            pix_scale = (w.pixel_scale_matrix[1, 1] * w.wcs.cunit[1]).to(u.arcsec).value
        except AssertionError:
            cd = w.pixel_scale_matrix
            _, eig_vec = np.linalg.eig(cd)
            cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
            pix_scale = (cd_diag[1, 1] * w.wcs.cunit[1]).to(u.arcsec).value

        mask = objs[0]
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
        placement_bounds = shape.get_extents()
    else:
        placement_bounds = None

    # Store a copy of the input and output tables
    input_output = []

    # Store the recovery rates
    recovery_rate = []

    for j in range(len(bins) - 1):
        # Set magnitude range for bin
        min_mag = bins[j]
        max_mag = bins[j + 1]

        # Store the number of objects that were placed and the number of objects that were recovered.
        placed = []
        recovered = []

        for i in range(nsteps):
            # Now generate the image with artificial stars.
            make_stars(image_name, output_image, starlist_dir, model=model, fwhm=fwhm, mag_zero=mag_zero,
                       min_mag=min_mag, max_mag=max_mag, nstars=10, placement_bounds=placement_bounds)

            # Run SExtractor again on the altered image
            run_sex(output_image, alt_out_cat, mag_zero=mag_zero, seeing_fwhm=fwhm, sex_config=config_file,
                    param_file=param_file)

            # Read in both the starlist as a truth catalog and the altered image catalog
            true_stars = Table.read(
                '{starlist_dir}/{image_id}_stars.dat'.format(starlist_dir=starlist_dir, image_id=image_id),
                names=['x', 'y', 'selection_band'], format='ascii')
            altered_cat = Table.read(alt_out_cat, format='ascii.sextractor')

            # Preform aperture correction to the magnitude based on the published values in Ashby et al. 2009
            altered_cat['MAG_APER'] += aper_corr  # From mag_auto - mag_aper values.

            # Match the coordinates between the truth catalog and the SExtractor catalog.
            max_sep = fwhm * u.arcsec

            wcs = WCS(output_image)
            true_coord = SkyCoord.from_pixel(true_stars['x'], true_stars['y'], wcs=wcs)
            cat_coord = SkyCoord(altered_cat['ALPHA_J2000'], altered_cat['DELTA_J2000'], unit=u.degree)

            idx, sep, _ = true_coord.match_to_catalog_sky(cat_coord)

            # Only accept objects that are within the maximum separation.
            alt_cat_matched = altered_cat[idx][sep <= max_sep]
            true_stars_matched = true_stars[sep <= max_sep]

            # Require that the matched stars have magnitudes within 0.2 selection_band of the input magnitudes
            alt_cat_mag_matched = alt_cat_matched[
                np.abs(true_stars_matched['selection_band'] - alt_cat_matched['MAG_APER']) <= mag_diff]

            # Join the input and output catalogs
            true_stars_for_join = hstack([true_stars, Table([true_coord.ra, true_coord.dec],
                                                            names=['ALPHA_J2000', 'DELTA_J2000'])])
            alt_cat_for_join = Table(altered_cat[idx]['X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000',
                                                      'MAG_APER'])
            # Add separation to the altered catalog
            alt_cat_for_join['SEP'] = sep

            # alt_cat_for_join.mask = np.tile(~(sep <= max_sep), (len(alt_cat_for_join.columns), 1))
            merged_table = hstack([true_stars_for_join, alt_cat_for_join], table_names=['input', 'output'])
            input_output.append(merged_table)

            # Append the number of placed and recovered objects into their respective containers.
            placed.append(len(true_stars))
            recovered.append(len(alt_cat_mag_matched))

        if len(placed) != 0:
            recovery_rate.append(np.sum(recovered) / np.sum(placed))

    # Create a dictionary entry with the image name as the key and the recovery_rate list as the value.
    # This will allow for the rates to be identifiable to the image they were created from.
    dict_rate = {cluster_id: recovery_rate}

    # Merge all the input/output tables and write to file
    input_output_merged = vstack(input_output)
    # input_output_merged.filled(-99)
    input_output_merged.write(root_dir + '/Input_Output_catalogs/{image_id}_inout.fits'.format(image_id=image_id),
                              overwrite=True)

    return dict_rate


#  RegEx for finding the cluster/image ids
cluster_image = re.compile(r'I[12]_(SPT-CLJ\d+-\d+)')

# Magnitude bins
mag_bins = np.arange(10.0, 22.5, 0.5)

# Number of iterations per magnitude bin
placements_per_mag = 100

# Magnitude threshold
recovery_mag_thresh = 0.2

# Model type
psf_model = 'gaussian'

irac_data_sptsz = {1: {'psf_fwhm': 1.95, 'zeropt': 17.997, 'aper_corr': -0.1, },
                   2: {'psf_fwhm': 2.02, 'zeropt': 17.538, 'aper_corr': -0.11},
                   'config_file': hcc_prefix + 'Data/Comp_Sim/sex_configs/default.sex',
                   'param_file': hcc_prefix + 'Data/Comp_Sim/sex_configs/default.param',
                   'root_dir': hcc_prefix + 'Data/Comp_Sim',
                   'image_dir': hcc_prefix + 'Data/Images'}

irac_data_sptpol = {1: {'psf_fwhm': 1.95, 'zeropt': 18.789, 'aper_corr': -0.05},
                    2: {'psf_fwhm': 2.02, 'zeropt': 18.316, 'aper_corr': -0.11},
                    'config_file': hcc_prefix + 'Data/Comp_Sim/SPTpol/sex_configs/default.sex',
                    'param_file': hcc_prefix + 'Data/Comp_Sim/SPTpol/sex_configs/default.param',
                    'root_dir': hcc_prefix + 'Data/Comp_Sim/SPTpol',
                    'image_dir': hcc_prefix + 'Data/SPTPol/images/cluster_cutouts'}

irac_data_rereduced = {1: {'psf_fwhm': 1.95, 'zeropt': 18.789, 'aper_corr': -0.05},
                       2: {'psf_fwhm': 2.02, 'zeropt': 18.316, 'aper_corr': -0.11},
                       'config_file': hcc_prefix + 'Data/Comp_Sim/rereduced/sex_configs/default.sex',
                       'param_file': hcc_prefix + 'Data/Comp_Sim/rereduced/sex_configs/default.param',
                       'root_dir': hcc_prefix + 'Data/Comp_Sim/rereduced',
                       'image_dir': hcc_prefix + 'Data/SPTPol/images/rereduced_images'}

# Image directory
survey = 'rereduced'
# survey = 'pol'
if survey == 'SZ':
    # SPT-SZ/targeted IRAC SPTpol
    config_dict = irac_data_sptsz
elif survey == 'pol':
    # SSDF SPTpol
    config_dict = irac_data_sptpol
else:
    config_dict = irac_data_rereduced

# Channel 2 science images
ch2_images = glob.glob(config_dict['image_dir'] + '/I2*_mosaic.cutout.fits')

# Remove two clusters from the SPT-SZ input list as they have blank I2 images that cause problems
if survey == 'SZ':
    ch2_images = [image_name for image_name in ch2_images
                  if not any(cluster_name in image_name for cluster_name in ['SPT-CLJ2332-5358', 'SPT-CLJ2341-5726'])]

# Set up the completeness simulation functions partially evaluated except for the image list
I2_completeness = partial(completeness,
                          bins=mag_bins, nsteps=placements_per_mag,
                          fwhm=config_dict[2]['psf_fwhm'], mag_zero=config_dict[2]['zeropt'],
                          aper_corr=config_dict[2]['aper_corr'],
                          config_file=config_dict['config_file'],
                          param_file=config_dict['param_file'],
                          root_dir=config_dict['root_dir'],
                          mag_diff=recovery_mag_thresh, model=psf_model, reg_file=True)

# Record start time
start_time = time()

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    pool_results = pool.map(I2_completeness, ch2_images)

    if pool.is_master():
        completeness_results = {cluster_id: recovery_rates for result in pool_results
                                for cluster_id, recovery_rates in result.items()}
print('Simulation run time: {}'.format(time() - start_time))

# Add the magnitude values used to create the completeness rates.
completeness_results['magnitude_bins'] = list(mag_bins)

# Save results to disk
# results_filename = config_dict['root_dir'] + \
#                    '/Results/SPT{survey}_I2_results_{model}_fwhm{fwhm}_corr{corr}_mag{mag_diff}.json'.format(
#                        survey=survey, model=psf_model, fwhm=irac_data_sptpol[2]['psf_fwhm'],
#                        corr=irac_data_sptpol[2]['aper_corr'],
#                        mag_diff=recovery_mag_thresh)
results_filename = config_dict['root_dir'] + \
                   '/Results/SPT{survey}_resampled_I2_results_{model}_fwhm{fwhm}_corr{corr}_mag{mag_diff}.json'.format(
                       survey=survey, model=psf_model, fwhm=irac_data_sptpol[2]['psf_fwhm'],
                       corr=irac_data_sptpol[2]['aper_corr'],
                       mag_diff=recovery_mag_thresh)
with open(results_filename, 'w') as f:
    json.dump(completeness_results, f, ensure_ascii=False, indent=4, sort_keys=True)
