"""
SPT_AGN_emcee_preprocessing.py
Author: Benjamin Floyd

Performs the GPF and cluster dictionary construction as a preprocessing step to the MCMC sampling. Results are stored in
a JSON file for later use.
"""

import json
from time import time

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from schwimmbad import MPIPool
from scipy.spatial.distance import cdist

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def rebin(a, rebin_factor, wcs=None):
    """
    Rebin an image to the new shape and adjust the WCS
    :param a: Original image
    :param rebin_factor: rebinning scale factor
    :param wcs: Optional, original WCS object
    :return:
    """
    newshape = tuple(rebin_factor * x for x in a.shape)

    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')  # recast the coordinates to int32
    new_image = a[tuple(indices)]

    if wcs is not None:
        new_wcs = wcs.deepcopy()
        new_wcs.pixel_shape = new_image.shape  # Update the NAXIS1/2 values
        new_wcs.wcs.cd /= rebin_factor  # Update the pixel scale

        # Check if the WCS has a PC matrix which is what AstroPy generates. If it exists, just delete it and stick with
        # the CD matrix as the majority of the images have that natively.
        if new_wcs.wcs.has_pc():
            del new_wcs.wcs.pc

        # Transform the reference pixel coordinate
        old_crpix = wcs.wcs.crpix
        new_crpix = np.floor(old_crpix) / a.shape * new_image.shape + old_crpix - np.floor(old_crpix)
        new_wcs.wcs.crpix = new_crpix

        return new_image, new_wcs

    return new_image


def good_pixel_fraction(r, center, cluster_id, rescale_factor=None):
    # Read in the mask file and the mask file's WCS
    image, header = mask_dict[cluster_id]  # This is provided by the global variable mask_dict
    image_wcs = WCS(header)

    if rescale_factor is not None:
        image, image_wcs = rebin(image, rescale_factor, wcs=image_wcs)

    # From the WCS get the pixel scale
    try:
        assert image_wcs.pixel_scale_matrix[0, 1] == 0.
        pix_scale = image_wcs.pixel_scale_matrix[1, 1] * image_wcs.wcs.cunit[1]
    except AssertionError:
        # The pixel scale matrix is not diagonal. We need to diagonalize first
        cd = image_wcs.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        pix_scale = cd_diag[1, 1] * image_wcs.wcs.cunit[1]

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

    # Convert our radius to pixels
    r_pix = (r / pix_scale).decompose()
    r_pix = r_pix.value

    # Because we potentially integrate to larger radii than can be fit on the image we will need to increase the size of
    # our mask. To do this, we will pad the mask with a zeros out to the radius we need.
    # Find the width needed to pad the image to include the largest radius inside the image.
    width = ((int(round(np.max(r_pix) - center_pix[1])),
              int(round(np.max(r_pix) - (image.shape[0] - center_pix[1])))),
             (int(round(np.max(r_pix) - center_pix[0])),
              int(round(np.max(r_pix) - (image.shape[1] - center_pix[0])))))

    # Insure that we are adding a non-negative padding width.
    width = tuple(tuple([i if i >= 0 else 0 for i in axis]) for axis in width)

    large_image = np.pad(image, pad_width=width, mode='constant', constant_values=0)

    # Generate a list of all pixel coordinates in the padded image
    image_coords = np.dstack(np.mgrid[0:large_image.shape[0], 0:large_image.shape[1]]).reshape(-1, 2)

    # The center pixel's coordinate needs to be transformed into the large image system
    center_coord = np.array(center_pix) + np.array([width[1][0], width[0][0]])
    center_coord = center_coord.reshape((1, 2))

    # Compute the distance matrix. The entries are a_ij = sqrt((x_j - cent_x)^2 + (y_i - cent_y)^2)
    image_dists = cdist(image_coords, np.flip(center_coord)).reshape(large_image.shape)

    # select all pixels that are within the annulus
    good_pix_frac = []
    for j in np.arange(len(r_pix) - 1):
        pix_ring = large_image[np.where((r_pix[j] <= image_dists) & (image_dists < r_pix[j + 1]))]

        # Calculate the fraction
        good_pix_frac.append(np.sum(pix_ring) / len(pix_ring))

    return good_pix_frac


def generate_catalog_dict(cluster):
    cutout_id = cluster['Cutout_ID'][0]
    cutout_sz_cent = cluster['SZ_RA', 'SZ_DEC'][0]
    cutout_completeness = cluster['COMPLETENESS_CORRECTION']
    cutout_radial_arcmin = cluster['RADIAL_SEP_ARCMIN']

    # Find the appropriate mesh step size.
    mask_wcs = WCS(mask_dict[cutout_id][1])
    try:
        assert mask_wcs.pixel_scale_matrix[0, 1] == 0.
        pix_scale = mask_wcs.pixel_scale_matrix[1, 1] * mask_wcs.wcs.cunit[1]
    except AssertionError:
        # The pixel scale matrix is not diagonal. We need to diagonalize first
        cd = mask_wcs.pixel_scale_matrix
        _, eig_vec = np.linalg.eig(cd)
        cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
        pix_scale = cd_diag[1, 1] * mask_wcs.wcs.cunit[1]

    # Generate a radial integration mesh.
    rall = np.arange(0., max_radius.value, pix_scale.to_value(u.arcmin) / rescale_fact)

    # Compute the good pixel fractions
    cluster_gpf_all = good_pixel_fraction(rall, cutout_sz_cent, cutout_id,
                                          rescale_factor=rescale_fact)

    # Select only the objects within the same radial limit we are using for integration.
    radial_arcmin_maxr = cutout_radial_arcmin[cutout_radial_arcmin <= rall[-1]]
    completeness_weight_maxr = cutout_completeness[cutout_radial_arcmin <= rall[-1]]

    # Construct our cluster dictionary with all data needed for the sampler.
    # Additionally, store only values in types that can be serialized to JSON
    cluster_dict = {'gpf_rall': cluster_gpf_all, 'rall': list(rall), 'radial_arcmin_maxr': list(radial_arcmin_maxr),
                    'completeness_weight_maxr': list(completeness_weight_maxr)}

    return cutout_id, cluster_dict


hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
max_radius = 5.0 * u.arcmin  # Maximum integration radius in arcmin

rescale_fact = 6  # Factor by which we will rescale the mask images to gain higher resolution

# Read in the mock catalog
sdwfs_catalog = Table.read(f'{hcc_prefix}Data/Output/SDWFS_cutout_IRAGN.fits')

# Read in the mask files for each cluster
sdwfs_catalog_grp = sdwfs_catalog.group_by('Cutout_ID')
mask_dict = {cluster_id: fits.getdata(hcc_prefix + mask_file, header=True) for cluster_id, mask_file
             in zip(sdwfs_catalog_grp.groups.keys['Cutout_ID'],
                    sdwfs_catalog_grp['MASK_NAME'][sdwfs_catalog_grp.groups.indices[:-1]])}

# Compute the good pixel fractions for each cluster and store the array in the catalog.
print('Generating Good Pixel Fractions.')
start_gpf_time = time()
with MPIPool() as pool:
    # if not pool.is_master():
    #     pool.wait()
    #     sys.exit(0)
    pool_results = pool.map(generate_catalog_dict, sdwfs_catalog_grp.groups)

    if pool.is_master():
        catalog_dict = {cluster_id: cluster_info for cluster_id, cluster_info in filter(None, pool_results)}

print('Time spent calculating GPFs: {:.2f}s'.format(time() - start_gpf_time))

# Store the results in a JSON file to be used later by the MCMC sampler
preprocess_file = 'SDWFS_IRAGN_preprocessing.json'
with open(preprocess_file, 'w') as f:
    json.dump(catalog_dict, f, ensure_ascii=False, indent=4)
