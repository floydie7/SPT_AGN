"""
SPT_AGN_emcee_preprocessing.py
Author: Benjamin Floyd

Performs the GPF and cluster dictionary construction as a preprocessing step to the MCMC sampling. Results are stored in
a JSON file for later use.
"""

import json
from argparse import ArgumentParser
from time import time

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from schwimmbad import MPIPool
from scipy.spatial.distance import cdist
from synphot import SourceSpectrum, SpectralElement, units

from k_correction import k_corr_abs_mag

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def rebin(a, rebin_factor, wcs=None) -> tuple[np.ndarray, WCS] | np.ndarray:
    """
    Rebin an image to the new shape and adjust the WCS.

    Parameters
    ----------
    a: array-like
        Original image.
    rebin_factor: float
        Rebinning scale factor.
    wcs: WCS, optional
        Original image world coordinate system (WCS) object.

    Returns
    -------
    new_image: np.ndarray
        The rebinned image.
    new_wcs: WCS
        The updated WCS for the rebinned object.
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


def good_pixel_fraction(r, z, r500, center, cluster_id, rescale_factor=None):
    """
    Computes the fraction of unmasked pixels within an annulus.

    Parameters
    ----------
    r: array-like
        Radial axis on which we will form our annuli.
    z: float
        Redshift of the cluster.
    r500: u.Quantity
        R500 radius of the cluster.
    center: Table
        Center of the cluster.
    cluster_id: str
        Cluster name/ID.
    rescale_factor: float, optional
        Factor by which to rescale the original image. This rebins the pixels of the image into pixels of size
        `old_pix / rescale_factor`.

    Returns
    -------
    good_pix_frac: list of float
        A list of the fractional area within each annulus that is unmasked.
    """

    # Read in the mask file and the mask file's WCS
    image, header = mask_dict[cluster_id]  # This is provided by the global variable mask_dict
    image_wcs = WCS(header)

    if rescale_factor is not None:
        image, image_wcs = rebin(image, rescale_factor, wcs=image_wcs)

    # From the WCS get the pixel scale
    pix_scale = image_wcs.proj_plane_pixel_scales()[0]

    # Convert our center into pixel units
    center_pix = image_wcs.wcs_world2pix(center['SZ_RA'], center['SZ_DEC'], 0)

    # Convert our radius to pixels
    r_pix = r * r500 * cosmo.arcsec_per_kpc_proper(z).to(pix_scale.unit / u.Mpc) / pix_scale
    r_pix = r_pix.value

    # Because we potentially integrate to larger radii than can be fit on the image we will need to increase the size of
    # our mask. To do this, we will pad the mask with a zeros out to the radius we need.
    # Find the width needed to pad the image to include the largest radius inside the image.
    width = ((int(round(np.max(r_pix) - center_pix[1])),
              int(round(np.max(r_pix) - (image.shape[0] - center_pix[1])))),
             (int(round(np.max(r_pix) - center_pix[0])),
              int(round(np.max(r_pix) - (image.shape[1] - center_pix[0])))))

    # Ensure that we are adding a non-negative padding width.
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


def generate_catalog_dict(cluster: Table) -> tuple[str, dict]:
    """Parses the input catalog into a dictionary structure containing only the necessary information for the MCMC
    sampler."""
    cluster_id = cluster['SPT_ID'][0]
    cluster_z = cluster['REDSHIFT'][0]
    cluster_m500 = cluster['M500'][0] * u.Msun
    cluster_r500 = cluster['R500'][0] * u.Mpc
    cluster_sz_cent = cluster['SZ_RA', 'SZ_DEC'][0]
    cluster_completeness = cluster['COMPLETENESS_CORRECTION']
    cluster_radial_r500 = cluster['RADIAL_SEP_R500']
    cluster_agn_membership = cluster['SELECTION_MEMBERSHIP']
    j_band_abs_mag = cluster['J_ABS_MAG']

    # Set up a switch to handle the options for the radial separation
    # radial_switch = {0.0: cluster['RADIAL_SEP_R500'],
    #                  0.5: cluster['RADIAL_SEP_R500_HALF_OFFSET'],
    #                  0.75: cluster['RADIAL_SEP_R500_075_OFFSET'],
    #                  1.0: cluster['RADIAL_SEP_R500_OFFSET']}
    # cluster_radial_r500 = radial_switch[args.miscentering]

    # Determine the maximum integration radius for the cluster in terms of r500 units.
    max_radius_r500 = max_radius * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc / u.arcmin) / cluster_r500

    # Find the appropriate mesh step size. Since we work in r500 units we convert the pixel scale from angle/pix to
    # r500/pix.
    mask_wcs = WCS(mask_dict[cluster_id][1])
    pix_scale = mask_wcs.proj_plane_pixel_scales()[0]
    pix_scale_r500 = pix_scale * cosmo.kpc_proper_per_arcmin(cluster_z).to(u.Mpc / pix_scale.unit) / cluster_r500

    # Generate a radial integration mesh.
    rall = np.arange(0., max_radius_r500, pix_scale_r500 / rescale_fact)
    # rall = np.linspace(0., max_radius_r500.value, num=10_000)

    # Compute the good pixel fractions
    cluster_gpf_all = good_pixel_fraction(rall, cluster_z, cluster_r500, cluster_sz_cent, cluster_id,
                                          rescale_factor=rescale_fact)
    # cluster_gpf_all = None

    # Select only the objects within the same radial limit we are using for integration.
    radial_r500_maxr = cluster_radial_r500[cluster_radial_r500 <= rall[-1]]
    completeness_weight_maxr = cluster_completeness[cluster_radial_r500 <= rall[-1]]
    agn_membership_maxr = cluster_agn_membership[cluster_radial_r500 <= rall[-1]]
    j_band_abs_mag_maxr = j_band_abs_mag[cluster_radial_r500 <= rall[-1]]

    # For the luminosity integration mesh we will compute the equivalent J-band absolute magnitude from the apparent
    # 4.5 um magnitudes at the cluster redshift.
    faint_end_45_apmag = 17.46  # Vega mag
    bright_end_45_apmag = 10.45  # Vega mag
    irac_45_filter = SpectralElement.from_file(f'{hcc_prefix}Data_Repository/filter_curves/Spitzer_IRAC'
                                               f'/080924ch2trans_full.txt', wave_unit=u.um)
    flamingos_j_filter = SpectralElement.from_file(f'{hcc_prefix}Data_Repository/filter_curves/KPNO/KPNO_2.1m'
                                                   f'/FLAMINGOS/FLAMINGOS.BARR.J.MAN240.ColdWitness.txt',
                                                   wave_unit=u.nm)
    qso2_sed = SourceSpectrum.from_file(f'{hcc_prefix}Data_Repository/SEDs/Polletta-SWIRE/QSO2_template_norm.sed',
                                        wave_unit=u.Angstrom, flux_unit=units.FLAM)
    faint_end_j_absmag = k_corr_abs_mag(faint_end_45_apmag, z=cluster_z, f_lambda_sed=qso2_sed,
                                        zero_pt_obs_band=179.7 * u.Jy, zero_pt_em_band='vega',
                                        obs_filter=irac_45_filter, em_filter=flamingos_j_filter, cosmo=cosmo)
    bright_end_j_absmag = k_corr_abs_mag(bright_end_45_apmag, z=cluster_z, f_lambda_sed=qso2_sed,
                                         zero_pt_obs_band=179.7 * u.Jy, zero_pt_em_band='vega',
                                         obs_filter=irac_45_filter, em_filter=flamingos_j_filter, cosmo=cosmo)

    # Generate a luminosity integration mesh defined by the J-band equivalents of the 4.5 um apparent magnitude limits
    jall = np.linspace(bright_end_j_absmag, faint_end_j_absmag, num=400)

    # Construct our cluster dictionary with all data needed for the sampler.
    # Additionally, store only values in types that can be serialized to JSON
    cluster_dict = {'redshift': cluster_z, 'm500': cluster_m500.value, 'r500': cluster_r500.value,
                    'gpf_rall': cluster_gpf_all, 'rall': list(rall), 'radial_r500_maxr': list(radial_r500_maxr),
                    'completeness_weight_maxr': list(completeness_weight_maxr),
                    'agn_membership_maxr': list(agn_membership_maxr),
                    'j_abs_mag': list(j_band_abs_mag_maxr), 'jall': list(jall)}

    return cluster_id, cluster_dict


parser = ArgumentParser(description='Generates a preprocessing file for use in MCMC sampling.')
parser.add_argument('catalog', help='Catalog to process. Needs to be given as a fully qualified path name.')
parser.add_argument('--output', help='Output filename', default='SPTcl_IRAGN_preprocessing.json', type=str)
parser.add_argument('--rejection', action='store_true', help='Use the rejection sampling flag to filter the catalog.')
parser.add_argument('--miscentering', help='Factor of miscentering to be used.', choices=[0.5, 0.75, 1.0], default=0.0,
                    type=float)
parser_grp = parser.add_mutually_exclusive_group()
parser_grp.add_argument('--cluster-only', action='store_true',
                        help='Generate a preprocessing file only on cluster objects.')
parser_grp.add_argument('--background-only', action='store_true',
                        help='Generate preprocessing file only on background objects.')
args = parser.parse_args()

# hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'
hcc_prefix = ''
max_radius = 5. * u.arcmin  # Maximum integration radius in arcmin

rescale_fact = 6  # Factor by which we will rescale the mask images to gain higher resolution

# Read in the mock catalog
sptcl_catalog = Table.read(args.catalog)
# sptcl_catalog = Table.read('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/pure_poisson/'
#                            'mock_AGN_catalog_t10.000_e4.00_z-1.00_b1.00_rc0.100_C0.316_maxr5.00_seed3775_6x2_fullMasks_forGPF_withLF.fits')

# Filter the catalog using the rejection flag
# if args.rejection:
#     sptcl_catalog = sptcl_catalog[sptcl_catalog['COMPLETENESS_REJECT'].astype(bool)]

# Separate the cluster and background objects
cluster_only = sptcl_catalog[sptcl_catalog['CLUSTER_AGN'].astype(bool)]
background_only = sptcl_catalog[~sptcl_catalog['CLUSTER_AGN'].astype(bool)]

if args.cluster_only:
    # Run on only cluster objects
    sptcl_catalog = cluster_only
elif args.background_only:
    # Run on only background objects
    sptcl_catalog = background_only
else:
    # Run on full catalog
    sptcl_catalog = sptcl_catalog

# Read in the mask files for each cluster
sptcl_catalog_grp = sptcl_catalog.group_by('SPT_ID')
mask_dict = {cluster_id: fits.getdata(f'{hcc_prefix}{mask_file}', header=True) for cluster_id, mask_file
             in zip(sptcl_catalog_grp.groups.keys['SPT_ID'],
                    sptcl_catalog_grp['MASK_NAME'][sptcl_catalog_grp.groups.indices[:-1]])}

# Compute the good pixel fractions for each cluster and store the array in the catalog.
print('Generating Good Pixel Fractions.')
start_gpf_time = time()
with MPIPool() as pool:
    # if not pool.is_master():
    #     pool.wait()
    #     sys.exit(0)
    pool_results = pool.map(generate_catalog_dict, sptcl_catalog_grp.groups)

    if pool.is_master():
        catalog_dict = {cluster_id: cluster_info for cluster_id, cluster_info in filter(None, pool_results)}

print('Time spent calculating GPFs: {:.2f}s'.format(time() - start_gpf_time))

# Store the results in a JSON file to be used later by the MCMC sampler
local_dir = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Port_Rebuild_Tests/pure_poisson/'
# preprocess_file = f'{local_dir}SPTcl_IRAGN_preprocessing_fullMasks_withGPF_withLF_2kdenseJall_100cl_t50.json'
preprocess_file = f'{local_dir}{args.output}'
with open(preprocess_file, 'w') as f:
    json.dump(catalog_dict, f, ensure_ascii=False, indent=4)
print(args.output)
