"""
SPTPol_cutout_trim_and_triage.py
Author: Benjamin Floyd

Trims the larger 16'x16' SPTpol 100d cluster cutouts to 5'x5' cutouts and performs a triage on the coverage maps
requiring that at least __% of pixels be above our coverage cutoff of 4 exposures.
"""
import datetime
import glob
import re
from itertools import groupby
from importlib_metadata import version

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization import imshow_norm, ZScaleInterval


def cluster_keys(name):
    return cluster_id_pattern.search(name).group(0)


def make_cutouts(cluster_id):
    images = cutout_dict[cluster_id]

    # Sort the images by type and read them in
    image_dict = {}
    for image in images:
        if 'I1' in image and 'cov' not in image:
            image_dict['I1_sci'] = fits.getdata(image, header=True)
        elif 'I1' in image and 'cov' in image:
            image_dict['I1_cov'] = fits.getdata(image, header=True)
        elif 'I2' in image and 'cov' not in image:
            image_dict['I2_sci'] = fits.getdata(image, header=True)
        elif 'I2' in image and 'cov' in image:
            image_dict['I2_cov'] = fits.getdata(image, header=True)

    # Read in the photometric catalog
    catalog = Table.read(cutout_catalogs[cluster_id])

    # Set the cluster center for the anchor point of the cutout
    cluster_center = SkyCoord(huang['RA'][huang['SPT_ID'] == cluster_id],
                              huang['Dec'][huang['SPT_ID'] == cluster_id], unit='deg')

    # Size of the cutout will be 5x5 arcmin
    cutout_size = 5 * u.arcmin

    temp_dict = {}
    for image_type, (image, header), in image_dict.items():
        # Get the WCS from the header
        wcs = WCS(header)

        # Make the cutout
        cutout = Cutout2D(image, cluster_center, cutout_size, wcs=wcs, mode='partial')

        # Update the header from the original image
        header.update(cutout.wcs.to_header())

        # Add comments to the header documenting the cutout
        header['history'] = f'Cutout generated using Astropy v{version("astropy")}'
        header['history'] = datetime.datetime.now().isoformat(' ', timespec='seconds')
        header['history'] = 'Cutout created by Benjamin Floyd'

        # Create the HDU
        cutout_hdu = fits.PrimaryHDU(cutout.data, header=header)

        # The cutout names use the standard format used for the SPT-SZ cutouts
        cutout_fname = 'Data/SPTPol/images/cluster_cutouts_trimmed/{band}_{spt_id}_mosaic{cov}.cutout.fits' \
            .format(band='I1' if 'I1' in image_type else 'I2',
                    spt_id=cluster_id,
                    cov='_cov' if 'cov' in image_type else '')

        temp_dict[image_type] = (cutout_fname, cutout_hdu)

        if image_type == 'I2_sci':
            # Update the x and y pixel coordinates in the catalog
            new_xy_image = cutout.wcs.wcs_world2pix(catalog['ALPHA_J2000'], catalog['DELTA_J2000'], 0)
            catalog['X_IMAGE'] = new_xy_image[0]
            catalog['Y_IMAGE'] = new_xy_image[1]

            # Trim the catalog to only include objects within the image
            image_objs = ((cutout.xmin_cutout <= catalog['X_IMAGE']) &
                          (catalog['X_IMAGE'] <= cutout.xmax_cutout) &
                          (cutout.ymin_cutout <= catalog['Y_IMAGE']) &
                          (catalog['Y_IMAGE'] <= cutout.ymax_cutout))

            # Set the file name for the catalog
            catalog_name = f'Data/SPTPol/catalogs/cluster_cutouts_trimmed/{cluster_id}.SSDFv9.cat'

            # Add the catalog to the dictionary
            temp_dict['catalog'] = (catalog_name, catalog[image_objs])

    return temp_dict


def triage_cutouts(threshold):
    ch1_min_cov = 4
    ch2_min_cov = 4

    for cluster_id in trimmed_cutouts:
        # Pull the coverage maps
        I1_cov = trimmed_cutouts[cluster_id]['I1_cov'][1].data
        I2_cov = trimmed_cutouts[cluster_id]['I2_cov'][1].data

        # Generate a quick mask of good pixels
        mask = np.logical_and((I1_cov >= ch1_min_cov), (I2_cov >= ch2_min_cov))

        # Calculate the percentage of good pixels in the image
        good_pixel_ratio = np.count_nonzero(mask) / mask.size

        # If the cluster does not have a good pixel ratio of at least our threshold, mark it for removal
        if not good_pixel_ratio >= threshold:
            # Show the mask
            fig, ax = plt.subplots()
            ax.imshow(mask.astype(int), origin='lower', cmap='gray_r')
            ax.set(title=cluster_id)
            plt.show()
            yield cluster_id, good_pixel_ratio


def write_files(cluster_dict):
    for cluster_info in cluster_dict.values():
        for data_type, data_info in cluster_info.items():
            file_name, data = data_info
            if data_type != 'catalog':
                data.writeto(file_name, overwrite=True)
            else:
                data.write(file_name, format='ascii', overwrite=True)


# Pattern for cluster ids
cluster_id_pattern = re.compile(r'SPT-CLJ\d+-\d+')

# Read in the SPTPol 100d cluster catalog
huang = Table.read('Data/sptpol100d_catalog_huang19.fits')

# Get a list of all the cutout images
cutout_images = glob.glob('Data/SPTPol/images/cluster_cutouts_large/*.fits')

# Sort the list using the cluster ids
cutout_images = sorted(cutout_images, key=cluster_keys)

# Group the files by cluster
cutout_dict = {k: list(g) for k, g in groupby(cutout_images, key=cluster_keys)}

# Repeat for the catalogs
cutout_catalogs = glob.glob('Data/SPTPol/catalogs/cluster_cutouts_large/*.fits')
cutout_catalogs = {cluster_keys(f): f for f in cutout_catalogs}

# Make the trimmed cutouts
trimmed_cutouts = {cluster_id: make_cutouts(cluster_id) for cluster_id in cutout_dict}

# Perform the triage on the cutouts
# clusters_to_remove = np.array(list(triage_cutouts(threshold=0.7)))

# Write the surviving clusters to disk
# write_files(trimmed_cutouts)
