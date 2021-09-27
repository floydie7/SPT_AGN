"""
pixel_scale_zero_mag_test.py
Author: Benjamin Floyd

Confirms the pixel scale for all SPT images and attempts to compute the correct zero point magnitude for the catalogs.
"""

from astropy.wcs import WCS
import numpy as np
import glob
import astropy.units as u
import re

id_pattern = re.compile(r'I[12]_(SPT-CLJ\d+-\d+)')

# Zero magnitude fluxes
F_0 = {1: 280.9 * u.Jy,
       2: 179.7 * u.Jy}


def check_pixel_scales(image_list, true_pixel_scale):
    # Cycle through all images and confirm the pixel scale is 0.8626716 arcseconds/pixel
    for image in image_list:
        wcs = WCS(image)
        try:
            assert wcs.pixel_scale_matrix[0, 1] == 0.
            pixel_scale = wcs.pixel_scale_matrix[1, 1] * wcs.wcs.cunit[1]
        except AssertionError:
            # The pixel scale matrix is not diagonal. We need to diagonalize first
            cd = wcs.pixel_scale_matrix
            _, eig_vec = np.linalg.eig(cd)
            cd_diag = np.linalg.multi_dot([np.linalg.inv(eig_vec), cd, eig_vec])
            pixel_scale = cd_diag[1, 1] * wcs.wcs.cunit[1]

        try:
            assert u.isclose(pixel_scale, true_pixel_scale)

        except AssertionError:
            cluster_id = id_pattern.search(image).group(1)
            print(f'Cluster {cluster_id} has a pixel scale of {pixel_scale.to(u.arcsec)}')


#%% SPT-SZ sample

# Image lists
I1_sci_images = glob.glob('Data/Images/I1_*_mosaic.cutout.fits')
I2_sci_images = glob.glob('Data/Images/I2_*_mosaic.cutout.fits')

sptsz_pixel_scale = 0.8626716 * u.arcsec

check_pixel_scales([*I1_sci_images, *I2_sci_images], true_pixel_scale=sptsz_pixel_scale)

# Conversion factor
sptsz_sr_per_pixel = sptsz_pixel_scale.to(u.deg)**2 * 3.04617e-4 * u.sr / u.deg**2
C_08 = (1 * u.MJy/u.sr * sptsz_sr_per_pixel).to(u.Jy)

ch2_zmag_08 = 2.5 * np.log10(F_0[2]/C_08)
print(f'Channel 2 zero-point magnitude for {sptsz_pixel_scale} pixels: {ch2_zmag_08:.3f} Vega mag')

#%% SPTpol/SSDF sample

# Image lists
I1_sci_images = glob.glob('Data/SPTPol/images/cluster_cutouts/I1_*_mosaic.cutout.fits')
I2_sci_images = glob.glob('Data/SPTPol/images/cluster_cutouts/I2_*_mosaic.cutout.fits')

ssdf_pixel_scale = 0.6000012 * u.arcsec

check_pixel_scales([*I1_sci_images, *I2_sci_images], true_pixel_scale=ssdf_pixel_scale)

ssdf_sr_per_pixel = ssdf_pixel_scale.to(u.deg)**2 * 3.04617e-4 * u.sr / u.deg**2
C_06 = (1 * u.MJy/u.sr * ssdf_sr_per_pixel).to(u.Jy)

ch2_zmag_06 = 2.5 * np.log10(F_0[2]/C_06)
print(f'Channel 2 zero-point magnitude for {ssdf_pixel_scale} pixels: {ch2_zmag_06:.3f} Vega mag')
