"""
SPTpol100d_update_IRAC_catalogs.py
Author: Benjamin Floyd

This updates the photometric catalogs to use the v10 SSDF catalog.
"""

import glob
import re

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from schwimmbad import MPIPool

id_pattern = re.compile(r'SPT-CLJ\d+-\d+')


def make_catalog_cutout(image_name):
    # Extract the cluster name
    cluster_id = id_pattern.search(image_name).group(0)

    # Read in the WCS
    wcs = WCS(image_name)

    # Create a mask for objects that are contained in the image
    on_image = wcs.footprint_contains(ssdf_coords)

    # Create sub-catalog for objects that are part of this cluster
    cluster_catalog = Table.from_pandas(ssdf_cat[on_image])

    # Write out the catalog
    cluster_catalog.write(f'Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPTpol_100d/{cluster_id}.SSDFv10.cat',
                          format='ascii')


# Collect the image file names
sptpol_img_names = glob.glob('Data_Repository/Images/SPT/Spitzer_IRAC/SPTpol_100d/I2*mosaic.cutout.fits')

# Read in the SSDF catalog
ssdf_template = Table.read('Data_Repository/Project_Data/SPT-IRAGN/SPTPol/catalogs/ssdf_table_template.cat',
                           format='ascii.sextractor')
ssdf_cat = pd.read_csv('Data_Repository/Catalogs/SSDF/SSDF2.20170125.v10.public.cat', delim_whitespace=True,
                       skiprows=52, names=ssdf_template.colnames)

# Create sky coordinate object for SSDF catalog
ssdf_coords = SkyCoord(ssdf_cat['ALPHA_J2000'], ssdf_cat['DELTA_J2000'], unit=u.deg)

with MPIPool() as pool:
    pool.map(make_catalog_cutout, sptpol_img_names)
