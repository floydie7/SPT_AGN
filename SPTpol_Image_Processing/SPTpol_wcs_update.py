"""
SPTpol_wcs_update.py
Author: Benjamin Floyd

Updates the reference pixels of the WCSs for SPTpol cluster cutouts.
"""

import re
import glob
from itertools import groupby
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

prefix = '/home/ben/PycharmProjects/SPT_AGN/'


def cluster_id(key):
    return re.search(r'SPT-CLJ\d+[+-]?\d+[-\d+]?', key).group(0)


# Read in the SPTpol 100d catalog
Huang = Table.read(f'{prefix}Data/sptpol100d_catalog_huang19.fits')

# Get list of files to process
cluster_cutouts = glob.glob(f'{prefix}Data/SPTPol/images/cluster_cutouts/*.fits')

# Group the cutouts by cluster
cutout_dict = {spt_id: list(cutout_files) for spt_id, cutout_files in groupby(sorted(cluster_cutouts, key=cluster_id),
                                                                              key=cluster_id)}

for spt_id, cutouts in cutout_dict.items():
    # Get the cluster coordinate from the table
    cluster_RADec = Huang['RA', 'Dec'][Huang['SPT_ID'] == spt_id]
    cluster_coord = SkyCoord(cluster_RADec['RA'][0], cluster_RADec['Dec'][0], unit='deg')

    # Iterate through the images, updating the WCS reference coordinates
    for cutout in cutouts:
        # Open the file in read/write mode
        with fits.open(cutout, 'update') as hdulist:
            # Get the WCS
            wcs = WCS(hdulist[0].header)

            # Find the cluster coordinate in pixels
            cluster_pix = cluster_coord.to_pixel(wcs=wcs, origin=1)

            # Update the reference pixel coordinates
            wcs.wcs.crpix = cluster_pix

            # Update the reference sky coordinates
            wcs.wcs.crval = [cluster_coord.ra.value, cluster_coord.dec.value]

            # Update the header
            hdulist[0].header.update(wcs.to_header())
