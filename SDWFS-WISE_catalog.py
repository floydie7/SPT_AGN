"""
SDWFS-WISE_catalog.py
Author: Benjamin Floyd

Fetches a catalog of all WISE sources within the SDWFS footprint.
"""

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

# We first need to determine the boundaries of the field
sdwfs_wcs = WCS('Data_Repository/Images/Bootes/SDWFS/I1_bootes.v32.fits')
sdwfs_footprint = SkyCoord(sdwfs_wcs.calc_footprint(), unit=u.deg)

Irsa.ROW_LIMIT = 1e9
wise_catalog = Irsa.query_region(catalog='catWISE_2020', spatial='Polygon', polygon=sdwfs_footprint,
                                 selcols='ra,dec,w1mpro,w1sigmpro,w1flux,w1sigflux,w2mpro,w2sigmpro,w2flux,w2sigflux')
wise_catalog.write('Data_Repository/Catalogs/Bootes/SDWFS/SDWFS_catWISE.ecsv')
print(wise_catalog.info)
print(wise_catalog.info('stats'))
