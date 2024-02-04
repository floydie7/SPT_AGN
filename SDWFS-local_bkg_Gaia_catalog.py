"""
SDWFS-local_bkg_Gaia_catalog.py
Author: Benjamin Floyd

Fetches a catalog from the Gaia archive for the SDWFS footprint.
"""

from astropy.wcs import WCS
from astroquery.gaia import Gaia

# We first need to determine the boundaries of the field
sdwfs_wcs = WCS('Data_Repository/Images/Bootes/SDWFS/I1_bootes.v32.fits')
sdwfs_footprint = sdwfs_wcs.calc_footprint().flatten()

gaia_job = Gaia.launch_job_async("SELECT "
                                 "gaia.source_id, gaia.ra, gaia.dec, "
                                 "gaia.parallax, gaia.pmra, gaia.pmdec, "
                                 "gaia.in_qso_candidates, gaia.in_galaxy_candidates "
                                 "FROM gaiadr3.gaia_source AS gaia "
                                 "WHERE "
                                 "CONTAINS(POINT(gaia.ra, gaia.dec), "
                                 f"POLYGON({', '.join(str(coord) for coord in sdwfs_footprint)})) = 1")
sdwfs_gaia = gaia_job.get_results()
sdwfs_gaia.write('Data_Repository/Catalogs/Bootes/SDWFS/SDWFS_Gaia.fits')
