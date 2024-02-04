"""
SSDF-local_bkg_Gaia_catalog.py
Author: Benjamin Floyd

Fetches the Gaia catalog for all SSDF tiles then stitches all tile-level catalogs together to make a single survey-level
catalog. Queries must be sent in tile-by-tile to get around the 3 million source limit per query by the Gaia server.
"""

import glob
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from astropy.table import vstack, unique

# Get the list of tile file names
ssdf_image_names = glob.glob('Data_Repository/Images/SSDF/I1_SSDF*_mosaic.fits')

tile_tables = []
for tile_name in ssdf_image_names:
    # Get the tile's footprint
    tile_wcs = WCS(tile_name)
    tile_footprint = tile_wcs.calc_footprint().flatten()

    # Submit the query to Gaia
    gaia_job = Gaia.launch_job_async("SELECT "
                                     "gaia.source_id, gaia.ra, gaia.dec, "
                                     "gaia.parallax, gaia.pmra, gaia.pmdec, "
                                     "gaia.in_qso_candidates, gaia.in_galaxy_candidates "
                                     "FROM gaiadr3.gaia_source AS gaia "
                                     "WHERE "
                                     "CONTAINS(POINT(gaia.ra, gaia.dec), "
                                     f"POLYGON({', '.join(str(coord) for coord in tile_footprint)})) = 1")
    tile_table = gaia_job.get_results()
    tile_tables.append(tile_table)

# Stack all catalogs together and remove any duplicate rows (due to the overlaps in the seams between the tiles)
tile_tables = vstack(tile_tables)
tile_tables = unique(tile_tables, keys='source_id')

# Write the catalog out
tile_tables.write('Data_Repository/Catalogs/SSDF/SSDF_Gaia_sources.fits')
