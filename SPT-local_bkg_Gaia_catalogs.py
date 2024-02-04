"""
SPT-local-bkg_Gaia_catalogs.py
Author: Benjamin Floyd

Sends queries to the Gaia archive to retrieve the catalogs of stars present in the local backgrounds of our clusters.
"""

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.gaia import Gaia
from tqdm import tqdm

# Increase the row limit on the Gaia queries
# Gaia.ROW_LIMIT = -1
# Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'

# Read in the SPTcl-IRAGN catalog (this way we only work on cluster in our sample)
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Group to only have cluster information
sptcl_iragn_grp = sptcl_iragn.group_by('SPT_ID')
sptcl_clusters = vstack([Table(cluster['SPT_ID', 'SZ_RA', 'SZ_DEC'][0])
                         for cluster in sptcl_iragn_grp.groups])

for cluster in tqdm(sptcl_clusters, desc='Submitting jobs to Gaia'):
    # Query the archive for all galaxies around the cluster center
    cluster_coord = SkyCoord(cluster['SZ_RA'], cluster['SZ_DEC'], unit=u.deg)
    # gaia_job = Gaia.cone_search_async(coordinate=cluster_coord, radius=1*u.deg,
    #                                   output_file=f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/'
    #                                               f'gaia/{cluster["SPT_ID"]}_bkgs_gaia.fits.gz',
    #                                   output_format='fits', dump_to_file=True,
    #                                   columns=['gaia.source_id', 'gaia.ra', 'gaia.dec',
    #                                            'gaia.parallax', 'gaia.pmra', 'gaia.pmdec',
    #                                            'gaia.in_qso_candidates', 'gaia.in_galaxy_candidates'])
    gaia_job = Gaia.launch_job_async("SELECT "
                                     "gaia.source_id, gaia.ra, gaia.dec, "
                                     "gaia.parallax, gaia.pmra, gaia.pmdec, "
                                     "gaia.in_qso_candidates, gaia.in_galaxy_candidates "
                                     "FROM gaiadr3.gaia_source AS gaia "
                                     "WHERE "
                                     "CONTAINS(POINT(gaia.ra, gaia.dec), "
                                     f"CIRCLE({cluster_coord.ra.value}, {cluster_coord.dec.value}, 1)) = 1 AND "
                                     "gaia.in_qso_candiadates")
    gaia_table = gaia_job.get_results()
    gaia_table.write('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/gaia/'
                     f'{cluster["SPT_ID"]}_bkgs_gaia.fits', overwrite=True)
