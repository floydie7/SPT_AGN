"""
SPTcl-HST-SNAP_close_pairs_matching.py
Author: Benjamin Floyd

Matches the HST-SNAP positional galaxy catalogs with the SPTcl-IRAGN catalog
"""

from astropy.table import Table, vstack, setdiff
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN_no-stars.fits')
sptcl_iragn = sptcl_iragn[sptcl_iragn['SELECTION_MEMBERSHIP'] >= 0.5]

hst_dirs = Path('Data_Repository/Images/SPT/HST-SNAP').glob('SPT-CLJ*')
for hst_dir in hst_dirs:
    cluster_id = hst_dir.name

    hst_cat = Table.read(str(*Path(hst_dir).glob('*.cat')), format='ascii.sextractor')
    iragn_cat = sptcl_iragn[sptcl_iragn['SPT_ID'] == cluster_id]
    cluster_z = iragn_cat['REDSHIFT'][0]

    hst_coords = SkyCoord(hst_cat['ALPHA_J2000'], hst_cat['DELTA_J2000'], distance=cosmo.angular_diameter_distance(cluster_z))
    ir_coords = SkyCoord(iragn_cat['ALPHA_J2000'], iragn_cat['DELTA_J2000'], distance=cosmo.angular_diameter_distance(cluster_z))

    idx, sep, _ = ir_coords.match_to_catalog_sky(hst_coords)
    hst_iragn = hst_cat[idx[sep <= 1 * u.arcsec]]

    hst_nonagn = setdiff(hst_cat, hst_iragn)

    hst_iragn_coords = SkyCoord(hst_iragn['ALPHA_J2000'], hst_iragn['DELTA_J2000'], distance=cosmo.angular_diameter_distance(cluster_z))
    hst_nonagn_coords = SkyCoord(hst_nonagn['ALPHA_J2000'], hst_nonagn['DELTA_J2000'], distance=cosmo.angular_diameter_distance(cluster_z))

    idx_agn, idx_all, d2d, d3d = hst_coords.search_around_3d(hst_iragn, 50 * u.kpc)
