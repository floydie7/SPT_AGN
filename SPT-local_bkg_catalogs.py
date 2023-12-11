"""
SPT-local_bkg_catalogs.py
Author: Benjamin Floyd

Downloads the WISE catalogs around each cluster out to a wide radius for use in `SPT-local_bkg_measurement`.
This replaces both the original `WISE_local_bkg_catalogs` and the modified test implementation in
`WISE_local_bkg_area_tests`.
"""
import astropy.cosmology.units as cu
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable, vstack, hstack
from astroquery.ipac.irsa import Irsa
from colossus.cosmology import cosmology
from colossus.halo.mass_adv import changeMassDefinitionCModel

# Increase the IRSA query limit
Irsa.ROW_LIMIT = 1e6

# Set up cosmology objects
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048, Tcmb0=2.7255, Neff=3)
colo_cosmo = cosmology.fromAstropy(cosmo, sigma8=0.8, ns=0.96, cosmo_name='concordance')

# Read in the SPTcl-IRAGN catalog (this way we only work on cluster in our sample)
sptcl_iragn = QTable.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Group to only have cluster information
sptcl_iragn_grp = sptcl_iragn.group_by('SPT_ID')
sptcl_clusters = vstack([QTable(cluster['SPT_ID', 'SZ_RA', 'SZ_DEC', 'REDSHIFT', 'M500', 'R500'][0])
                         for cluster in sptcl_iragn_grp.groups])

# Add units
sptcl_clusters['M500'].unit = u.Msun

clusters = []
for cluster in sptcl_clusters:
    # Compute the test cluster's 200-overdensity values
    m200, r200, c200 = changeMassDefinitionCModel(cluster['M500'].to_value(u.Msun / cu.littleh, cu.with_H0(cosmo.H0)),
                                                  cluster['REDSHIFT'], mdef_in='500c', mdef_out='200c',
                                                  profile='nfw', c_model='child18')
    m200 = (m200 * u.Msun / cu.littleh).to(u.Msun, cu.with_H0(cosmo.H0))
    r200 = (r200 * u.kpc / cu.littleh).to(u.Mpc, cu.with_H0(cosmo.H0))
    clusters.append(hstack([cluster, QTable(rows=[[m200, r200, c200]], names=['M200', 'R200', 'C200'])]))
sptcl_clusters = vstack(clusters)

for cluster in sptcl_clusters:
    # Query the archive for all galaxies around the cluster center
    cluster_coord = SkyCoord(cluster['SZ_RA'], cluster['SZ_DEC'], unit=u.deg)
    wise_catalog = Irsa.query_region(cluster_coord, catalog='catWISE_2020', spatial='Cone', radius=1*u.deg,
                                     selcols='ra,dec,w1mpro,w1sigmpro,w1flux,w1sigflux,'
                                             'w2mpro,w2sigmpro,w2flux,w2sigflux')

    # Merge the cluster information into the WISE galaxy catalog
    for colname in cluster.colnames:
        wise_catalog[colname] = cluster[colname]

    # Write the catalog to file
    wise_catalog.write('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/'
                       f'{cluster["SPT_ID"]}_wise_local_bkg.ecsv', overwrite=True)
