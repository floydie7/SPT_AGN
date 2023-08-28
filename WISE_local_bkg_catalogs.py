"""
WISE_local_bkg_catalogs.py
Author: Benjamin Floyd

Fetches the latest WISE catalogs of the local background around the SPT clusters.
"""

import astropy.cosmology.units as cu
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack
from astroquery.ipac.irsa import Irsa
from colossus.cosmology import cosmology
from colossus.halo.mass_adv import changeMassDefinitionCModel

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048, Tcmb0=2.7255, Neff=3)
colo_cosmo = cosmology.fromAstropy(cosmo, sigma8=0.8, ns=0.96, cosmo_name='concordance')

# Read in the SPTcl-IRAGN catalog (this way we only work on cluster in our sample)
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Group to only have cluster information
sptcl_iragn_grp = sptcl_iragn.group_by('SPT_ID')
sptcl_clusters = vstack([Table(cluster['SPT_ID', 'SZ_RA', 'SZ_DEC', 'REDSHIFT', 'M500', 'R500'][0])
                         for cluster in sptcl_iragn_grp.groups])

# Add units
sptcl_clusters['M500'].unit = u.Msun
sptcl_clusters['R500'].unit = u.Mpc

# Convert the cluster M500 to M200 masses
m200_data, r200_data, c200_data = [], [], []
for m500, z in sptcl_clusters['M500', 'REDSHIFT']:
    m500 *= u.Msun
    m200, r200, c200 = changeMassDefinitionCModel(m500.to_value(u.Msun / cu.littleh, cu.with_H0(cosmo.H0)),
                                                  z=z, mdef_in='500c', mdef_out='200c',
                                                  profile='nfw', c_model='child18')
    m200_data.append(m200)
    r200_data.append(r200)
    c200_data.append(c200)

# Add 200 overdensity information to catalogs
sptcl_clusters['M200'] = u.Quantity(m200_data, u.Msun / cu.littleh).to(u.Msun, cu.with_H0(cosmo.H0))
sptcl_clusters['R200'] = u.Quantity(r200_data, u.kpc / cu.littleh).to(u.Mpc, cu.with_H0(cosmo.H0))
sptcl_clusters['C200'] = c200_data

# Download the WISE Catalog in a radius of 2r200
for cluster in sptcl_clusters:
    max_radius = 3 * cluster['R200'] * u.Mpc * cosmo.arcsec_per_kpc_proper(cluster['REDSHIFT']).to(u.arcmin / u.Mpc)
    cluster_coord = SkyCoord(cluster['SZ_RA'], cluster['SZ_DEC'], unit=u.deg)
    wise_catalog = Irsa.query_region(cluster_coord, catalog='catWISE_2020', spatial='Cone', radius=max_radius,
                                     selcols='ra,dec,w1mpro,w1sigmpro,w1flux,w1sigflux,'
                                             'w2mpro,w2sigmpro,w2flux,w2sigflux')

    # Compute the radial distances of all the objects to the cluster center
    wise_coords = SkyCoord(wise_catalog['ra'], wise_catalog['dec'], unit=u.deg)
    sep_deg = cluster_coord.separation(wise_coords)

    # Excise the inner r200 of the WISE catalog
    sep_mpc = sep_deg * cosmo.kpc_proper_per_arcmin(cluster['REDSHIFT']).to(u.Mpc / sep_deg.unit)
    wise_catalog = wise_catalog[sep_mpc > 2 * cluster['R200'] * u.Mpc]

    # Add the cluster information to the WISE catalog
    for cluster_colname in cluster.colnames:
        wise_catalog[cluster_colname] = cluster[cluster_colname]

    # Write the catalog to file
    wise_catalog.write(f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/'
                       f'{cluster["SPT_ID"]}_wise_local_bkg.ecsv', overwrite=True)
