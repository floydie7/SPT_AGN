"""
SPT_AGN_recenter_catalogs.py
Author: Benjamin Floyd

Takes mock catalogs that have had radial distances calculated relative to an offset SZ center and recalculates the
distances to be relative to the true SZ center.
"""

import glob

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack


def recenter_clusters(grouped_table):
    """Recalculates the radial distances to be relative to the true SZ center."""
    for cluster in grouped_table.groups:
        # Make a copy of the table group
        cluster = cluster.copy()

        # Get the cluster's redshift
        z = cluster['REDSHIFT'][0]

        # Get the cluster's r500 radius
        r500 = cluster['r500'][0] * u.Mpc

        # Get the true SZ center
        sz_coord = SkyCoord(cluster['SZ_RA'][0], cluster['SZ_DEC'][0], unit=u.deg)

        # Get the object positions
        agn_coords = SkyCoord(cluster['RA'], cluster['DEC'], unit=u.deg)

        # Compute radial distances
        r_arcmin = sz_coord.separation(agn_coords).to(u.arcmin)
        r_r500 = r_arcmin * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc / u.arcmin) / r500

        # Replace the radial columns
        cluster['radial_arcmin'] = r_arcmin
        cluster['radial_r500'] = r_r500

        yield cluster


# Set the cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Get the original catalog file names
catalog_filenames = glob.glob('Data/MCMC/Mock_Catalog/Catalogs/Final_tests/Slope_tests/trial_2/realistic/*')

for f in catalog_filenames:
    # Read in the catalog
    catalog = Table.read(f, format='ascii')

    # Group by cluster
    catalog_grp = catalog.group_by('SPT_ID')

    # Perform the recentering
    recentered_catalog = vstack(list(recenter_clusters(catalog_grp)))

    # Write the new catalog to disk
    recentered_catalog.write(f.replace('trial_2', 'trial_4'), format='ascii', overwrite=True)
