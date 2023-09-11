"""
WISE_local_bkg_area_tests.py
Author: Benjamin Floyd

This script will help determine the appropriate area needed for the background annulus.
"""

import json

import astropy.cosmology.units as cu
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astro_compendium.utils.small_poisson import small_poisson
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable, hstack, vstack
from astroquery.ipac.irsa import Irsa
from colossus.cosmology import cosmology
from colossus.halo.mass_adv import changeMassDefinitionCModel
from scipy.interpolate import interp1d

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048, Tcmb0=2.7255, Neff=3)
colo_cosmo = cosmology.fromAstropy(cosmo, sigma8=0.8, ns=0.96, cosmo_name='concordance')

#%%
# Read in the SPTcl-IRAGN catalog (this way we only work on cluster in our sample)
sptcl_iragn = QTable.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Group to only have cluster information
sptcl_iragn_grp = sptcl_iragn.group_by('SPT_ID')
sptcl_clusters = vstack([QTable(cluster['SPT_ID', 'SZ_RA', 'SZ_DEC', 'REDSHIFT', 'M500', 'R500'][0])
                         for cluster in sptcl_iragn_grp.groups])

# Add units
sptcl_clusters['M500'].unit = u.Msun

#%%
# Select the cluster with the minimum angular r500 radius
R500_armin = sptcl_clusters['R500'] * cosmo.arcsec_per_kpc_proper(sptcl_clusters['REDSHIFT']).to(u.arcmin / u.Mpc)
min_r500_idx = np.argmin(R500_armin)
test_cluster = sptcl_clusters[min_r500_idx]

# Compute the test cluster's 200-overdensity values
m200, r200, c200 = changeMassDefinitionCModel(test_cluster['M500'].to_value(u.Msun / cu.littleh, cu.with_H0(cosmo.H0)),
                                              test_cluster['REDSHIFT'], mdef_in='500c', mdef_out='200c',
                                              profile='nfw', c_model='duffy08')
m200 = (m200 * u.Msun / cu.littleh).to(u.Msun, cu.with_H0(cosmo.H0))
r200 = (r200 * u.kpc / cu.littleh).to(u.Mpc, cu.with_H0(cosmo.H0))
test_cluster = hstack([test_cluster, QTable(rows=[[m200, r200, c200]], names=['M200', 'R200', 'C200'])])

#%%
# For the test cluster, query an initial large area centered on the cluster
Irsa.ROW_LIMIT = 1e6
cluster_coord = SkyCoord(test_cluster['SZ_RA'][0], test_cluster['SZ_DEC'][0], unit=u.deg)
wise_catalog = Irsa.query_region(cluster_coord, catalog='catWISE_2020', spatial='Cone', radius=1 * u.deg,
                                 selcols='ra,dec,w1mpro,w1sigmpro,w1flux,w1sigflux,'
                                         'w2mpro,w2sigmpro,w2flux,w2sigflux')

# Merge the cluster information into the WISE catalog
for colname in test_cluster.colnames:
    wise_catalog[colname] = test_cluster[colname]

# Save the file to disk just in case we need it in the future
wise_catalog.write(f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/'
                   f'large_test_catalog_{test_cluster["SPT_ID"]}.ecsv', overwrite=True)

#%%
# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

#%%
# Magnitude cuts
bright_end_cut = 14.00
faint_end_cut = 16.25

# Bin into 0.25 magnitude bins
mag_bin_width = 0.25
num_counts_mag_bins = np.arange(bright_end_cut, faint_end_cut, mag_bin_width)
mag_bin_centers = num_counts_mag_bins[:-1] + np.diff(num_counts_mag_bins) / 2

# Set the AGN selection color threshold
color_threshold = agn_purity_color(test_cluster['REDSHIFT'])

#%%
# Excise the cluster from the catalog
wise_coords = SkyCoord(wise_catalog['ra'], wise_catalog['dec'], unit=u.deg)
sep_deg = cluster_coord.separation(wise_coords)
sep_mpc = sep_deg * cosmo.kpc_proper_per_arcmin(test_cluster['REDSHIFT']).to(u.Mpc / sep_deg.unit)
wise_catalog['SEP_DEG'] = sep_deg
wise_catalog['SEP_MPC'] = sep_mpc

inner_radius_factor = 3.
inner_radius_mpc = inner_radius_factor * test_cluster['R200']
inner_radius_deg = inner_radius_mpc * cosmo.arcsec_per_kpc_proper(test_cluster['REDSHIFT']).to(u.deg / u.Mpc)

wise_bkg_catalog = wise_catalog[sep_mpc > inner_radius_mpc]

#%%
# Cut the catalog between the magnitude limits
wise_bkg_catalog = wise_bkg_catalog[(bright_end_cut <= wise_bkg_catalog['w2mpro']) & (wise_bkg_catalog['w2mpro'] <= faint_end_cut)]

# Select for AGN
wise_agn_catalog = wise_bkg_catalog[wise_bkg_catalog['w1mpro'] - wise_bkg_catalog['w2mpro'] >= color_threshold]

#%%
# Outer radius options
outer_radii = (np.arange(inner_radius_factor + 1, 30) * test_cluster['R200']
               * cosmo.arcsec_per_kpc_proper(test_cluster['REDSHIFT']).to(u.deg / u.Mpc))

# Calculate the areas for the different radius options
areas = np.pi * (outer_radii**2 - inner_radius_deg**2)

#%%
# Cycle through the outer radius (and associated area) options to create number count distributions
for i, (outer_radius, area) in enumerate(zip(outer_radii, areas)):
    r200_factor = i + 2

    # Filter the catalog with the outer radius
    catalog = wise_agn_catalog[wise_agn_catalog['SEP_DEG'] <= outer_radius]

    # Create histogram
    dn_dm, _ = np.histogram(catalog['w2mpro'], bins=num_counts_mag_bins)
    dn_dm_weighted = dn_dm / (area.value * mag_bin_width)

    # Compute the errors
    dn_dm_err = tuple(err / (area.value * mag_bin_width) for err in small_poisson(dn_dm))[::-1]

    # Create plot
    fig, ax = plt.subplots()
    ax.errorbar(mag_bin_centers, dn_dm_weighted, yerr=dn_dm_err, fmt='.')
    ax.set(title=f'{test_cluster["SPT_ID"][0]}, {outer_radius = :.2f} ({r200_factor} R200)', xlabel='W2 (Vega)',
           ylabel=r'$dN/dm$ [deg$^{-2}$ mag$^{-1}$]',
           yscale='log', ylim=[1, 300])
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/cluster_radii_tests/'
                f'{test_cluster["SPT_ID"][0]}_in{inner_radius_factor:g}r200_out{r200_factor}r200.pdf')
    # plt.show()
    plt.close()
