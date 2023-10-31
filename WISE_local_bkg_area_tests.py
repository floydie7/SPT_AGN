"""
WISE_local_bkg_area_tests.py
Author: Benjamin Floyd

This script will help determine the appropriate area needed for the background annulus.
"""

import json
from functools import partial
from typing import Any

import astropy.cosmology.units as cu
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astro_compendium.utils.small_poisson import small_poisson
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable, hstack, vstack, Table
from colossus.cosmology import cosmology
from colossus.halo.mass_adv import changeMassDefinitionCModel
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048, Tcmb0=2.7255, Neff=3)
colo_cosmo = cosmology.fromAstropy(cosmo, sigma8=0.8, ns=0.96, cosmo_name='concordance')


def log_power_law(x, alpha, beta):
    return alpha + beta * x


def red_chi_sq(ydata, ymodel, n_free, sigma=None):
    if sigma is not None:
        chisq = np.sum(((ydata - ymodel) / sigma) ** 2)
    else:
        chisq = np.sum((ydata - ymodel) ** 2)

    nu = ydata.size - 1 - n_free

    return chisq / nu


# %%
# Read in the SPTcl-IRAGN catalog (this way we only work on cluster in our sample)
sptcl_iragn = QTable.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Group to only have cluster information
sptcl_iragn_grp = sptcl_iragn.group_by('SPT_ID')
sptcl_clusters = vstack([QTable(cluster['SPT_ID', 'SZ_RA', 'SZ_DEC', 'REDSHIFT', 'M500', 'R500'][0])
                         for cluster in sptcl_iragn_grp.groups])

# Add units
sptcl_clusters['M500'].unit = u.Msun

# %%
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

# %%
# # For the test cluster, query an initial large area centered on the cluster
# Irsa.ROW_LIMIT = 1e6
cluster_coord = SkyCoord(test_cluster['SZ_RA'][0], test_cluster['SZ_DEC'][0], unit=u.deg)
# wise_catalog = Irsa.query_region(cluster_coord, catalog='catWISE_2020', spatial='Cone', radius=1 * u.deg,
#                                  selcols='ra,dec,w1mpro,w1sigmpro,w1flux,w1sigflux,'
#                                          'w2mpro,w2sigmpro,w2flux,w2sigflux')
#
# # Merge the cluster information into the WISE catalog
# for colname in test_cluster.colnames:
#     wise_catalog[colname] = test_cluster[colname]
#
# # Save the file to disk just in case we need it in the future
# wise_catalog.write(f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/'
#                    f'large_test_catalog_{test_cluster["SPT_ID"]}.ecsv', overwrite=True)

wise_catalog = Table.read(f'Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/'
                          f'large_test_catalog_{test_cluster["SPT_ID"][0]}.ecsv')

# %%
# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Read in the color threshold--dN/dm model fit parameters
with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/model_fits/SDWFS/'
          'SDWFS-WISE_dNdm_loglin_model_fits.json', 'r') as f:
    sdwfs_dndm_model_params = json.load(f)
dndm_slope_color = interp1d([float(color) for color in sdwfs_dndm_model_params.keys()],
                            [param_data['params'][1] for param_data in sdwfs_dndm_model_params.values()],
                            kind='previous')


def slope(z: float) -> np.ndarray[Any, Any]:
    return dndm_slope_color(agn_purity_color(z))


# %%
# Magnitude cuts
bright_end_cut = 14.00
faint_end_cut = 16.25

# Bin into 0.25 magnitude bins
mag_bin_width = 0.25
num_counts_mag_bins = np.arange(bright_end_cut, faint_end_cut, mag_bin_width)
mag_bin_centers = num_counts_mag_bins[:-1] + np.diff(num_counts_mag_bins) / 2

# Set the AGN selection color threshold
color_threshold = agn_purity_color(test_cluster['REDSHIFT'])

# %%
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

# %%
# Cut the catalog between the magnitude limits
wise_bkg_catalog = wise_bkg_catalog[
    (bright_end_cut <= wise_bkg_catalog['w2mpro']) & (wise_bkg_catalog['w2mpro'] <= faint_end_cut)]

# Select for AGN
# wise_agn_catalog = wise_bkg_catalog[wise_bkg_catalog['w1mpro'] - wise_bkg_catalog['w2mpro'] >= color_threshold]


# Read in the galaxy--AGN fraction data
with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/model_fits/SDWFS/SDWFS-WISE_fAGN.json', 'r') as f:
    sdwfs_f_agn = json.load(f)

# %%
# Outer radius options
outer_radii = (np.arange(inner_radius_factor + 1, 30) * test_cluster['R200']
               * cosmo.arcsec_per_kpc_proper(test_cluster['REDSHIFT']).to(u.deg / u.Mpc))

# Calculate the areas for the different radius options
areas = np.pi * (outer_radii ** 2 - inner_radius_deg ** 2)

# %%
# Cycle through the outer radius (and associated area) options to create number count distributions
for i, (outer_radius, area) in enumerate(zip(outer_radii, areas)):
    r200_factor = inner_radius_factor + 1 + i

    # Filter the catalog with the outer radius
    catalog = wise_bkg_catalog[wise_bkg_catalog['SEP_DEG'] <= outer_radius]

    # Create histogram
    dn_dm, _ = np.histogram(catalog['w2mpro'], bins=num_counts_mag_bins)
    dn_dm_weighted_gal = dn_dm / (area.value * mag_bin_width)

    # Scale the galaxy counts to AGN levels
    dn_dm_weighted_agn = dn_dm_weighted_gal * np.array(sdwfs_f_agn[f'{color_threshold[0]:.2f}'])
    log_dn_dm_weighted = np.log10(dn_dm_weighted_agn)

    # Compute the errors
    dn_dm_err = tuple(err / (area.value * mag_bin_width) for err in small_poisson(dn_dm))[::-1]
    dn_dm_symm_err_gal = np.sqrt(dn_dm_err[0] * dn_dm_err[1])

    # Scale the errors
    # dn_dm_symm_err *= np.array(sdwfs_f_agn[f'{color_threshold[0]:.2f}'])
    frac_err = dn_dm_symm_err_gal / dn_dm_weighted_gal
    dn_dm_symm_err_agn = dn_dm_symm_err_gal * frac_err
    log_dn_dm_symm_err = dn_dm_symm_err_agn / (dn_dm_weighted_agn * np.log(10))

    # Fit the model using a fixed slope from the SDWFS WISE data.
    slope_param = slope(test_cluster['REDSHIFT'])

    fixed_slope_power_law = partial(log_power_law, beta=slope_param)
    param_opt, param_cov = curve_fit(fixed_slope_power_law, mag_bin_centers, log_dn_dm_weighted,
                                     sigma=log_dn_dm_symm_err, maxfev=1_000)
    param_err = np.sqrt(np.diag(param_cov))

    mag_range = np.linspace(faint_end_cut, bright_end_cut, num=200)
    model_fit = fixed_slope_power_law(mag_range, alpha=param_opt[0])
    gof = red_chi_sq(log_dn_dm_weighted, fixed_slope_power_law(mag_bin_centers, param_opt[0]),
                     sigma=log_dn_dm_symm_err, n_free=1)

    # Create plot
    fig, ax = plt.subplots()
    ax.errorbar(mag_bin_centers, log_dn_dm_weighted, yerr=log_dn_dm_symm_err, fmt='.')
    ax.plot(mag_range, model_fit)
    ax.text(0.05, 0.75,
            s=fr'''Model: $\log\left(\frac{{dN}}{{dm}}\right) = \alpha + {slope_param[0]:.2f} m$
            $\alpha = {param_opt[0]:.2f} \pm {param_err[0]:.2f}$
            $\chi^2_{{\nu}} = {gof:.2f}$''', fontsize='large', transform=ax.transAxes)

    ax.set(title=f'{test_cluster["SPT_ID"][0]}, {outer_radius = :.2f} ({r200_factor} R200)', xlabel='W2 (Vega)',
           ylabel=r'$\log(dN/dm)$ [deg$^{-2}$ mag$^{-1}$]', ylim=[1., 2.5], xlim=[bright_end_cut, faint_end_cut])
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/cluster_radii_tests/'
                f'{test_cluster["SPT_ID"][0]}_in{inner_radius_factor:g}r200_out{r200_factor}r200_loglin_models_WISE_slopes_galAGN_constfracerrs.pdf')
    # plt.show()
    plt.close()
