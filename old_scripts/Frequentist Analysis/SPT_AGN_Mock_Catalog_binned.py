"""
SPT_AGN_Mock_Catalog_binned.py
Author: Benjamin Floyd

Diagnostics on the mock AGN catalog and related subcatalogs.
"""

from astropy.table import Table, vstack, join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from itertools import product
import astropy.units as u
from small_poisson import small_poisson

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)


def observed_surf_den(catalog):
    # Group the catalog by cluster
    cat_grped = catalog.group_by('SPT_ID')

    total_surf_den = []
    for cluster in cat_grped.groups:
        # Create a dictionary to store the relevant data.
        cluster_dict = {'SPT_ID': [cluster['SPT_ID'][0]]}

        # Sum over all the completeness corrections for all AGN within our selected aperture.
        cluster_counts = len(cluster)

        # Calculate the 1-sigma Poisson errors for the AGN observed.
        cluster_err = small_poisson(cluster_counts, s=1)

        # Using the area enclosed by our aperture calculate the surface density.
        cluster_surf_den = cluster_counts / 1.

        # Also convert our errors to surface densities.
        cluster_surf_den_err_upper = cluster_err[0] / 1.
        cluster_surf_den_err_lower = cluster_err[1] / 1.

        cluster_dict.update({'obs_surf_den': [cluster_surf_den],
                             'obs_upper_surf_den_err': [cluster_surf_den_err_upper],
                             'obs_lower_surf_den_err': [cluster_surf_den_err_lower]})
        total_surf_den.append(Table(cluster_dict))

    # Combine all cluster tables into a single table to return
    surf_den_table = vstack(total_surf_den)

    # Calculate the variance of the errors.
    surf_den_table['obs_surf_den_var'] = (surf_den_table['obs_upper_surf_den_err']
                                          * surf_den_table['obs_lower_surf_den_err'])

    return surf_den_table


def chi_sq(catalog, obs_surf_den_table, nsteps):
    # Generate posible values for the parameters
    eta = np.linspace(-3., 3., nsteps)
    zeta = np.linspace(-3., 3., nsteps)
    beta = np.linspace(-3., 3., nsteps)

    # Create the list of all possible values of (eta, zeta, beta)
    params = list(product(eta, zeta, beta))

    chi_sq_results = []
    for param_set in params:
        # Extract our parameters
        eta, zeta, beta = param_set

        catalog_grp = catalog.group_by('SPT_ID')

        # For each cluster determine the model value and assign it to the correct observational value
        model_tables = []
        for cluster in catalog_grp.groups:
            # Find the number of AGN in the cluster. This will be the sum of the completeness values in a real data set.
            n_agn = len(cluster)

            # Calculate the model value for the cluster
            model_value = 1. / n_agn * ((1 + cluster['REDSHIFT'][0]) ** eta
                                        * ((cluster['M500'][0] * u.Msun) / (1e15 * u.Msun)) ** zeta
                                        * np.sum((cluster['r_r500_radial']) ** beta))

            # Store the model values in a table.
            cluster_dict = {'SPT_ID': [cluster['SPT_ID'][0]], 'model_values': [model_value]}
            model_tables.append(Table(cluster_dict))

        # Combine all the model tables into a single table.
        model_table = vstack(model_tables)

        # Join the observed and model tables together based on the SPT_ID keys
        joint_table = join(obs_surf_den_table, model_table, keys='SPT_ID')

        # Our likelihood is then the chi-squared likelihood.
        chi_sq_val = np.sum((joint_table['obs_surf_den'] - joint_table['model_values']) ** 2
                            / joint_table['obs_surf_den_var'])

        chi_sq_results.append(chi_sq_val)

    # Find the minimum chi-squared value
    idx = np.argmin(chi_sq_results)
    min_chisq = chi_sq_results[idx]
    min_params = params[idx]

    print(min_chisq, min_params)

    return params, chi_sq_results, min_chisq


# Read in the full mock catalog
full_mock = Table.read('Data/MCMC/Mock_Catalog/Catalogs/mock_AGN_catalog.cat', format='ascii')
full_mock['M500'].unit = u.Msun

# Read in the subcatalogs
subcats = [Table.read('Data/MCMC/Mock_Catalog/Catalogs/'+f, format='ascii')
           for f in os.listdir('Data/MCMC/Mock_Catalog/Catalogs/') if 'subcatalog' in f]

# Plot the marginalized distributions of the three data axes.
fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True)
ax0.hist(1 + full_mock['REDSHIFT'], bins=50)
ax0.set(xlabel='$1+z$')

ax1.hist(full_mock['M500']/1e15, bins=50)
ax1.set(xlabel=r'$M_{500}/ 10^{15} M_\odot$')

ax2.hist(full_mock['r_r500_radial'], bins=50)
ax2.set(xlabel=r'$r /r_{500}$')

plt.tight_layout()

# Plot the marginalized distributions of the three data axes.
fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True)
ax0.hist(1 + subcats[0]['REDSHIFT'], bins=50)
ax0.set(xlabel='$1+z$')

ax1.hist(subcats[0]['M500']/1e15, bins=50)
ax1.set(xlabel=r'$M_{500}/ 10^{15} M_\odot$')

ax2.hist(subcats[0]['r_r500_radial'], bins=50)
ax2.set(xlabel=r'$r /r_{500}$')

plt.tight_layout()

# cluster = subcats[0].group_by('SPT_ID').groups[0]
# fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True)
# ax0.hist(1 + cluster['REDSHIFT'], bins=50)
# ax0.set(xlabel='$1+z$')
#
# ax1.hist(cluster['M500']/1e15, bins=50)
# ax1.set(xlabel=r'$M_{500}/ 10^{15} M_\odot$')
#
# ax2.hist(cluster['r_r500_radial'], bins=50)
# ax2.set(xlabel=r'$r /r_{500}$')
#
# plt.tight_layout()
#
plt.show()

#
# # Calculate the "observed" surface density and variance
# obs_cluster_surf_den = observed_surf_den(subcats[0])
#
# # Calculate the chi-squared values
# parameters, chi_squared, min_chi_squared= chi_sq(subcats[0], obs_cluster_surf_den, 5)
#
# # Transpose the parameter array to isolate lists for each parameter
# parameters = np.transpose(parameters)
# eta_param = parameters[0]
# zeta_param = parameters[1]
# beta_param = parameters[2]
#
# # Create a meshgrid to use for plotting
# eta_grid, zeta_grid, beta_grid = np.meshgrid(eta_param, zeta_param, beta_param)
#
# # Set contour levels (from Avni 1976)
# levels = [min_chi_squared+2.30, min_chi_squared+4.61, min_chi_squared+9.21]
#
# # Make plot.
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# cs = ax.contour3D(eta_grid, zeta_grid, chi_squared, cmap='binary')
# # plt.clabel(cs, inline=1, manual=['68%', '90%', '99%'])
# ax.set(xlabel=r'$\eta$', ylabel=r'$\zeta$', zlabel=r'$\chi^2$')
# # plt.savefig('Data/MCMC/Mock_Catalog/Plots/Chi_Sq_Contour.pdf', format='pdf')
# plt.show()
