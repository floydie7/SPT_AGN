"""
mock_snr_theta_trend.py
Author: Benjamin Floyd

Makes a diagnostic plot to see SNR-theta trends
"""

import glob
import json
import re
from typing import Any

import numpy as np
import pandas as pd
from astropy.table import Table
from matplotlib import colors, cm, pyplot as plt
from numpy.polynomial import Polynomial


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


param_pattern = re.compile(r'(?:[tezbCx]|rc)(-*\d+.\d+|\d+)')
eta_zeta_snr_pattern = re.compile(r'(?:[ez]|snr)(-*\d+.\d+|\d+)')
theta_eta_zeta_snr_pattern = re.compile(r'(?:[tez]|snr)(-*\d+.\d+|\d+)')

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]

catalog_list = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/'
                         'eta_zeta_slopes/raw_grid/308cl_catalogs/*_tez_grid.fits')

eta_range = [-5., -3., 0., 3., 4., 5.]
zeta_range = [-2., -1., 0., 1., 2.]
eta_zeta_grid = np.array(np.meshgrid(eta_range, zeta_range)).T.reshape(-1, 2)

tez_snr_data = []
for catalog_file in catalog_list:
    # Extract the parameter set
    params = np.array(param_pattern.findall(catalog_file), dtype=float)
    # Read in the catalog
    catalog = Table.read(catalog_file)

    # Bin the catalog by redshift
    catalog_binned = catalog.group_by(np.digitize(catalog['REDSHIFT'], z_bins))

    redshift_bin_snr, redshift_bin_var = [], []
    bkg_means, bkg_errs = [], []
    for redshift_bin in catalog_binned.groups:
        # Separate the catalog into cluster and background objects
        cluster_agn = redshift_bin[redshift_bin['CLUSTER_AGN'].astype(bool)]
        background_agn = redshift_bin[~redshift_bin['CLUSTER_AGN'].astype(bool)]

        # Compute the corrected number of objects in the catalogs
        num_cl_agn = cluster_agn['COMPLETENESS_CORRECTION'].sum()

        # Find the mean and standard deviation of the background AGN distribution in this redshift bin
        background_agn_mean = background_agn['COMPLETENESS_CORRECTION'].mean()
        background_agn_std = background_agn['COMPLETENESS_CORRECTION'].std()
        bkg_means.append(background_agn_mean)
        bkg_errs.append(background_agn_std)

        # Set the field error to be the field-to-field standard deviation
        field_err = background_agn_std

        # Reduce the field error by the number of clusters present in the redshift bin
        field_err /= np.sqrt(len(redshift_bin.group_by('SPT_ID').groups))

        # Compute the SNR for the redshift bin and variance
        redshift_bin_snr.append(num_cl_agn / field_err)
        redshift_bin_var.append(field_err ** 2)
    redshift_bin_snr = np.asarray(redshift_bin_snr)
    redshift_bin_var = np.asarray(redshift_bin_var)

    # Combine the SNRs for the redshift bins into a single catalog value by inverse variance weighting
    catalog_snr = np.sum(redshift_bin_snr / redshift_bin_var) / np.sum(1 / redshift_bin_var)
    catalog_snr_err = np.sum(1 / redshift_bin_var) ** -0.5

    theta, eta, zeta = params[:3]
    tez_snr_data.append([theta, eta, zeta, catalog_snr, catalog_snr_err])

    # Make diagnostic plot showing the distributions of the backgrounds
    fig, ax = plt.subplots()
    ax.errorbar(z_bins, bkg_means, yerr=bkg_errs, marker='.')
    ax.set(xlabel='redshift', ylabel=r'$N_{bkg}$', title=fr'$(\theta, \eta, \zeta) = ({theta}, {eta}, {zeta})')
    plt.show()

tez_snr_df = pd.DataFrame(tez_snr_data, columns=['theta', 'eta', 'zeta', 'snr', 'snr_err'])
tez_snr_df_grp = tez_snr_df.groupby(by=['eta', 'zeta'])

# Create extrapolations of our curves
funct_lib = {}
fit_lib = {}
for name, group in tez_snr_df_grp:
    poly: Polynomial = Polynomial.fit(group['snr'], group['theta'], 1)
    fit_lib[str(name)] = poly.convert().coef
    funct_lib[name] = poly

# with open('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/eta_zeta_slopes/'
#           'snr_to_theta_fits.json', 'w') as f:
#     json.dump(fit_lib, f, cls=NumpyArrayEncoder)

cNorm = colors.Normalize(vmin=0, vmax=len(tez_snr_df_grp) - 1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap='jet')

# %%
target_snr = 5.83
targeted_theta = Table(rows=[[str(name), theta_snr(target_snr)] for name, theta_snr in funct_lib.items()],
                       names=['catalog', 'theta'])

# Save the table to file
# targeted_theta.write('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/'
#                      f'eta_zeta_slopes/targeted_snr/308cl/308cl_targeted_snr{target_snr:.2f}_theta_values.fits',
#                      overwrite=True)

# %%
theta_range = np.arange(0., 6.1, 0.01)
fig, ax = plt.subplots()
ax.hist(targeted_theta['theta'], bins=theta_range)
ax.set(xlabel=r'$\theta$', ylabel='Number of Catalogs', xlim=[0., 0.1])
plt.show()

# %%
snr_range = np.linspace(0., 15., num=100)
fig, ax = plt.subplots(figsize=(1.25 * 6.8, 1.25 * 4.8))
ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(len(funct_lib))])
for (name, theta_snr), (_, group) in zip(funct_lib.items(), tez_snr_df_grp):
    sorted_grp = group.sort_values('theta')
    ax.scatter(sorted_grp['theta'], sorted_grp['snr'], marker='.')
    ax.plot(theta_snr(snr_range), snr_range, label=name)
ax.axhline(y=5.83, ls='--', c='k')
ax.legend(loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2), borderaxespad=0.5, ncol=6)
ax.set(xlabel=r'$\theta$', ylabel='SNR', ylim=[0, 10], xlim=[0, 2])
plt.tight_layout()
plt.show()
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Port_Rebuild_Tests/eta-zeta_grid/'
#             f'308cl_snr_theta_trend_snr{target_snr:.2f}.pdf', bbox_inches='tight')
