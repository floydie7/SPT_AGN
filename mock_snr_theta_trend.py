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
    # Separate the cluster and background objects
    cluster_agn = catalog[catalog['CLUSTER_AGN'].astype(bool)]
    background_agn = catalog[~catalog['CLUSTER_AGN'].astype(bool)]
    # Compute the corrected number of objects in the catalogs
    num_cl_agn = cluster_agn['COMPLETENESS_CORRECTION'].sum()
    num_bkg_agn = background_agn['COMPLETENESS_CORRECTION'].sum()
    # Signal-to-Noise ratio is defined as the numbers of cluster / background AGN
    snr = num_cl_agn / num_bkg_agn

    theta, eta, zeta = params[:3]
    tez_snr_data.append([theta, eta, zeta, snr])

tez_snr_df = pd.DataFrame(tez_snr_data, columns=['theta', 'eta', 'zeta', 'snr'])
tez_snr_df_grp = tez_snr_df.groupby(by=['eta', 'zeta'])

# Create extrapolations of our curves
funct_lib = {}
fit_lib = {}
for name, group in tez_snr_df_grp:
    fit = np.polyfit(group['snr'], group['theta'], 1)
    fit_lib[str(name)] = fit
    funct_lib[name] = np.poly1d(fit)

with open(
        'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/eta_zeta_slopes/snr_to_theta_fits.json',
        'w') as f:
    json.dump(fit_lib, f, cls=NumpyArrayEncoder)

cNorm = colors.Normalize(vmin=0, vmax=len(tez_snr_df_grp) - 1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap='jet')

# %%
fig, ax = plt.subplots(figsize=(6.8, 1.5 * 4.8))
ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(len(tez_snr_df_grp))])
for name, group in tez_snr_df_grp:
    sorted_grp = group.sort_values('theta')
    ax.plot(sorted_grp['theta'], sorted_grp['snr'], label=name)
# ax.axhline(y=4.0217, ls='--', c='k', alpha=0.8)
ax.axhline(y=13, ls='--', c='k')
# ax.fill_between(x=np.linspace(0, 149.6), y1=0.5998, y2=7.0474, color='lightgrey')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=4)
ax.set(xlabel='theta', ylabel='SNR', yscale='linear', xlim=[0, 1000], ylim=[0, 15])
plt.tight_layout()
plt.show()

# %%
target_snr = 0.23
targeted_theta = Table(rows=[[str(name), theta_snr(target_snr)] for name, theta_snr in funct_lib.items()],
                       names=['catalog', 'theta'])

# Save the table to file
targeted_theta.write('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/'
                     f'eta_zeta_slopes/targeted_snr/308cl/308cl_targeted_snr{target_snr:.2f}_theta_values.fits',
                     overwrite=True)

# %%
snr_range = np.linspace(0., 15., num=100)
fig, ax = plt.subplots(figsize=(6.8, 1.5 * 4.8))
ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(len(funct_lib))])
for name, theta_snr in funct_lib.items():
    ax.plot(theta_snr(snr_range), snr_range, label=name)
ax.axhline(y=13, ls='--', c='k')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=4)
ax.set(xlabel='theta', ylabel='SNR', ylim=[0, 15], xlim=[0, 1000])
plt.tight_layout()
plt.show()
