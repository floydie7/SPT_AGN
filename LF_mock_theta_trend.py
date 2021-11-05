"""
LF_mock_theta_trend.py
Author: Benjamin Floyd

Examines the trend of object number counts in mock catalogs with different values of the cluster amplitude parameter.
"""

import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

theta_pattern = re.compile(r'_t(\d+.\d+)_')

# Read in the real catalog for comparison
spt_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Read in the SDWFS catalog for comparison
sdwfs_agn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')

# Read in the mock catalogs
mock_catalogs = {float(theta_pattern.search(f).group(1)): Table.read(f)
                 for f in glob.glob('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Final_tests/'
                                    'LF_tests/variable_theta/flagged_versions/*.fits')}

# Collect the corrected number counts
num_ct_no_reject = {theta: np.sum(cat['COMPLETENESS_CORRECTION'] * cat['SELECTION_MEMBERSHIP'])
                    for theta, cat in mock_catalogs.items()}
num_ct_reject = {theta: np.sum(cat['COMPLETENESS_CORRECTION'] * cat['SELECTION_MEMBERSHIP'],
                               where=cat['COMPLETENESS_REJECT'].astype(bool))
                 for theta, cat in mock_catalogs.items()}

# Calculate the real number count
real_num_ct = np.sum(spt_iragn['COMPLETENESS_CORRECTION'] * spt_iragn['SELECTION_MEMBERSHIP'])

# %% Fit curves to both data set
theta_range = np.arange(0.5, 4.0, 0.5)

trend = lambda theta, a, b: a * theta + b
inv_trend = lambda num, a, b: (num - b) / a

no_reject_fit, no_reject_cov = curve_fit(trend, theta_range, list(num_ct_no_reject.values()),
                                         sigma=np.sqrt(list(num_ct_no_reject.values())))
reject_fit, reject_cov = curve_fit(trend, theta_range, list(num_ct_reject.values()),
                                   sigma=np.sqrt(list(num_ct_reject.values())))

# Get the true theta values
true_theta_no_reject = inv_trend(real_num_ct, *no_reject_fit)
true_theta_reject = inv_trend(real_num_ct, *reject_fit)
print(f'True theta:\nrejection: {true_theta_reject:.3f}\t no rejection: {true_theta_no_reject:.3f}')

fig, ax = plt.subplots()
ax.scatter(theta_range, list(num_ct_no_reject.values()), c='tab:blue', label='No Rejection Sampling')
ax.plot(theta_range, trend(theta_range, *no_reject_fit), c='tab:blue')
ax.text(0.8, 0.1, rf'$\theta_\mathrm{{true}} = {true_theta_no_reject:.3f}$', c='tab:blue', transform=ax.transAxes)
ax.scatter(theta_range, list(num_ct_reject.values()), c='tab:orange', label='With Rejection Sampling')
ax.plot(theta_range, trend(theta_range, *reject_fit), c='tab:orange')
ax.text(0.8, 0.05, rf'$\theta_\mathrm{{true}} = {true_theta_reject:.3f}$', c='tab:orange', transform=ax.transAxes)
ax.axhline(real_num_ct, ls='--', c='k', label='True Number Count')
ax.legend()
ax.set(xlabel=r'$\theta$', ylabel='Corrected Number Count')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/'
            'LF_variable_theta_suite_flagged_versions.pdf')

# %% Examine the fractional reduction of the rejection sampling
fract_reduct = [num_ct_reject[t] / num_ct_no_reject[t] for t in np.arange(0.5, 4.0, 0.5)]
print(f'Fractional reduction by rejection sampling: {np.mean(fract_reduct):.2f}+-{np.std(fract_reduct):.3f}')

# Examine reductions by object type
cl_objs = {theta: cat[cat['CLUSTER_AGN'].astype(bool)] for theta, cat in mock_catalogs.items()}
bkg_objs = {theta: cat[~cat['CLUSTER_AGN'].astype(bool)] for theta, cat in mock_catalogs.items()}

cl_no_reject = np.array([np.sum(cat['COMPLETENESS_CORRECTION'] * cat['SELECTION_MEMBERSHIP'])
                         for cat in cl_objs.values()])
bkg_no_reject = np.array([np.sum(cat['COMPLETENESS_CORRECTION'] * cat['SELECTION_MEMBERSHIP'])
                          for cat in bkg_objs.values()])
cl_reject = np.array([np.sum(cat['COMPLETENESS_CORRECTION'] * cat['SELECTION_MEMBERSHIP'],
                             where=cat['COMPLETENESS_REJECT'].astype(bool))
                      for cat in cl_objs.values()])
bkg_reject = np.array([np.sum(cat['COMPLETENESS_CORRECTION'] * cat['SELECTION_MEMBERSHIP'],
                              where=cat['COMPLETENESS_REJECT'].astype(bool))
                       for cat in bkg_objs.values()])
print(f'Fractional reduction by rejection sampling\n'
      f'Cluster Objects: {np.mean(cl_reject / cl_no_reject):.2f}+-{np.std(cl_reject / cl_no_reject):.4f}\n'
      f'Background Objects: {np.mean(bkg_reject / bkg_no_reject):.2f}+-{np.std(bkg_reject / bkg_no_reject):.4f}')

# Examine the color-redshift distribution of the theta=2.5 mock
mock_25 = mock_catalogs[2.5][mock_catalogs[2.5]['COMPLETENESS_REJECT'].astype(bool)]
mock_25_cl = mock_25[mock_25['CLUSTER_AGN'].astype(bool)]
mock_25_bkg = mock_25[~mock_25['CLUSTER_AGN'].astype(bool)]
cluster_z = np.array([grp['REDSHIFT'][0] for grp in mock_25.group_by('SPT_ID').groups])

# %%
fig, ax = plt.subplots()
ax.hexbin(mock_25_cl['REDSHIFT'], mock_25_cl['I1_I2'], gridsize=50, mincnt=1, extent=(0., 1.8, 0.25, 1.7),
          cmap='Reds', bins='log')
ax.hexbin(mock_25_bkg['REDSHIFT'], mock_25_bkg['I1_I2'], gridsize=50, mincnt=1, extent=(0., 1.8, 0.25, 1.7),
          cmap='Blues', bins='log', alpha=0.6)
ax.vlines(cluster_z, ymin=0., ymax=0.05, colors='tab:green', transform=ax.get_xaxis_transform(),
          label='Cluster Redshifts')
handles, labels = ax.get_legend_handles_labels()
handles.extend([Line2D([0], [0], marker='h', color='none', markerfacecolor='lightcoral', markeredgecolor='firebrick',
                       markersize=10),
                Line2D([0], [0], marker='h', color='none', markerfacecolor='lightblue', markeredgecolor='steelblue',
                       markersize=10)])
labels.extend(['Cluster Objects', 'Background Objects'])
ax.legend(handles, labels, loc='upper left', frameon=False)
ax.set(title=r'Mock Catalog $\theta=2.5$ (Rejection Sampled)', xlabel='Cluster Redshift', ylabel='[3.6] - [4.5] (Vega)',
       ylim=[0.25, 1.7])
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/'
#             'LF_variable_flagged_mock_t2.5_color_zcl.pdf')
plt.show()

fig, ax = plt.subplots()
ax.hexbin(mock_25_cl['OBJ_REDSHIFT'], mock_25_cl['I1_I2'], gridsize=50, mincnt=1, extent=(0., 1.8, 0.25, 1.7),
          cmap='Reds', bins='log')
ax.hexbin(mock_25_bkg['OBJ_REDSHIFT'], mock_25_bkg['I1_I2'], gridsize=50, mincnt=1, extent=(0., 1.8, 0.25, 1.7),
          cmap='Blues', bins='log', alpha=0.6)
ax.vlines(cluster_z, ymin=0., ymax=0.05, colors='tab:green', transform=ax.get_xaxis_transform(),
          label='Cluster Redshifts')
handles, labels = ax.get_legend_handles_labels()
handles.extend([Line2D([0], [0], marker='h', color='none', markerfacecolor='lightcoral', markeredgecolor='firebrick',
                       markersize=10),
                Line2D([0], [0], marker='h', color='none', markerfacecolor='lightblue', markeredgecolor='steelblue',
                       markersize=10)])
labels.extend(['Cluster Objects', 'Background Objects'])
ax.legend(handles, labels, loc='upper left', frameon=False)
ax.set(title=r'Mock Catalog $\theta=2.5$ (Rejection Sampled)', xlabel='Object Redshift', ylabel='[3.6] - [4.5] (Vega)',
       ylim=[0.25, 1.7])
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/'
#             'LF_variable_flagged_mock_t2.5_color_zobj.pdf')
plt.show()

# %% Compare the background objects with SDWFS
color_bins = np.arange(0.4, 1.7, 0.1)
fig, ax = plt.subplots()
ax.hist(sdwfs_agn['I1_MAG_APER4'] - sdwfs_agn['I2_MAG_APER4'], bins=color_bins,
        weights=((sdwfs_agn['COMPLETENESS_CORRECTION'] * sdwfs_agn['SELECTION_MEMBERSHIP'])
                 / len(sdwfs_agn.group_by('CUTOUT_ID').groups.keys)), label='SDWFS Objects')
ax.hist(mock_25_bkg['I1_I2'], bins=color_bins, alpha=0.5,
        weights=((mock_25_bkg['COMPLETENESS_CORRECTION'] * mock_25_bkg['SELECTION_MEMBERSHIP'])
                 / len(mock_25.group_by('SPT_ID').groups.keys)),
        label='Mock Background Objects')
ax.legend()
ax.set(title=r'Mock Catalog $\theta=2.5$ (Rejection Sampled)', xlabel='[3.6] - [4.5] (Vega)',
       ylabel='Corrected Number per Cutout')
plt.show()

# %% Compare the full mock dataset with the SPTcl-IRAGN catalog
fig, ax = plt.subplots()
ax.hist(sdwfs_agn['I1_MAG_APER4'] - sdwfs_agn['I2_MAG_APER4'], bins=color_bins,
        # weights=((sdwfs_agn['COMPLETENESS_CORRECTION'] * sdwfs_agn['SELECTION_MEMBERSHIP'])
        #          / len(sdwfs_agn.group_by('CUTOUT_ID').groups.keys)),
        weights=np.full_like(sdwfs_agn['I1_MAG_APER4'] - sdwfs_agn['I2_MAG_APER4'],
                             1 / len(sdwfs_agn.group_by('CUTOUT_ID').groups.keys)),
        label='SDWFS Objects', histtype='bar')
ax.hist(spt_iragn['I1_MAG_APER4'] - spt_iragn['I2_MAG_APER4'], bins=color_bins,
        # weights=((spt_iragn['COMPLETENESS_CORRECTION'] * spt_iragn['SELECTION_MEMBERSHIP'])
        #          / len(spt_iragn.group_by('SPT_ID').groups.keys)),
        weights=np.full_like(spt_iragn['I1_MAG_APER4'] - spt_iragn['I2_MAG_APER4'],
                             1 / len(spt_iragn.group_by('SPT_ID').groups.keys)),
        label='SPTcl-IRAGN', alpha=0.7)
ax.hist([mock_25_bkg['I1_I2']], bins=color_bins,
        # weights=[((mock_25_bkg['COMPLETENESS_CORRECTION'] * mock_25_bkg['SELECTION_MEMBERSHIP'])
        #           / len(mock_25.group_by('SPT_ID').groups.keys)),
        #          ((mock_25_cl['COMPLETENESS_CORRECTION'] * mock_25_cl['SELECTION_MEMBERSHIP'])
        #           / len(mock_25.group_by('SPT_ID').groups.keys))],
        weights=np.full_like(mock_25_bkg['I1_I2'], 1 / len(mock_25.group_by('SPT_ID').groups.keys)),
        label=['Mock Background Objects', 'Mock Cluster Objects'],
        stacked=True, histtype='step', lw=2)
ax.legend()
ax.set(title=r'Mock Catalog $\theta=2.5$ (Rejection Sampled)', xlabel='[3.6] - [4.5] (Vega)',
       ylabel='Corrected Number per Cutout')
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/'
#             'LF_variable_flagged_mock_t2.5_color_hist_per_img_compare.pdf')
plt.show()

#%% Explore the trends between the color (or color-error) and the selection membership
fig, ax = plt.subplots()
ax.scatter(spt_iragn['I1_MAG_APER4'] - spt_iragn['I2_MAG_APER4'], spt_iragn['SELECTION_MEMBERSHIP'],
           marker='.', label='SPTcl', alpha=0.2, color='tab:blue')
ax.scatter(sdwfs_agn['I1_MAG_APER4'] - sdwfs_agn['I2_MAG_APER4'], sdwfs_agn['SELECTION_MEMBERSHIP'],
           marker='.', label='SDWFS', alpha=0.2, color='tab:orange')
ax.scatter(mock_25['I1_I2'], mock_25['SELECTION_MEMBERSHIP'],
           marker='.', label='Mock (CL+BKG)', alpha=0.2, color='tab:green')
ax.axvline(0.7, ls='--', c='k')
ax.legend()
ax.set(title=r'Mock Catalog $\theta=2.5$', xlabel='[3.6] - [4.5]', ylabel=r'$\mu_\mathrm{AGN}$', xlim=[0.4, 1.6])
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/Catalog_comparisons/'
            'LF_variable_flagged_mock_t2.5_mu_agn_color_all.pdf')
plt.show()

fig, ax = plt.subplots()
ax.scatter(mock_25_cl['I1_I2'], mock_25_cl['SELECTION_MEMBERSHIP'], marker='.', label='Mock (CL)', alpha=0.3,
           color='tab:red')
ax.scatter(mock_25_bkg['I1_I2'], mock_25_bkg['SELECTION_MEMBERSHIP'],
           marker='.', label='Mock (BKG)', alpha=0.3, color='tab:purple')
ax.axvline(0.7, ls='--', c='k')
ax.legend()
ax.set(title=r'Mock Catalog $\theta=2.5$', xlabel='[3.6] - [4.5]', ylabel=r'$\mu_\mathrm{AGN}$', xlim=[0.4, 1.6])
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/Catalog_comparisons/'
            'LF_variable_flagged_mock_t2.5_mu_agn_color_mock-only.pdf')
plt.show()
