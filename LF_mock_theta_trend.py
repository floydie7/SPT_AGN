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
from scipy.optimize import curve_fit

theta_pattern = re.compile(r'_t(\d+.\d+)_')

# Read in the real catalog for comparison
real_catalog = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

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

# Calcuate the real number count
real_num_ct = np.sum(real_catalog['COMPLETENESS_CORRECTION'] * real_catalog['SELECTION_MEMBERSHIP'])

# Fit curves to both data set
theta_range = np.arange(0.5, 4.0, 0.5)


def trend(theta, a, b):
    return a * theta + b


def inv_trend(num, a, b):
    return (num - b) / a


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
