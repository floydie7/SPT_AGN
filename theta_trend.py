"""
theta_trend.py
Author: Benjamin Floyd

For the signal-to-noise testing, find the trend of theta vs number of objects in the various catalogs then find the
correct value of theta the corresponds to the true number in the real data set.
"""

import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from astropy.table import Table, vstack

# Read in the real data catalog
real_catalog = vstack([Table.read('Data/Output/SPT_IRAGN.fits'), Table.read('Data/Output/SPTPol_IRAGN.fits')])

# Get all directories
mock_dirs = glob.glob('Data/MCMC/Mock_Catalog/Catalogs/Final_tests/Slope_tests/trial_1/*/')

for directory in mock_dirs:
    # Get the label for this mock catalog set from the directory name
    mock_set_label = re.search(r'/(e\d+(?:\.\d+)_z-?\d+(?:\.\d+))', directory).group(1)

    # Grab the filenames of all the mock catalogs
    mock_fnames = glob.glob(f'{directory}*.cat')

    # Read in the mock catalogs
    mock_catalogs = {float(re.search(r'_t(\d+(?:\.\d+))_', filename).group(1)): Table.read(filename, format='ascii')
                     for filename in mock_fnames}

    # Make the lists of theta values and number of objects in each catalog
    theta_values = np.array(list(mock_catalogs.keys()))
    number_of_objs = np.array([len(catalog) for catalog in mock_catalogs.values()])

    # Fit a linear model to the data
    theta_trend = lambda theta, a, b: a * theta + b
    trend_fit, trend_cov = op.curve_fit(theta_trend, theta_values, number_of_objs, sigma=np.sqrt(number_of_objs))

    # Build an interpolation of our trend
    # inverse_theta = interp1d(number_of_objs, theta_values)
    inverse_theta = lambda n, a, b: (n - b) / a

    # Find the real data theta value
    # real_theta = inverse_theta(len(real_catalog))
    real_theta_comp = inverse_theta(real_catalog['COMPLETENESS_CORRECTION'].sum(), *trend_fit)

    fig, ax = plt.subplots()
    ax.scatter(theta_values, number_of_objs, label='Mock catalogs')
    # ax.scatter(real_theta, len(real_catalog), marker='*', color='C1', facecolor='none', label='Real catalog')
    ax.scatter(real_theta_comp, real_catalog['COMPLETENESS_CORRECTION'].sum(), marker='*', color='C1',
               label='Real catalog (completeness corrected)')
    x = np.linspace(np.min(theta_values), np.max(theta_values), num=100)
    ax.plot(x, theta_trend(x, *trend_fit), label='Fit trend')
    ax.set(xlabel=r'$\theta$', ylabel='Number of objects', title=f'Mock Set: {mock_set_label}')

    # axins = ax.inset_axes([0.1, 0.55, 0.4, 0.4])
    # axins.scatter(theta_values, number_of_objs)
    # # axins.scatter(real_theta, len(real_catalog), marker='*', color='C1', facecolor='none')
    # axins.scatter(real_theta_comp, real_catalog['COMPLETENESS_CORRECTION'].sum(), marker='*', color='C1')
    # axins.set(xlim=[0., 0.25], ylim=[2300, 4000])
    # ax.indicate_inset_zoom(axins, label=None)

    ax.legend(loc='lower right')
    fig.savefig(
        'Data/MCMC/Mock_Catalog/Plots/Final_tests/Slope_tests/trial_1/catalog_set_theta_trends/'
        f'mock_catalog_theta_trend_{mock_set_label}.pdf',
        format='pdf')
    # plt.show()

    # print('theta for real catalog: {:.3f}'.format(real_theta))
    print(f'{mock_set_label}: theta for real catalog (completeness corrected): {real_theta_comp:.3f}')
