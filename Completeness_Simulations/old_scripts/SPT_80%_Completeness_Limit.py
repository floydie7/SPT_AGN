"""
SPT_80%_Completeness_Limit.py
Author: Benjamin Floyd

This script determines the magnitude at which the SPT clusters are 80% complete.
"""

from __future__ import print_function, division

import numpy as np
from scipy.interpolate import interp1d

# Read in the completeness curves for Ch2.
ch2_rates = np.load('Data/Comp_Sim/Results/SPT_I2_results_gaussian_fwhm202_corr011_mag02.npy').item()

# Extract the magnitude bins from the dictionary and shift the data point to the center of the bin.
ch2_mag_bins = ch2_rates.pop('magnitude_bins')[:-1]
ch2_mag_bins += 0.25

# Build a list of all magnitudes along the 80% completeness line.
mag_80 = []
for key in ch2_rates:
    data = ch2_rates[key]

    # Interpolate the completeness inverse function.
    mag_funct = interp1d(data, ch2_mag_bins, kind='linear')

    # Query the magnitude corresponding to the 80% completeness level.
    mag_80.append(mag_funct(0.8))

# Print the statistics
print('mean: {0}, std dev: {1}, median: {2}'.format(np.mean(mag_80), np.std(mag_80), np.median(mag_80)))
print('Bright-end (95% limit) magnitude: {0}'.format(np.percentile(mag_80, 5)))
print('Faint-end (95% limit) magnitude: {0}'.format(np.percentile(mag_80, 95)))


