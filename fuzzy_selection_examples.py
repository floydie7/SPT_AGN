"""
fuzzy_selection_examples.py
Author: Benjamin Floyd

Creates plots illustrating the fuzzy selection process that includes the Eddington bias correction.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from scipy.stats import norm

# Read in the SDWFS background number counts file
sdwfs_field_number_dist = 'Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/' \
                          'SDWFS_number_count_distribution_normed.json'
with open(sdwfs_field_number_dist, 'r') as f:
    field_number_distribution = json.load(f)
field_number_counts = field_number_distribution['normalized_number_counts']
color_bins = field_number_distribution['color_bins']
color_bin_min, color_bin_max = np.min(color_bins), np.max(color_bins)

# Create an interpolation of our number count distribution
color_probability_distribution = interp1d(color_bins, field_number_counts)

# Read in the SPTcl-IRAGN catalog
sptcl = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

# Arbitrarily we will pick SPT-CLJ2314-5544 as an example
spt2314 = sptcl[sptcl['SPT_ID'] == 'SPT-CLJ2314-5554']

# From SPT2314, pick an AGN
example_agn = spt2314[7]

# Get the color and color error of the AGN
I1_I2_color = example_agn['I1_MAG_APER4'] - example_agn['I2_MAG_APER4']
I1_I2_color_err = np.sqrt((2.5 * example_agn['I1_FLUXERR_APER4'] / (np.log(10) * example_agn['I1_FLUX_APER4'])) ** 2
                          + (2.5 * example_agn['I2_FLUXERR_APER4'] / (np.log(10) * example_agn['I2_FLUX_APER4'])) ** 2)


# Convolve the error distribution for each object with the overall number count distribution
def object_integrand(x):
    return norm(loc=I1_I2_color, scale=I1_I2_color_err).pdf(x) * color_probability_distribution(x)


# Compute the normalization
color_prob_in_denom = quad(object_integrand, a=color_bin_min, b=color_bin_max, limit=int(1e5))[0]

#%% Find the new mean and stddev
xx = np.linspace(color_bin_min, color_bin_max, num=500)
corrected_mean = np.average(xx, weights=object_integrand(xx) / color_prob_in_denom)
corrected_std = np.sqrt(np.average((xx - corrected_mean)**2, weights=object_integrand(xx) / color_prob_in_denom))

#%% Make the plot
fig, ax = plt.subplots()
ax.bar(color_bins, field_number_counts, width=np.diff(color_bins)[0], label='SDWFS Number Counts')
ax.plot(xx, norm(loc=I1_I2_color, scale=I1_I2_color_err).pdf(xx), ls='--', c='C1',
        label=rf'Original: $\mu={I1_I2_color:.2f}, \sigma={I1_I2_color_err:.2f}$')
ax.plot(xx, object_integrand(xx) / color_prob_in_denom, c='C1',
        label=rf'Corrected: $\mu={corrected_mean:.2f}, \sigma={corrected_std:.2f}$')
ax.axvline(0.7, ls='--', c='k')
ax.fill_between(xx, y1=0, y2=object_integrand(xx) / color_prob_in_denom, where=(xx >= 0.7), interpolate=True, alpha=0.4,
                label=f"Degree of Membership = {spt2314['SELECTION_MEMBERSHIP'][7]:.2f}")
ax.legend(loc='upper left', frameon=False)
ax.set(xlabel=r'[3.6 $\mu$m] - [4.5 $\mu$m] (Vega)', xlim=[-0.33, 1.5])
ax.yaxis.set_visible(False)
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/fuzzy_selection_example_SPT2314_AGN7.pdf')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/fuzzy_selection_example_SPT2314_AGN7.png')
