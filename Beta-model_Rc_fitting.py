"""
Beta-model_Rc_fitting.py
Author: Benjamin Floyd

Fits a model to core radii fits obtained from Mohr et al. (1999).
"""

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from scipy import stats
# We need to convert the h_50^-1 scaling from the data to h-less units.
H0 = 70
little_h_50 = H0 / 50

# Core radius data from Mohr+99 Table 2, single component beta model fits only.
rc_mean_h50 = np.array([0.483, 0.237, 0.176, 0.367, 0.360, 0.160, 0.386, 0.131, 0.117, 0.584, 0.486, 0.213, 0.044,
                        0.049, 0.262, 0.495, 0.216, 0.246, 0.194, 0.097, 0.173, 0.258, 0.056, 0.0078, 0.056, 0.015,
                        0.056])
rc_uerr_h50 = np.array([0.028, 0.016, 0.035, 0.324, 0.046, 0.009, 0.026, 0.022, 0.047, 0.046, 0.042, 0.070, 0.007,
                        0.022, 0.034, 0.040, 0.045, 0.073, 0.028, 0.006, 0.016, 0.026, 0.012, 0.011, 0.006, 0.049,
                        0.015])
rc_lerr_h50 = np.array([0.028, 0.019, 0.034, 0.309, 0.043, 0.009, 0.026, 0.014, 0.031, 0.039, 0.041, 0.069, 0.006,
                        0.021, 0.031, 0.038, 0.038, 0.057, 0.027, 0.007, 0.015, 0.025, 0.011, 0.011, 0.006, 0.015,
                        0.028])

# Calculate the 1-sigma range of our data
rc_quants_h50 = np.quantile(rc_mean_h50, [0.16, 0.5, 0.84])

fig, ax = plt.subplots()
ax.errorbar(rc_mean_h50, np.arange(0, len(rc_mean_h50)),  xerr=np.array([rc_lerr_h50, rc_uerr_h50]), fmt='o')
ax.axvline(x=rc_quants_h50[1], linestyle='--', color='k')
ax.axvspan(rc_quants_h50[0], rc_quants_h50[2], alpha=0.4, color='gray')
ax.tick_params(axis='y', which='both', left=False, labelleft=False)
ax.set(xlabel=r'$r_c\, [h_{50}^{-1}$ Mpc]')
# fig.savefig('Data/Plots/Mohr+99_single_beta_rc_h50_mpc.pdf', format='pdf')
plt.show()

# Convert the data into h-less units
rc_mean_mpc = rc_mean_h50 / little_h_50
rc_uerr_mpc = rc_uerr_h50 / little_h_50
rc_lerr_mpc = rc_lerr_h50 / little_h_50

# Calculate the 1-sigma range on the h-less data
rc_quants_mpc = np.quantile(rc_mean_mpc, [0.16, 0.5, 0.84])

fig, ax = plt.subplots()
ax.errorbar(rc_mean_mpc, np.arange(0, len(rc_mean_mpc)),  xerr=np.array([rc_lerr_mpc, rc_uerr_mpc]), fmt='o')
ax.axvline(x=rc_quants_mpc[1], linestyle='--', color='k')
ax.axvspan(rc_quants_mpc[0], rc_quants_mpc[2], alpha=0.4, color='gray')
ax.tick_params(axis='y', which='both', left=False, labelleft=False)
ax.set(xlabel=r'$r_c$ [Mpc]')
# fig.savefig('Data/Plots/Mohr+99_single_beta_rc_mpc.pdf', format='pdf')
plt.show()

#%% Fit models to data
gamma_mle = stats.gamma.fit(rc_mean_mpc, floc=0.)
lognorm_mle = stats.lognorm.fit(rc_mean_mpc, floc=0.)
norm_mle = stats.norm.fit(rc_mean_mpc)

# Make frozen random variables
gamma_rv = stats.gamma(*gamma_mle)
lognorm_rv = stats.lognorm(*lognorm_mle)
norm_rv = stats.norm(*norm_mle)

# Make plot
fig, ax = plt.subplots()
hist, bins, _ = ax.hist(rc_mean_mpc, 'auto', density=True, align='mid')
x = np.linspace(min(bins), max(bins), 10000)
ax.plot(x, gamma_rv.pdf(x), label=r'Gamma($k={0:.2f}, \theta={2:.2f}$)'.format(*gamma_mle))
ax.plot(x, lognorm_rv.pdf(x), label=r'Lognormal($\mu={2:.2f}, \sigma={0:.2f}$)'.format(*lognorm_mle))
ax.plot(x, norm_rv.pdf(x), label=r'Norm($\mu={0:.2f}, \sigma={1:.2f}$)'.format(*norm_mle))
ax.legend()
ax.set(title='MLE fits to Mohr+99 core radii values', xlabel=r'$r_c$ [Mpc]', ylabel='Normalized Counts')
# fig.savefig('Data/Plots/Mohr+99_prob_model_fits.pdf', format='pdf')
plt.show()

#%% Generate a log-normal distribution using the mean and standard deviation from Sarazin86
mu = 0.25 * little_h_50
sigma = 0.04 * little_h_50
print(f'Sarazin core radius values: r_c = {mu:.3f} +- {sigma:.3f}')

fig, ax = plt.subplots()
hist, bins, _ = ax.hist(rc_mean_mpc, 'auto', density=True, align='mid')
x = np.linspace(min(bins), max(bins), 10000)
pdf = stats.lognorm.pdf(x, s=sigma, scale=mu)
ax.plot(x, pdf, label=r'Lognormal($\mu={mu:.2f}, \sigma={sigma:.2f}$)'.format(mu=mu, sigma=sigma))
ax.legend()
ax.set(title='Mohr+99 data with Sarazin86 core radius parameters', xlabel=r'$r_c$ [Mpc]', ylabel='Normalized Counts')
# fig.savefig('Data/Plots/Mohr+99_Sarazin86_parameters.pdf', format='pdf')
plt.show()
