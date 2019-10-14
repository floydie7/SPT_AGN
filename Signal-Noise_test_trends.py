"""
Signal-Noise_test_trends.py
Author: Benjamin Floyd

Analyzes the trends of the MCMC parameters from the Signal-Noise chains.
"""

import re

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np


def keyfunct(n):
    return re.search(r'_(\d+\.\d+)_', n).group(1)


filename = 'Data/MCMC/Mock_Catalog/Chains/signal-noise_tests/' \
           'emcee_run_w30_s1000000_mock_tvariable_e1.2_z-1.0_b0.5_C0.371_full_spt_snr_tests.h5'

# Get the chain names
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

# orig_chain_names = [chain_name for chain_name in chain_names if '_theta_prior_pm0.5theta_true' not in chain_name]
# narrow_chain_names = [chain_name for chain_name in chain_names if '_theta_prior_pm0.5theta_true' in chain_name]
#
# orig_theta_list = [float(keyfunct(chain_name)) for chain_name in chain_names
#               if '_theta_prior_pm0.5theta_true' not in chain_name]
# narrow_theta_list = [float(keyfunct(chain_name)) for chain_name in chain_names
#               if '_theta_prior_pm0.5theta_true' in chain_name]
theta_list = [float(keyfunct(chain_name)) for chain_name in chain_names]

# Read in the samplers
# orig_samplers = [emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in orig_chain_names]
# narrow_samplers = [emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in narrow_chain_names]
samplers = [emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in chain_names]


labels = ['theta', 'eta', 'zeta', 'beta', 'C']
eta_true = 1.2
zeta_true = -1.0
beta_true = 0.5
C_true = 0.371
rc_true = 0.1

# Extract the flat chains
mcmc_fits = {chain_name: {} for chain_name in chain_names}
for sampler, chain_name, theta_true in zip(samplers, chain_names, theta_list):
    # Get the chain from the sampler
    samples = sampler.get_chain()

    # To get the number of iterations ran, number of walkers used, and the number of parameters measured
    nsteps, nwalkers, ndim = samples.shape

    try:
        # Calculate the autocorrelation time
        tau_est = sampler.get_autocorr_time()

        tau = np.mean(tau_est)

        # Remove the burn-in. We'll use ~3x the autocorrelation time
        burnin = int(3 * tau)

        # We will also thin by roughly half our autocorrelation time
        thinning = int(tau // 2)

    except emcee.autocorr.AutocorrError:
        tau_est = sampler.get_autocorr_time(quiet=True)
        tau = np.mean(tau_est)

        burnin = int(nsteps // 3)
        thinning = 1

    flat_samples = sampler.get_chain(discard=burnin, thin=thinning, flat=True)

    # Produce the corner plot
    # truths = [theta_true, eta_true, zeta_true, beta_true, C_true]
    # fig = corner.corner(flat_samples, labels=labels, truths=truths, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    # fig.savefig('Corner_plot_mock_t{theta}_e{eta}_z{zeta}_b{beta}_C{C}_{chain_name}.pdf'
    #             .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true, chain_name=chain_name),
    #             format='pdf')

    for i in range(ndim):
        mcmc_fits[chain_name][labels[i]] = np.percentile(flat_samples[:, i], [16, 50, 84])

# eta plot
eta_values = [fit['eta'][1] for fit in mcmc_fits.values()]
lower_err, upper_err = [], []
for fit in mcmc_fits.values():
    q = np.diff(fit['eta'])
    lower_err.append(q[0])
    upper_err.append(q[1])
eta_errors = [lower_err, upper_err]

fig, ax = plt.subplots()
ax.errorbar(theta_list, eta_values, yerr=eta_errors, fmt='o')
ax.axhline(y=1.2, ls='--', label=r'True $\eta = 1.2$')
ax.set(xlabel=r'$\theta$', ylabel=r'$\eta$')
ax.legend()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Signal-Noise_tests/full_spt/mcmc_trends/eta_trend.pdf', format='pdf')

# zeta plot
zeta_values = [fit['zeta'][2] for fit in mcmc_fits.values()]
lower_err, upper_err = [], []
for fit in mcmc_fits.values():
    q = np.diff(fit['zeta'])
    lower_err.append(q[0])
    upper_err.append(q[1])
zeta_errors = [lower_err, upper_err]

fig, ax = plt.subplots()
ax.errorbar(theta_list, zeta_values, yerr=zeta_errors, fmt='o')
ax.axhline(y=-1.0, ls='--', label=r'True $\zeta =-1.0$')
ax.set(xlabel=r'$\theta$', ylabel=r'$\zeta$')
ax.legend()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Signal-Noise_tests/full_spt/mcmc_trends/zeta_trend.pdf', format='pdf')
