"""
Signal-Noise_GTC_comparisons.py
Author: Benjamin Floyd

Creates corner plots comparing chains, mostly within a trial but some between.
"""

import re

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import pygtc


def keyfunct(n):
    return re.search(r'_(\d+\.\d+)_', n).group(1)


def get_theta_list(name_list):
    return [float(keyfunct(chain_name)) for chain_name in name_list]


def get_flat_chains(samplers):
    flat_chains = []
    for sampler in samplers:
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
        flat_chains.append(flat_samples)

    return flat_chains


# True parameter values
theta_true = None     # Amplitude.
eta_true = 1.2       # Redshift slope
beta_true = 0.5      # Radial slope
zeta_true = -1.0     # Mass slope
C_true = 0.371       # Background AGN surface density
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$C$']
truths = [theta_true, eta_true, zeta_true, beta_true, C_true]


filename = 'Data/MCMC/Mock_Catalog/Chains/signal-noise_tests/' \
           'emcee_run_w30_s1000000_mock_tvariable_e1.2_z-1.0_b0.5_C0.371_full_spt_snr_tests.h5'

# Get the chain names
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

#%%  Trial 2 Trends

# Load in the trial 2 chains
trial2_names = [chain_name for chain_name in chain_names if 'trial2' in chain_name]
trial2_samplers = [emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in trial2_names]

# Get the flat chains
trial2_flat_chains = get_flat_chains(trial2_samplers)

# Keep only the eta and zeta portions
trial2_eta_zeta_chains = [chain[:, 1:3] for chain in trial2_flat_chains]

# Make plot
chain_labels = [r'$\theta = {theta}$'.format(theta=theta) for theta in get_theta_list(trial2_names)]
trial2_fig = pygtc.plotGTC(trial2_eta_zeta_chains, paramNames=labels[1:3], chainLabels=chain_labels,
                           truths=truths[1:3], figureSize=8)
trial2_fig.suptitle('Trial 2')
trial2_fig.show()

#%% Trial 3 Trends

# Load in the trial 3 chains
trial3_names = [chain_name for chain_name in chain_names if 'trial3' in chain_name and '2.000' not in chain_name]
trial3_samplers = [emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in trial3_names]

# Get the flat chains
trial3_flat_chains = get_flat_chains(trial3_samplers)

# Keep only the eta and zeta portions
trial3_eta_zeta_chains = [chain[:, 1:3] for chain in trial3_flat_chains]

# Make plot
chain_labels = [r'$\theta = {theta}$'.format(theta=theta) for theta in get_theta_list(trial3_names)]
trial3_fig = pygtc.plotGTC(trial3_eta_zeta_chains, paramNames=labels[1:3], chainLabels=chain_labels,
                           truths=truths[1:3], figureSize=8)
trial3_fig.suptitle('Trial 3')
trial3_fig.show()

#%% Trial 4 Trends

# Load in the trial 4 chains
trial4_names = [chain_name for chain_name in chain_names if 'trial4' in chain_name]
trial4_samplers = [emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in trial4_names]

# Get the flat chains
trial4_flat_chains = get_flat_chains(trial4_samplers)

# Match the trial 3 and 4 chains for theta = 0.153
trial3_4_t0153_chains = [trial3_flat_chains[3], trial4_flat_chains[1]]

# Set theta_true
truths[0] = 0.153

# Make plot
chain_labels = [r'Trial 3: $P_0$ = A Priori', r'Trial 4: $P_0$ = Prior Space']
trial4_fig = pygtc.plotGTC(trial3_4_t0153_chains, paramNames=labels, chainLabels=chain_labels,
                           truths=truths, figureSize=8)
trial4_fig.suptitle(r'$\theta = 0.153$    Trials 3 and 4')
trial4_fig.show()

#%% Trial 5 Trends

# Load in the trial 4 chains
trial5_names = [chain_name for chain_name in chain_names if 'trial5' in chain_name]
trial5_samplers = [emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in trial5_names]

# Get the flat chains
trial5_flat_chains = get_flat_chains(trial5_samplers)

# Set theta_true
truths[0] = 0.153

# Make plot
chain_labels = [r'$r_c = {rc}$'.format(rc=re.search(r'rc_input(\d.\d+)_', chain_name).group(1)) for chain_name in trial5_names]
trial5_fig = pygtc.plotGTC(trial5_flat_chains, paramNames=labels, chainLabels=chain_labels,
                           truths=truths, figureSize=8)
trial5_fig.suptitle(r'$\theta = 0.153$    True $r_c = 0.1$')
trial5_fig.show()
