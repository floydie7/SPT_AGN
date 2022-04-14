"""
MCMC_Generate_Plots_Suite_Compare.py
Author: Benjamin Floyd

Creates corner plots of chains that we wish to compare on a single plot.
"""

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pygtc import plotGTC
import re


def get_chains(samplers):
    for chain_name, sampler in samplers.items():
        nsteps, _, _ = sampler.get_chain().shape

        try:
            # Calculate the autocorrelation time
            tau_est = sampler.get_autocorr_time()
            assert np.isfinite(tau_est).all()

            tau = np.mean(tau_est)

            # Remove the burn-in. We'll use ~3x the autocorrelation time
            burnin = int(3 * tau)

            # We will not thin if we don't have to
            thinning = 1

        except emcee.autocorr.AutocorrError:
            tau_est = sampler.get_autocorr_time(quiet=True)
            tau = np.mean(tau_est)

            burnin = int(nsteps // 3)
            thinning = 1

        except AssertionError:
            tau_est = sampler.get_autocorr_time(quiet=True)
            tau = np.nanmean(tau_est)

            burnin = int(nsteps // 3)
            thinning = 1

        flat_samples = sampler.get_chain(discard=burnin, thin=thinning, flat=True)

        yield chain_name, flat_samples


id_pattern = re.compile(r'clseed(\d+)_objseed(\d+)')
font = {'family': 'DejaVu Sans', 'size': 14}

theta_true = 2.6
eta_true = 4.0
zeta_true = -1.0
beta_true = 1.0
rc_true = 0.1
C_true = 0.333
truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, C_true]
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']

filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Final_tests/mock_rework/' \
           'emcee_chains_semiemperical_rng_seeds.h5'

# Get a list of the chain runs stored in our file
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

# Load in all samplers from the file
sampler_dict = {chain_name: emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in chain_names}

background_only_samplers = {id_pattern.search(chain_name).group(0): sampler_obj for chain_name, sampler_obj
                            in sampler_dict.items() if 'background-only' in chain_name}
cluster_only_samplers = {id_pattern.search(chain_name).group(0): sampler_obj for chain_name, sampler_obj
                         in sampler_dict.items() if 'cluster-only' in chain_name}
full_los_samplers = {id_pattern.search(chain_name).group(0): sampler_obj for chain_name, sampler_obj
                     in sampler_dict.items() if 'cluster+background' in chain_name}

# Create the background-only plot
bkg_chains = {chain_name: chains for chain_name, chains in get_chains(background_only_samplers)}
label_list = [labels[-1]]
truth_list = [truths[-1]]

fig = plotGTC(chains=list(bkg_chains.values()), chainLabels=list(bkg_chains.keys()), paramNames=label_list,
              truths=truth_list, smoothingKernel=1, figureSize=10, customLegendFont=font, customLabelFont=font, customTickFont=font)
fig.suptitle('Background-only Chains')
fig.show()

# Create the cluster-only plot
cl_chains = {chain_name: chains for chain_name, chains in get_chains(cluster_only_samplers)}
label_list = labels[:-1]
truth_list = truths[:-1]

fig = plotGTC(chains=list(cl_chains.values()), chainLabels=list(cl_chains.keys()), paramNames=label_list,
              truths=truth_list, smoothingKernel=1, figureSize=10, customLegendFont=font, customLabelFont=font, customTickFont=font)
fig.suptitle('Cluster-only Chains')
fig.show()

# Create the cluster+background plot
los_chains = {chain_name: chains for chain_name, chains in get_chains(full_los_samplers)}
label_list = labels
truth_list = truths

fig = plotGTC(chains=list(los_chains.values()), chainLabels=list(los_chains.keys()), paramNames=label_list,
              truths=truth_list, smoothingKernel=1, figureSize=10, customLegendFont=font, customLabelFont=font, customTickFont=font)
fig.suptitle('Cluster + Background Chains')
fig.show()
