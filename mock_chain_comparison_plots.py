"""
mock_chain_comparison_plots.py
Author: Benjamin Floyd

Plots multiple versions of corner plots together to be able to compare them.
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
from pygtc import plotGTC
import h5py
import re

# Sampler files
chain_dir = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Port_Rebuild_Tests/eta-zeta_grid'
snr_583_chains_fname = f'{chain_dir}/emcee_mock_eta-zeta_grid_308cl_snr5.83.h5'
snr_20_chains_fname = f'{chain_dir}/emcee_mock_eta-zeta_grid_308cl_snr20.h5'

# Get chain names
with h5py.File(snr_583_chains_fname, 'r') as f, h5py.File(snr_20_chains_fname, 'r') as g:
    snr_583_chain_names = list(f.keys())
    snr_20_chain_names = list(g.keys())

# Filter for desired chains
catalog_name = 'e-5.00_z2.00'
chain_qualifiers = 'wideEtaZeta'
snr_583_chain_names = [chain_name for chain_name in snr_583_chain_names
                       if (catalog_name in chain_name) and (chain_qualifiers in chain_name)]
snr_20_chain_names = [chain_name for chain_name in snr_20_chain_names if catalog_name in chain_name]

# Read in samplers
snr_583_samplers = {chain_name: emcee.backends.HDFBackend(snr_583_chains_fname, name=chain_name)
                    for chain_name in snr_583_chain_names}
snr_20_samplers = {chain_name: emcee.backends.HDFBackend(snr_20_chains_fname, name=chain_name)
                   for chain_name in snr_20_chain_names}
samplers = snr_583_samplers | snr_20_samplers

# Use a standard burn in of 1500 steps
burnin = 1500
chains = {chain_name: sampler.get_chain(discard=burnin, flat=True) for chain_name, sampler in samplers.items()}

# Pad the cluster-only chain with a null axis, so we can compare it with the others
if cl_only_chains := [s for s in list(chains.keys()) if 'cl-only' in s]:
    for cl_only_name in cl_only_chains:
        chains[cl_only_name] = np.hstack((chains[cl_only_name], np.full(chains[cl_only_name].shape[0], np.nan)
                                          .reshape(chains[cl_only_name].shape[0], 1)))

# %% Define plotting parameters
chain_label_parser = re.compile(r'snr(\d+(?:.\d+)?)(?:_wideEtaZeta_)?(apriori|cl-only)?')
chain_label_info = [chain_label_parser.findall(s)[0] for s in list(chains.keys())]
chain_label_info = [(snr, suffix if suffix or snr != '5.83' else 'burnin') for snr, suffix in chain_label_info]
suffix_opts = {'apriori': ' (a priori initialization)', 'cl-only': ' (cluster only)', 'burnin': ' (500 step burn-in)'}
chain_labels = [rf'SNR$\sim${snr}{suffix_opts.get(suffix, "")}' for snr, suffix in chain_label_info]
param_names = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C_0$']
param_pattern = re.compile(r'(?:[tezbCx]|rc)(-*\d+.\d+|\d+)')
truths = [np.array(param_pattern.findall(chain_name), dtype=float) for chain_name in chains.keys()]
truth_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(chain_labels)]
truth_ls = ['--'] * len(chain_labels)
font_family = {'family': 'DejaVu Sans'}

# Make plot
fig = plotGTC(chains=list(chains.values()), chainLabels=chain_labels, paramNames=param_names, figureSize=8,
              truths=truths, truthColors=truth_colors, truthLineStyles=truth_ls,
              customLabelFont=font_family, customLegendFont=font_family, customTickFont=font_family)
fig.suptitle(rf'Mock Catalog ($\eta, \zeta$) = ({", ".join(param_pattern.findall(catalog_name))})')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Port_Rebuild_Tests/eta-zeta_grid/snr_20/'
            f'corner_plot_{catalog_name}_comparisons.pdf')
fig.show()
