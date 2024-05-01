"""
mock_suite_recovery_check.py
Benjamin Floyd

Checks how the posterior fit parameters compare to the true input parameters.
"""

import re
import emcee
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

param_pattern = re.compile(r'(?:[tezbCx]|rc)(-*\d+.\d+|\d+)')
eta_range = [-5, -3, 0, 3, 4, 5]
zeta_range = [-2, -1, 0, 1, 2]

# Load in the MCMC fits
filename = ('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/local_backgrounds/'
            'SPTcl-Mock_SNR_cl+bkg_chains.h5')
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

sampler_dict = {chain_name: emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in chain_names}

mcmc_fits = {}
for sampler_name, sampler in sampler_dict.items():
    # To get the number of iterations ran, number of walkers used, and the number of parameters measured
    nsteps, nwalkers, ndim = sampler.get_chain().shape

    # Compute the burn-in period for the chains
    try:
        # Calculate the autocorrelation time (discarding the initial burn-in time)
        tau_est = sampler.get_autocorr_time()
        assert np.isfinite(tau_est).all()

        tau = np.mean(tau_est)

        # Remove the burn-in. We'll use ~3x the autocorrelation time
        burnin = int(3 * tau)
    except emcee.autocorr.AutocorrError:
        tau_est = sampler.get_autocorr_time(quiet=True)
        tau = np.mean(tau_est)

        burnin = int(nsteps // 3)

    except AssertionError:
        tau_est = sampler.get_autocorr_time(quiet=True)
        tau = np.nanmean(tau_est)

        burnin = int(nsteps // 3)

    # Extract the flattened chain
    flat_samples = sampler.get_chain(discard=burnin, flat=True)

    # Compute the quantiles for the posteriors
    mcmc_fits[sampler_name] = {'eta': np.percentile(flat_samples[:, 1], [16, 50, 84]),
                               'zeta': np.percentile(flat_samples[:, 2], [16, 50, 84])}

eta_trend, zeta_trend = {}, {}
for sample_name, posterior_fit in mcmc_fits.items():
    _, eta_true, zeta_true, *_ = np.array(param_pattern.findall(sample_name), dtype=float)

#%% Make plot
norm = mpl.colors.Normalize(vmin=-5, vmax=5, clip=True)
cmapper = mpl.cm.ScalarMappable(norm=norm, cmap='seismic')
fig, axarr = plt.subplots(ncols=2, nrows=2, sharey='row', sharex='col', figsize=(2 * 6.4, 4.8), constrained_layout=True)
for sample_name, posterior_fit in mcmc_fits.items():
    _, eta_true, zeta_true, *_ = np.array(param_pattern.findall(sample_name), dtype=float)
    # eta dependency
    axarr[0, 0].errorbar(eta_true+0.1*zeta_true, posterior_fit['eta'][1] - eta_true,
                         yerr=np.diff(posterior_fit['eta']).reshape((2, 1)),
                         mfc=cmapper.to_rgba(zeta_true), marker='o', mec='k', c='k')
    axarr[1, 0].errorbar(eta_true+0.1*zeta_true, posterior_fit['zeta'][1] - zeta_true,
                         yerr=np.diff(posterior_fit['zeta']).reshape((2, 1)),
                         mfc=cmapper.to_rgba(zeta_true), marker='s', mec='k', c='k')

    # zeta dependency
    axarr[0, 1].errorbar(zeta_true+0.03*eta_true, posterior_fit['eta'][1] - eta_true,
                         yerr=np.diff(posterior_fit['eta']).reshape((2, 1)),
                         mfc=cmapper.to_rgba(eta_true), marker='o', mec='k', c='k')
    axarr[1, 1].errorbar(zeta_true+0.03*eta_true, posterior_fit['zeta'][1] - zeta_true,
                         yerr=np.diff(posterior_fit['zeta']).reshape((2, 1)),
                         mfc=cmapper.to_rgba(eta_true), marker='s', mec='k', c='k')

# Show color bar
cb = plt.colorbar(cmapper, ax=axarr.ravel().tolist(), pad=0.01, aspect=30, label=r'Input $\eta$ or $\zeta$', spacing='uniform')
cb.set_ticks([-5, -3, -2, -1, 0, 1, 2, 3, 4, 5])
cb.set_ticklabels([f'{c}' for c in [-5, -3, -2, -1, 0, 1, 2, 3, 4, 5]])

# Add reference lines
for ax in axarr.flatten():
    ax.axhline(y=0, ls='--', c='k', alpha=0.3)

# Set axis properties
axarr[1, 0].xaxis.set_ticks(eta_range)
axarr[1, 1].xaxis.set_ticks(zeta_range)
axarr[0, 0].set(ylabel=r'Posterior Fit - True $\eta$')
axarr[1, 0].set(xlabel=r'Input $\eta$', ylabel=r'Posterior Fit - True $\zeta$')
axarr[1, 1].set(xlabel=r'Input $\zeta$')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/local_backgrounds/cluster+background/'
            'eta-zeta_posterior_fit-input.pdf')
plt.show()
