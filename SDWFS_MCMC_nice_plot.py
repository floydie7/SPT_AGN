"""
SDWFS_MCMC_nice_plot.py
Author: Benjamin Floyd

Generates a nice posterior plot to illustrate how the prior on the background parameter is chosen.
"""
import emcee.backends
import h5py
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from pygtc import plotGTC
from scipy.stats import norm

# Read in the chain file and get the correct chain name
fname = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Chains/emcee_chains_SDWFS_IRAGN.h5'
with h5py.File(fname, 'r') as f:
    chain_names = list(f.keys())
chain_name = chain_names[3]

# Load in the sampler object
sampler = emcee.backends.HDFBackend(fname, name=chain_name)

# Calculate the burn-in
tau = np.mean(sampler.get_autocorr_time())
burnin = int(3 * tau)

# Flatten chain
flat_samples = sampler.get_chain(discard=burnin, flat=True)

# Compute the HDI (and other stats)
idata = az.from_emcee(sampler, var_names=['C']).sel(draw=slice(burnin, None))
c_hdi = az.hdi(idata, hdi_prob=0.68).data_vars['C'].data
c_median = np.mean(flat_samples[:, 0])
c_stats = np.insert(c_hdi, 1, c_median)
bounds = np.diff(c_stats)

# Plot posterior
fig = plotGTC(flat_samples, paramNames='$C$', figureSize='AandA_page', doOnly1dPlot=True, smoothingKernel=0, nBins=20)
ax = fig.gca()
for x in c_stats:
    ax.axvline(x=x, ls='--', c='k', alpha=0.5)
ax.set_title(rf'SDWFS $C = {c_stats[1]:.3f}^{{+{bounds[1]:.3f}}}_{{-{bounds[0]:.3f}}}$')
fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Plots/Nice_Corner_plot_{chain_name}.pdf')

# Also plot a sample figure of the prior we will use
c_prior_sigma = 0.024
c_prior = norm(loc=c_median, scale=c_prior_sigma)
xx = np.linspace(c_median - 4*c_prior_sigma, c_median + 4*c_prior_sigma, num=100)
fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.plot(xx, c_prior.pdf(xx))
ax.axvline(x=c_median - c_prior_sigma, ls='--', c='k', alpha=0.5)
ax.axvline(x=c_median + c_prior_sigma, ls='--', c='k', alpha=0.5)
ax.set(title=fr'$\mathcal{{P}} = \mathcal{{N}}(\mu={c_median:.3f}, \sigma={c_prior_sigma:.3f})$',
       xlabel=r'$C\/[\mathrm{arcmin}^{-2}]$')
ax.yaxis.set_visible(False)
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Plots/C_prior_for_clusters_new_square.pdf')

