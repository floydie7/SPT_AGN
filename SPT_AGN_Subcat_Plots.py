"""
SPT_AGN_Subcat_Plots.py
Author: Benjamin Floyd

Creates the plots using the emcee chains ran on Tusker.
"""

import corner
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# For diagnostic purposes, set the values of the parameters.
eta_true = 1.2
beta_true = -1.5
zeta_true = -1.0

ndim = 4
nwalkers = 100
nsteps = 500

# # Load in the chains
# files = os.listdir('Data/MCMC/Mock_Catalog/Chains/Gauss_beta_sig10_Prior')
# files.sort()
# chain = [np.load('Data/MCMC/Mock_Catalog/Chains/Gauss_beta_sig10_Prior/'+f) for f in files]
chain = np.load('Data/MCMC/Mock_Catalog/emcee_run_w100_s500_sc00.npy')

# for i in range(len(chain)):
# Plot the chains
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, sharex=True)
ax0.plot(chain[:, :, 0].T, color='k', alpha=0.4)
ax0.yaxis.set_major_locator(MaxNLocator(5))
ax0.set(ylabel=r'$\theta$', title='MCMC Chains')

ax1.plot(chain[:, :, 1].T, color='k', alpha=0.4)
ax1.axhline(eta_true, color='b')
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.set(ylabel=r'$\eta$')

ax2.plot(chain[:, :, 2].T, color='k', alpha=0.4)
ax2.axhline(zeta_true, color='b')
ax2.yaxis.set_major_locator(MaxNLocator(5))
ax2.set(ylabel=r'$\zeta$')

ax3.plot(chain[:, :, 3].T, color='k', alpha=0.4)
ax3.axhline(beta_true, color='b')
ax3.yaxis.set_major_locator(MaxNLocator(5))
ax3.set(ylabel=r'$\beta$')

# ax4.plot(sampler.chain[:, :, 3].T, color='k', alpha=0.4)
# ax4.yaxis.set_major_locator(MaxNLocator(5))
# ax4.set(ylabel=r'$C$', xlabel='Steps')

fig.savefig('Data/MCMC/Mock_Catalog/Plots/Param_chains_mock_catalog_sc00.pdf', format='pdf')

# Remove the burnin, typically 1/3 number of steps
burnin = nsteps // 3
samples = chain[:, burnin:, :].reshape((-1, ndim))

# Produce the corner plot
fig = corner.corner(samples, labels=[r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$'],
                    truths=[None, eta_true, zeta_true, beta_true],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Corner_plot_mock_catalog_sc00.pdf', format='pdf')

theta_mcmc, eta_mcmc, zeta_mcmc, beta_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                 zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print("""MCMC Results:
theta = {theta[0]:.2f} +{theta[1]:.3f} -{theta[2]:.3f}
eta = {eta[0]:.2f} +{eta[1]:.3f} -{eta[2]:.3f} (truth: {eta_true})
zeta = {zeta[0]:.2f} +{zeta[1]:.3f} -{zeta[2]:.3f} (truth: {zeta_true})
beta = {beta[0]:.2f} +{beta[1]:.3f} -{beta[2]:.3f} (truth: {beta_true})"""
      .format(theta=theta_mcmc, eta=eta_mcmc, zeta=zeta_mcmc, beta=beta_mcmc,
              eta_true=eta_true, zeta_true=zeta_true, beta_true=beta_true))

