"""
SPT_AGN_Subcat_Plots.py
Author: Benjamin Floyd

Creates the plots using the emcee chains ran on Tusker.
"""

import numpy as np
import corner
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

# For diagnostic purposes, set the values of the parameters.
eta_true = 1.2
beta_true = -1.5
zeta_true = -1.0

ndim = 3
nwalkers = 64
nsteps = 500

# Load in the chains
files = os.listdir('Data/MCMC/Mock_Catalog/Chains/')
files.sort()
chain = [np.load('Data/MCMC/Mock_Catalog/Chains/'+f) for f in files]

for i in range(len(chain)):
   # Plot the chains
   fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
   ax1.plot(chain[i][:, :, 0].T, color='k', alpha=0.4)
   ax1.axhline(eta_true, color='b')
   ax1.yaxis.set_major_locator(MaxNLocator(5))
   ax1.set(ylabel=r'$\eta$', title='MCMC Chains')
   
   ax2.plot(chain[i][:, :, 1].T, color='k', alpha=0.4)
   ax2.axhline(beta_true, color='b')
   ax2.yaxis.set_major_locator(MaxNLocator(5))
   ax2.set(ylabel=r'$\zeta$')
   
   ax3.plot(chain[i][:, :, 2].T, color='k', alpha=0.4)
   ax3.axhline(zeta_true, color='b')
   ax3.yaxis.set_major_locator(MaxNLocator(5))
   ax3.set(ylabel=r'$\beta$')
   
   # ax4.plot(sampler.chain[:, :, 3].T, color='k', alpha=0.4)
   # ax4.yaxis.set_major_locator(MaxNLocator(5))
   # ax4.set(ylabel=r'$C$', xlabel='Steps')
   
   fig.savefig('Data/MCMC/Mock_Catalog/Plots/Param_chains_mock_catalog_w64_s500_sc{:02d}.pdf'.format(i), format='pdf')
   
   # Remove the burnin, typically 1/3 number of steps
   burnin = nsteps//3
   samples = chain[i][:, burnin:, :].reshape((-1, ndim))
   
   # Produce the corner plot
   fig = corner.corner(samples, labels=[r'$\eta$', r'$\zeta$', r'$\beta$'], truths=[eta_true, zeta_true, beta_true], quantiles=[0.16, 0.5, 0.84], show_titles=True)
   fig.savefig('Data/MCMC/Mock_Catalog/Plots/Corner_plot_mock_catalog_w64_s500_sc{:02d}.pdf'.format(i), format='pdf')

