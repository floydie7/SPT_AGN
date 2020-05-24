"""
MCMC_Generate_nice_corner.py
Author: Benjamin Floyd

Creates a nice corner plot using Sebastian Bocquet's pyGTC package.
"""

import re

import emcee
import h5py
import numpy as np
from pygtc import plotGTC

max_radius = 5.0  # Maximum integration radius in r500 units

# Our file storing the full test suite
filename = 'Data/MCMC/Mock_Catalog/Chains/Final_tests/slope_tests/' \
           'emcee_run_w36_s1000000_mock_tvariable_evariable_zvariable_b1.0_rc_0.1_C0.371_slope_tests.h5'

# Get a list of the chain runs stored in our file
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

chain_names = [chain_name for chain_name in chain_names if 'trial5c' in chain_name]

# Select our chain
chain_name = chain_names[0]
sampler = emcee.backends.HDFBackend(filename, name=chain_name)

theta_true, eta_true, zeta_true = np.array(re.findall(r'-?\d+(?:\.\d+)', chain_name), dtype=np.float)
beta_true = 1.0  # Radial slope
rc_true = 0.1  # Core radius (in r500)
C_true = 0.371  # Background AGN surface density
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']
truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, C_true]

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

# Get print out of fit parameters
str_list = ['_________Parameter Fits_________\n']
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    str_list.append(r'{labels} = ${median:.3f}^{{{upper_err}}}_{{{lower_err}}}$ (truth: {true})'
                    .format(labels=labels[i], median=mcmc[1], upper_err=f'+{q[1]:.4f}', lower_err=f'-{q[0]:.4f}',
                            true=truths[i]))

str_list.append('Mean acceptance fraction: {:.2f}'.format(np.mean(sampler.accepted / sampler.iteration)))

text_str = '\n'.join(str_list)

# Make the plot
plotName = 'Data/MCMC/Mock_Catalog/Plots/Final_tests/Slope_tests/trial_5/Slope_tests_trial5c_pretty_corner.png'
fig = plotGTC(flat_samples, paramNames=labels, truths=truths, priors=[None, None, None, None, None, (C_true, 0.157)],
              figureSize='APJ_page',
              plotName=plotName)
fig.text(0.57, 0.6, text_str, transform=fig.transFigure, fontsize=9, bbox=dict(boxstyle='round',
                                                                               facecolor='ghostwhite', alpha=0.5))
fig.savefig(plotName)
