"""
MCMC_covariance_grid.py
Author: Benjamin Floyd

Generates a grid of covariance plots from the emcee trials.
"""

import re

import corner
import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Set parameter values
# theta_true = 2.5  # Amplitude.
# eta_true = 1.2  # Redshift slope
# zeta_true = -1.0  # Mass slope
beta_true = 1.0  # Radial slope
rc_true = 0.1  # Core radius (in r500)
C_true = 0.371  # Background AGN surface density

max_radius = 5.0  # Maximum integration radius in r500 units

# Our file storing the full test suite
filename = 'Data/MCMC/Mock_Catalog/Chains/Final_tests/slope_tests/' \
           'emcee_run_w36_s1000000_mock_tvariable_evariable_zvariable_b1.0_rc_0.1_C0.371_slope_tests.h5'

# Get a list of the chain runs stored in our file
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

chain_names = [chain_name for chain_name in chain_names if 'trial4' in chain_name]

# Load in all samplers from the file
sampler_dict = {chain_name: emcee.backends.HDFBackend(filename, name=chain_name)
                for chain_name in sorted(chain_names, key=lambda x: re.search(r'z-?\d+(?:\.\d+)', x).group(0))}

# Process each chain
fig, axarr = plt.subplots(nrows=7, ncols=6, figsize=(16, 17))
for ax, (chain_name, sampler) in zip(axarr.flatten(), sampler_dict.items()):
    # print('-----\n{}'.format(chain_name))
    theta_true, eta_true, zeta_true = np.array(re.findall(r'-?\d+(?:\.\d+)', chain_name), dtype=np.float)

    # Get the chain from the sampler
    samples = sampler.get_chain()

    # To get the number of iterations ran, number of walkers used, and the number of parameters measured
    nsteps, nwalkers, ndim = samples.shape

    labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']
    truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, C_true]

    # These chains have problems with calculating the autocorrelation. Skip making the corner plot.
    if chain_name in ['trial2_t4.711_e0.00_z-1.00', 'trial3_t7.816_e0.00_z-1.50', 'trial4_t4.711_e0.00_z-1.00']:
        continue

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
    # flat_log_prob_samples = sampler.get_log_prob(discard=burnin, thin=thinning, flat=True)
    # all_samples = np.concatenate((flat_samples, flat_log_prob_samples[:, None]), axis=1)
    # labels[-1] = r'$\ln(Post)$'

    eta_samples = flat_samples[:, 1]
    zeta_samples = flat_samples[:, 2]

    corner.hist2d(eta_samples, zeta_samples, ax=ax)
    ax.plot(truths[1], truths[2], "s", color="#4682b4")
    ax.axvline(truths[1], color="#4682b4")
    ax.axhline(truths[2], color="#4682b4")

axarr = np.fliplr(axarr)

for i, zeta in enumerate(np.arange(-1.75, 0, 0.25)[::-1]):
    axarr[i, 0].set(ylabel=rf'$\zeta$ = {zeta}')
    axarr[i, -1].set(ylabel=rf'$\zeta$ = {zeta}')
    axarr[i, 0].yaxis.set_label_position('right')

for i, eta in enumerate(np.arange(0., 6.)):
    axarr[0, i].set(xlabel=rf'$\eta$ = {eta}')
    axarr[-1, i].set(xlabel=rf'$\eta$ = {eta}')
    axarr[0, i].xaxis.set_label_position('top')

plt.tight_layout()

plt.show()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Final_tests/Slope_tests/trial_4/eta_zeta_covariance_trends.png')
