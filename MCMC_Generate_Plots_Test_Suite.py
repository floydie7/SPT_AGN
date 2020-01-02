"""
MCMC_Generate_Plots.py
Author: Benjamin Floyd

Generates the chain walker and corner plots for a previously generated emcee chain file.
"""

import corner
import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Set parameter values
theta_true = 2.5  # Amplitude.
eta_true = 1.2  # Redshift slope
zeta_true = -1.0  # Mass slope
beta_true = 1.0  # Radial slope
rc_true = 0.1  # Core radius (in r500)
C_true = 0.371  # Background AGN surface density

max_radius = 5.0  # Maximum integration radius in r500 units

# Our file storing the full test suite
filename = 'Data/MCMC/Mock_Catalog/Chains/Final_tests/core_radius_tests/' \
           'emcee_run_w36_s1000000_mock_t0.094_e1.2_z-1.0_b0.5_rc_variable_C0.371_core_radius_tests.h5'

# Get a list of the chain runs stored in our file
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

chain_names = [chain_name for chain_name in chain_names if 'trial9' in chain_name]

# Load in all samplers from the file
sampler_dict = {chain_name: emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in chain_names}

# Process each chain
for chain_name, sampler in sampler_dict.items():
    print('-----\n{}'.format(chain_name))
    # theta_true = float(re.search(r'_(\d+\.\d+)_', chain_name).group(1))

    # Get the chain from the sampler
    samples = sampler.get_chain()

    # To get the number of iterations ran, number of walkers used, and the number of parameters measured
    nsteps, nwalkers, ndim = samples.shape

    labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']
    truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, C_true]

    # Plot the chains
    fig, axes = plt.subplots(nrows=ndim, ncols=1, sharex='col')
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], color='k', alpha=0.3)
        ax.axhline(truths[i], color='b')
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set(xlim=[0, len(samples)], ylabel=labels[i])

    axes[0].set(title=chain_name)
    axes[-1].set(xlabel='Steps')

    fig.savefig('Data/MCMC/Mock_Catalog/Plots/Final_tests/core_radius_tests/trial_9/'
                'Param_chains_mock_t{theta}_e{eta}_z{zeta}_b{beta}_rc{rc}_C{C}_{chain_name}_full_spt.pdf'
                .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, rc=rc_true, C=C_true,
                        chain_name=chain_name),
                format='pdf')

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

    # Produce the corner plot
    fig = corner.corner(flat_samples, labels=labels, truths=truths, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.suptitle(chain_name)
    fig.savefig('Data/MCMC/Mock_Catalog/Plots/Final_tests/core_radius_tests/trial_9/'
                'Corner_plot_mock_t{theta}_e{eta}_z{zeta}_b{beta}_rc{rc}_C{C}_{chain_name}_full_spt.pdf'
                .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, rc=rc_true, C=C_true,
                        chain_name=chain_name), format='pdf')

    print('Iterations ran: {}'.format(sampler.iteration))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f} (truth: {true})'
              .format(labels=labels[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0], true=truths[i]))

    print('Mean acceptance fraction: {:.2f}'.format(np.mean(sampler.accepted / sampler.iteration)))

    # Get estimate of autocorrelation time
    print('Autocorrelation time: {:.1f}'.format(tau))
# plt.show()
