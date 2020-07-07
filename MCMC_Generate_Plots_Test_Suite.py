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

# labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$']

# Our file storing the full test suite
filename = 'Data/MCMC/SPT_Data/Chains/emcee_chains_SPTcl_IRAGN.h5'

# Get a list of the chain runs stored in our file
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

chain_names = [chain_name for chain_name in chain_names if 'wider_eta' in chain_name]

# Load in all samplers from the file
sampler_dict = {chain_name: emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in chain_names}

# Process each chain
for chain_name, sampler in sampler_dict.items():
    print(f'-----\n{chain_name}')

    # Get the chain from the sampler
    samples = sampler.get_chain()

    # To get the number of iterations ran, number of walkers used, and the number of parameters measured
    nsteps, nwalkers, ndim = samples.shape

    # Plot the chains
    fig, axes = plt.subplots(nrows=ndim, ncols=1, sharex='col')
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], color='k', alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set(xlim=[0, len(samples)], ylabel=labels[i])

    axes[0].set(title=chain_name)
    axes[-1].set(xlabel='Steps')

    fig.savefig(f'Data/MCMC/SPT_Data/Plots/Param_chains_SPTcl_{chain_name}.pdf', format='pdf')
    fig.savefig(f'Data/MCMC/SPT_Data/Plots/Param_chains_SPTcl_{chain_name}.png', format='png')

    try:
        # Calculate the autocorrelation time
        tau_est = sampler.get_autocorr_time()
        assert np.isfinite(tau_est).all()

        tau = np.mean(tau_est)

        # Remove the burn-in. We'll use ~3x the autocorrelation time
        # burnin = int(3 * tau)
        burnin = int(nsteps // 3)

        # We will also thin by roughly half our autocorrelation time
        thinning = int(tau // 2)

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
    # flat_log_prob_samples = sampler.get_log_prob(discard=burnin, thin=thinning, flat=True)
    # all_samples = np.concatenate((flat_samples, flat_log_prob_samples[:, None]), axis=1)
    # labels[-1] = r'$\ln(Post)$'

    # Produce the corner plot
    fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.suptitle(chain_name)
    fig.savefig(f'Data/MCMC/SPT_Data/Plots/Corner_plot_SPTcl_{chain_name}.pdf', format='pdf')

    print(f'Iterations ran: {sampler.iteration}')
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f}'
              .format(labels=labels[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0]))

    print(f'Mean acceptance fraction: {np.mean(sampler.accepted / sampler.iteration):.2f}')

    # Get estimate of autocorrelation time
    print(f'Autocorrelation time: {tau:.1f}')
    plt.close('all')
# plt.show()
