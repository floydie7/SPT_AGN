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

theta_true = 2.5
eta_true = 1.2
zeta_true = -1.0
beta_true = 1.0
rc_true = 0.1
C_true = 0.371
# truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, C_true]
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']
# labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$']
# labels = [r'$C$']

# Our file storing the full test suite
# filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Chains/emcee_chains_SDWFS_IRAGN.h5'
# filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Final_tests/fuzzy_selection/' \
#            'emcee_chains_Mock_fuzzy_selection.h5'
# filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SPT_Data/Chains/emcee_chains_SPTcl_fuzzy_selection.h5'
filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SPT_Data/Chains/emcee_chains_SPTcl_fuzzy_selection_snapshot2021-05-05T135416-0500.h5'

# Get a list of the chain runs stored in our file
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

chain_names = [chain_name for chain_name in chain_names if '_LF' in chain_name]

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
        # ax = axes
        ax.plot(samples[:, :, i], color='k', alpha=0.3)
        # ax.axhline(truths[i], color='b')
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set(xlim=[0, len(samples)], ylabel=labels[i])

    axes[0].set(title=chain_name)
    axes[-1].set(xlabel='Steps')
    # axes.set(title=chain_name)
    # axes.set(xlabel='Steps')

    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Plots/'
    #             f'Param_chains_SDWFS_Background_{chain_name}.png', dpi=300)
    # fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/fuzzy_selection/'
    #             f'Param_chains_Mock_t{theta_true}_e{eta_true}_z{zeta_true}_b{beta_true}_rc{rc_true}_C{C_true}'
    #             f'_{chain_name}.png', dpi=300)
    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SPT_Data/Plots/fuzzy_selection/'
    #             f'Param_chains_SPTcl_{chain_name}.png', dpi=300)
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SPT_Data/Plots/fuzzy_selection/'
                f'Param_chains_SPTcl_{chain_name}_snapshot2021-05-05T135416-0500.png', dpi=300)
    plt.show()

    try:
        # Calculate the autocorrelation time
        tau_est = sampler.get_autocorr_time()
        assert np.isfinite(tau_est).all()

        tau = np.mean(tau_est)

        # Remove the burn-in. We'll use ~3x the autocorrelation time
        burnin = int(3 * tau)
        # burnin = int(nsteps // 3)

        # We will also thin by roughly half our autocorrelation time
        # thinning = int(tau // 2)
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
    # flat_log_prob_samples = sampler.get_log_prob(discard=burnin, thin=thinning, flat=True)
    # all_samples = np.concatenate((flat_samples, flat_log_prob_samples[:, None]), axis=1)
    # labels[-1] = r'$\ln(Post)$'

    # Produce the corner plot
    # fig = corner.corner(flat_samples, labels=labels, truths=truths, quantiles=[0.16, 0.5, 0.84], show_titles=True,
    #                     title_fmt='.3f', smooth=1, plot_datapoints=False)
    fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f',
                        plot_datapoints=False)
    fig.suptitle(chain_name)
    plt.tight_layout()

    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Plots/'
    #             f'Corner_plot_SDWFS_Background_{chain_name}.pdf')
    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/fuzzy_selection/'
    #             f'Corner_plot_Mock_t{theta_true}_e{eta_true}_z{zeta_true}_b{beta_true}_rc{rc_true}_C{C_true}'
    #             f'_{chain_name}.pdf')
    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SPT_Data/Plots/fuzzy_selection/'
    #             f'Corner_plot_SPTcl_{chain_name}.pdf')
    plt.show()

    print(f'Iterations ran: {sampler.iteration}')
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f}'
              .format(labels=labels[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0]))
        # print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f} (truth: {true})'
        #       .format(labels=labels[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0], true=truths[i]))

    print(f'Mean acceptance fraction: {np.mean(sampler.accepted / sampler.iteration):.2f}')

    # Get estimate of autocorrelation time
    print(f'Autocorrelation time: {tau:.1f}')
    plt.close('all')
# plt.show()
