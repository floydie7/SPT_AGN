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

theta_true = 3.0
eta_true = 4.0
zeta_true = -1.0
beta_true = 1.0
rc_true = 0.1
C_true = 0.333
truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, C_true]
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C$']

# Our file storing the full test suite
# filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Chains/emcee_chains_SDWFS_IRAGN.h5'
# filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Final_tests/fuzzy_selection/' \
#            'emcee_chains_Mock_fuzzy_selection.h5'
filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Final_tests/LF_tests/' \
           'emcee_chains_SPTcl_lf_tests.h5'

# Get a list of the chain runs stored in our file
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

chain_names = [chain_name for chain_name in chain_names if 't3.0' in chain_name]

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
        if ndim == 1:
            ax = axes
            truth_value = truths[-1]
            label_value = labels[-1]
        else:
            ax = axes[i]
            truth_value = truths[i]
            label_value = labels[i]
        ax.plot(samples[:, :, i], color='k', alpha=0.3)
        ax.axhline(truth_value, color='b')
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set(xlim=[0, len(samples)], ylabel=label_value)

    if ndim == 1:
        axes.set(title=chain_name)
        axes.set(xlabel='Steps')
    else:
        axes[0].set(title=chain_name)
        axes[-1].set(xlabel='Steps')

    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Plots/'
    #             f'Param_chains_SDWFS_Background_{chain_name}.png', dpi=300)
    # fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/fuzzy_selection/'
    #             f'Param_chains_Mock_t{theta_true}_e{eta_true}_z{zeta_true}_b{beta_true}_rc{rc_true}_C{C_true}'
    #             f'_{chain_name}.png', dpi=300)
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/'
                f'Param_chains_SPTcl_{chain_name}.png', dpi=300)
    # plt.show()

    try:
        # Calculate the autocorrelation time
        tau_est = sampler.get_autocorr_time()
        assert np.isfinite(tau_est).all()

        tau = np.mean(tau_est)

        # Remove the burn-in. We'll use ~3x the autocorrelation time
        burnin = int(3 * tau)

        # We will not thin if we don't have to
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
    if ndim == 1:
        label_list = [labels[-1]]
        truth_list = [truths[-1]]
    elif ndim == 5:
        label_list = labels[:-1]
        truth_list = truths[:-1]
    else:
        label_list = labels
        truth_list = truths
    fig = corner.corner(flat_samples, labels=label_list, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f',
                        plot_datapoints=False, truths=truth_list)
    fig.suptitle(chain_name)
    plt.tight_layout()

    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/SDWFS_Background/Plots/'
    #             f'Corner_plot_SDWFS_Background_{chain_name}.pdf')
    # fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/fuzzy_selection/'
    #             f'Corner_plot_Mock_t{theta_true}_e{eta_true}_z{zeta_true}_b{beta_true}_rc{rc_true}_C{C_true}'
    #             f'_{chain_name}.pdf')
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Final_tests/LF_tests/'
                f'Corner_plot_SPTcl_{chain_name}.pdf')
    # plt.show()

    print(f'Iterations ran: {sampler.iteration}')
    for i in range(ndim):
        if ndim == 1:
            label_list = [labels[-1]]
            truth_list = [truths[-1]]
        elif ndim == 5:
            label_list = labels[:-1]
            truth_list = truths[:-1]
        else:
            label_list = labels
            truth_list = truths
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f} (truth: {true})'
              .format(labels=label_list[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0],
                      true=truth_list[i]))

    print(f'Mean acceptance fraction: {np.mean(sampler.accepted / sampler.iteration):.2f}')

    # Get estimate of autocorrelation time
    print(f'Autocorrelation time: {tau:.1f}')
    plt.close('all')
# plt.show()
