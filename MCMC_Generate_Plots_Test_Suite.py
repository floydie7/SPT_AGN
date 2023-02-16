"""
MCMC_Generate_Plots.py
Author: Benjamin Floyd

Generates the chain walker and corner plots for a previously generated emcee chain file.
"""
import json
import re

import corner
import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

# Read in the purity and surface density files
with (open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f,
      open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_background_prior_distributions.json',
           'r') as g):
    sdwfs_purity_data = json.load(f)
    sdwfs_prior_data = json.load(g)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
threshold_bins = sdwfs_prior_data['color_thresholds'][:-1]

# Set up interpolators
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')
agn_surf_den = interp1d(threshold_bins, sdwfs_prior_data['agn_surf_den'], kind='previous')


# For convenience, set up the function compositions
def agn_prior_surf_den(redshift: float) -> float:
    return agn_surf_den(agn_purity_color(redshift))


cluster_amp = 1.

theta_true = 5.0
eta_true = 4.0
zeta_true = -1.0
beta_true = 1.0
rc_true = 0.1
c0_true = agn_prior_surf_den(0.)

theta_true *= cluster_amp
c0_true *= cluster_amp
# truths = [np.log(theta_true), eta_true, zeta_true, beta_true, np.log(rc_true), np.log(c0_true)]
truths = [theta_true, eta_true, zeta_true, beta_true, rc_true, c0_true]
# truths = [theta_true, eta_true, zeta_true, beta_true, rc_true]
# truths = [eta_true, zeta_true, beta_true, rc_true, c0_true]
# labels = [r'$\ln \theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$\ln r_c$', r'$\ln C_0$']
labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C_0$']
# labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$']
# labels = [r'$\eta$', r'$\zeta$', r'$\beta$', r'$r_c$', r'$C_0$']

# Our file storing the full test suite
# filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Port_Rebuild_Tests/pure_poisson/emcee_mock_pure_poisson.h5'
filename = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Chains/Port_Rebuild_Tests/eta-zeta_grid/emcee_mock_eta-zeta_grid_snr0.23.h5'

# Get a list of the chain runs stored in our file
with h5py.File(filename, 'r') as f:
    chain_names = list(f.keys())

chain_names = [chain_name for chain_name in chain_names if 'walkerBurnin' in chain_name]

# Load in all samplers from the file
sampler_dict = {chain_name: emcee.backends.HDFBackend(filename, name=chain_name) for chain_name in chain_names}

# Process each chain
for chain_name, sampler in sampler_dict.items():
    print(f'-----\n{chain_name}')

    param_pattern = re.compile(r'(?:[tezbCx]|rc)(-*\d+.\d+|\d+)')
    truths = np.array(param_pattern.findall(chain_name), dtype=float)

    # Get the chain from the sampler
    samples = sampler.get_chain()

    # To get the number of iterations ran, number of walkers used, and the number of parameters measured
    nsteps, nwalkers, ndim = samples.shape

    # Exponentiate the chains of the log-sampled parameters
    # if ndim == 1:
    #     samples[:, :, 0] = np.exp(samples[:, :, 0])
    # elif ndim == 5:
    #     samples[:, :, 0] = np.exp(samples[:, :, 0])
    #     samples[:, :, 4] = np.exp(samples[:, :, 4])
    # else:
    #     samples[:, :, 0] = np.exp(samples[:, :, 0])
    #     samples[:, :, 4] = np.exp(samples[:, :, 4])
    #     samples[:, :, 5] = np.exp(samples[:, :, 5])

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

    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Port_Rebuild_Tests/eta-zeta_grid/snr_0.23/run_5/'
                f'Param_chains_mock_{chain_name}_expParams.pdf')
    plt.show()

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
    # if ndim == 1:
    #     flat_samples[:, 0] = np.exp(flat_samples[:, 0])
    # elif ndim == 5:
    #     flat_samples[:, 0] = np.exp(flat_samples[:, 0])
    #     flat_samples[:, 4] = np.exp(flat_samples[:, 4])
    # else:
    #     flat_samples[:, 0] = np.exp(flat_samples[:, 0])
    #     flat_samples[:, 4] = np.exp(flat_samples[:, 4])
    #     flat_samples[:, 5] = np.exp(flat_samples[:, 5])
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

    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Plots/Port_Rebuild_Tests/eta-zeta_grid/snr_0.23/run_5/'
                f'Corner_plot_mock_{chain_name}_expParams.pdf')
    plt.show()

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
        print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f} (truth: {true:.2f})'
              .format(labels=label_list[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0],
                      true=truth_list[i]))

    print(f'Mean acceptance fraction: {np.mean(sampler.accepted / sampler.iteration):.2f}')

    # Get estimate of autocorrelation time
    print(f'Autocorrelation time: {tau:.1f}')
    plt.close('all')
# plt.show()
