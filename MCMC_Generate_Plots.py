"""
MCMC_Generate_Plots.py
Author: Benjamin Floyd

Generates the chain walker and corner plots for a previously generated emcee chain file.
"""

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# True parameter values
theta_true = 0.2     # Amplitude.
eta_true = 1.2       # Redshift slope
beta_true = 0.5      # Radial slope
zeta_true = -1.0     # Mass slope
C_true = 0.371       # Background AGN surface density

max_radius = 5.0  # Maximum integration radius in r500 units

# Read in the chain file
filename = 'emcee_run_w30_s1000000_mock_tvariable_e1.2_z-1.0_b0.5_C0.371_snr_tests.h5'
sampler = emcee.backends.HDFBackend(filename, name='snr_test_0.200_bkg_free_rc_fixed')

# Get the chain from the sampler
samples = sampler.get_chain()

# To get the number of iterations ran, number of walkers used, and the number of parameters measured
nsteps, nwalkers, ndim = samples.shape

labels = [r'$\theta$', r'$\eta$', r'$\zeta$', r'$\beta$', r'$C$']
truths = [theta_true, eta_true, zeta_true, beta_true, C_true]

# # Plot the chains
# fig, axes = plt.subplots(nrows=ndim, ncols=1, sharex='col')
# for i in range(ndim):
#     ax = axes[i]
#     ax.plot(samples[:, :, i], color='k', alpha=0.3)
#     ax.axhline(truths[i], color='b')
#     ax.yaxis.set_major_locator(MaxNLocator(5))
#     ax.set(xlim=[0, len(samples)], ylabel=labels[i])
#
# axes[0].set(title='MCMC Chains')
# axes[-1].set(xlabel='Steps')

# fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/'
#             'Param_chains_mock_t{theta}_e{eta}_z{zeta}_b{beta}_C{C}_maxr{maxr}'
#             '_data_to_5r500_no_mask_background_fixed_int_to_0.5r500.pdf'
#             .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true, maxr=max_radius),
#             format='pdf')

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
fig = corner.corner(flat_samples, labels=labels, truths=truths, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    title_fmt='.3f', smooth=1, plot_datapoints=False)
axes = np.array(fig.axes).reshape((ndim, ndim))
for ax in axes[:, 0]:
    ax.set(xlim=[0, 1])
# fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/'
#             'Corner_plot_mock_t{theta}_e{eta}_z{zeta}_b{beta}_C{C}_maxr{maxr}'
#             '_data_to_5r500_no_mask_background_fixed_int_to_0.5r500.pdf'
#             .format(theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true, maxr=max_radius),
#             format='pdf')
# fig.savefig('Corner_plot_mock_t0.2_e1.2_z-1.0_b0.5_C0.371_snr_test_adjusted_t_plot.pdf', format='pdf')


print('Iterations ran: {}'.format(sampler.iteration))
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print('{labels} = {median:.3f} +{upper_err:.4f} -{lower_err:.4f} (truth: {true})'
          .format(labels=labels[i].strip('$\\'), median=mcmc[1], upper_err=q[1], lower_err=q[0], true=truths[i]))

print('Mean acceptance fraction: {:.2f}'.format(np.mean(sampler.accepted / sampler.iteration)))

# Get estimate of autocorrelation time
print('Autocorrelation time: {:.1f}'.format(tau))
plt.show()
