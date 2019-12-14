"""
NFW_Beta_fitting.py
Author: Benjamin Floyd

Fits a beta-model to a NFW model to obtain the equivalent beta parameters for realistic NFW halos.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from astropy.cosmology import FlatLambdaCDM, WMAP7
from astropy.table import Table, vstack
from concentration_Mass_Relation.converters import mdelta_to_m200
from concentration_Mass_Relation.mass_concentrations import duffy_concentration
from concentration_Mass_Relation.nfw import NFW

from custom_math import rchisq

cosmo_concord = FlatLambdaCDM(H0=70, Om0=0.3)


def beta_model(r, a, beta, rc):
    return a * (1 + (r / rc) ** 2) ** (-1.5 * beta + 0.5)


# Read in the AGN catalog and obtain the median mass and redshift of the sample
agn_cat = vstack([Table.read('Data/Output/SPT_IRAGN.fits'), Table.read('Data/Output/SPTPol_IRAGN.fits')])
median_z = np.median(agn_cat['REDSHIFT'])
median_m500 = np.median(agn_cat['M500'])

# Convert the mass from M500 to M200
median_m200 = mdelta_to_m200(median_m500, duffy_concentration, 500, args=(median_z, cosmo_concord))
median_r200 = (3 * median_m200 * u.Msun / (4 * np.pi * 200 * WMAP7.critical_density(median_z).to('Msun/Mpc3'))) ** (
        1 / 3)

# van der Burg et al. (2014) parameters
c_vdb = 5.14
r200_vdb = np.median([0.9, 0.8, 1.0, 0.8, 0.8, 0.6, 1.8, 0.8, 0.9, 0.5])
z_vdb = np.median([0.867, 1.335, 0.869, 1.004, 0.956, 1.035, 0.871, 1.156, 1.177, 1.196])

# Zenteno et al. (2016) parameters
c_zenteno = 2.36
m200_zenteno = np.median([16.61, 24.14, 19.26, 13.25, 12.96, 14.75, 12.87, 14.10, 13.47, 12.43, 17.59, 20.12, 12.59,
                          17.96, 13.87, 19.14, 16.73, 26.64, 12.70, 15.95, 14.51, 15.13, 27.37, 12.39, 14.28,
                          19.88]) * 1e14
z_zenteno = np.median([0.350, 0.870, 0.284, 0.415, 0.500, 0.300, 0.438, 0.458, 0.422, 0.59, 0.421, 0.36, 0.345, 0.972,
                       0.176, 0.226, 0.164, 0.296, 0.232, 0.342, 1.132, 0.097, 0.351, 0.358, 0.775, 0.596])
r200_zenteno = (3 * m200_zenteno * u.Msun / (4 * np.pi * 200 * WMAP7.critical_density(z_zenteno).to('Msun/Mpc3'))) ** (
        1 / 3)

# Concentration using Duffy et al. (2008) c-M relation with my cluster sample
c_duffy = duffy_concentration(m200=median_m200, z=median_z, cosmo=cosmo_concord)

# Set up NFW profiles
nfw_vdb = NFW(size=r200_vdb, c=c_vdb, z=z_vdb, size_type='radius', cosmology=cosmo_concord)
nfw_zenteno = NFW(size=m200_zenteno, c=c_zenteno, z=z_zenteno, size_type='mass', cosmology=WMAP7)
nfw_duffy = NFW(size=median_m200, c=c_duffy, z=median_z, size_type='mass', cosmology=cosmo_concord)

r_vdb = np.linspace(0.01, r200_vdb, num=100)
r_zenteno = np.linspace(0.01, r200_zenteno.value, num=100)
r_duffy = np.linspace(0.01, median_r200.value, num=100)

nfw_profile_vdb = nfw_vdb.density(r_vdb) / nfw_vdb.rho_c
nfw_profile_zenteno = nfw_zenteno.density(r_zenteno) / nfw_zenteno.rho_c
nfw_profile_duffy = nfw_duffy.density(r_duffy) / nfw_duffy.rho_c

# Fit the Beta model to the NFW model
param_bounds = ([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 1.0])
fit_vdb, cov_vdb = op.curve_fit(beta_model, r_vdb, nfw_profile_vdb, bounds=param_bounds, sigma=nfw_profile_vdb * 0.1)
fit_zenteno, cov_zenteno = op.curve_fit(beta_model, r_zenteno, nfw_profile_zenteno, bounds=param_bounds,
                                        sigma=nfw_profile_zenteno * 0.1)
fit_duffy, cov_duffy = op.curve_fit(beta_model, r_duffy, nfw_profile_duffy, bounds=param_bounds,
                                    sigma=nfw_profile_duffy * 0.1)

# Compute the errors for each parameter
err_vdb = np.sqrt(np.diag(cov_vdb))
err_zenteno = np.sqrt(np.diag(cov_zenteno))
err_duffy = np.sqrt(np.diag(cov_duffy))

# Generate the beta models using the fit solutions above
beta_vbd = beta_model(r_vdb, *fit_vdb)
beta_zenteno = beta_model(r_zenteno, *fit_zenteno)
beta_duffy = beta_model(r_duffy, *fit_duffy)

# Compute reduced chi-squared goodness of fit values for the optimal parameters
rchi2_vdb = rchisq(nfw_profile_vdb, beta_vbd, deg=len(fit_vdb), sd=nfw_profile_vdb * 0.1)
rchi2_zenteno = rchisq(nfw_profile_zenteno, beta_zenteno, deg=len(fit_zenteno), sd=nfw_profile_zenteno * 0.1)
rchi2_duffy = rchisq(nfw_profile_duffy, beta_duffy, deg=len(fit_duffy), sd=nfw_profile_duffy * 0.1)

# %%
fig, ax = plt.subplots()
ax.plot(r_vdb / r200_vdb, nfw_profile_vdb, label=rf'$c_{{g}}={c_vdb}$ van der Burg+14')
ax.plot(r_zenteno / r200_zenteno, nfw_profile_zenteno, label=rf'$c_{{g}}={c_zenteno}$ Zenteno+16')
ax.plot(r_duffy / median_r200, nfw_profile_duffy, label=rf'$c_{{DM}}={c_duffy:.2f}$ SPTcl using Duffy+08 c-M')
ax.set(xlabel=r'$r/r_{200}$', ylabel=r'$\rho(r)/\rho_c$', yscale='log')
ax.legend()
plt.show()
fig.savefig('Data/Plots/NFW_Beta_fitting/NFW_profile_comparisons.pdf', format='pdf')

# %% Combined comparison plot
fig, ax = plt.subplots()
ax.plot(r_vdb / r200_vdb, nfw_profile_vdb, color='C0', label=rf'$c_{{g}}={c_vdb}$ van der Burg+14')
ax.plot(r_vdb / r200_vdb, beta_vbd, color='C0', ls='--',
        label=rf'Best Fit: $\beta = {fit_vdb[1]:.3f}\pm{err_vdb[1]:.4f}, r_c = {fit_vdb[2]:.3f}\pm{err_vdb[2]:.4f}$ Mpc')
ax.plot(r_zenteno / r200_zenteno, nfw_profile_zenteno, color='C1', label=rf'$c_{{g}}={c_zenteno}$ Zenteno+16')
ax.plot(r_zenteno / r200_zenteno, beta_zenteno, color='C1', ls='--',
        label=rf'Best Fit: $\beta = {fit_zenteno[1]:.3f}\pm{err_zenteno[1]:.4f}, '
              rf'r_c = {fit_zenteno[2]:.3f}\pm{err_zenteno[2]:.4f}$ Mpc')
ax.plot(r_duffy / median_r200, nfw_profile_duffy, color='C2',
        label=rf'$c_{{DM}}={c_duffy:.2f}$ SPTcl using Duffy+08 c-M')
ax.plot(r_duffy / median_r200, beta_duffy, color='C2', ls='--',
        label=rf'Best Fit: $\beta = {fit_duffy[1]:.3f}\pm{err_duffy[1]:.4f}, '
              rf'r_c = {fit_duffy[2]:.3f}\pm{err_duffy[2]:.4f}$ Mpc')
ax.set(xlabel=r'$r/r_{200}$', ylabel=r'$\rho(r)/\rho_c$', yscale='log')
ax.legend()
plt.show()
fig.savefig('Data/Plots/NFW_Beta_fitting/NFW_fit_Beta_combined_comparisons.pdf', format='pdf')

# %% Individual comparison plots
fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col', figsize=(6.4, 14.4))
ax[0].plot(r_vdb / r200_vdb, nfw_profile_vdb, color='C0', label=rf'NFW, $c_{{g}}={c_vdb}$')
ax[0].plot(r_vdb / r200_vdb, beta_vbd, color='C0', ls='--',
           label=rf'Best Fit, $\chi^2_\nu = {rchi2_vdb:.2f}$')
ax[0].text(0.05, 0.05, rf'$\beta = {fit_vdb[1]:.3f}\pm{err_vdb[1]:.4f}, r_c = {fit_vdb[2]:.3f}\pm{err_vdb[2]:.4f}$ Mpc',
           transform=ax[0].transAxes)
ax[0].set(title='van der Burg+14', ylabel=r'$\rho(r)/\rho_c$', yscale='log')
ax[0].legend()

ax[1].plot(r_zenteno / r200_zenteno, nfw_profile_zenteno, color='C1', label=rf'NFW, $c_{{g}}={c_zenteno}$')
ax[1].plot(r_zenteno / r200_zenteno, beta_zenteno, color='C1', ls='--',
           label=rf'Best Fit, $\chi^2_\nu = {rchi2_zenteno:.2f}$')
ax[1].text(0.05, 0.05, rf'$\beta = {fit_zenteno[1]:.3f}\pm{err_zenteno[1]:.4f}, '
                       rf'r_c = {fit_zenteno[2]:.3f}\pm{err_zenteno[2]:.4f}$ Mpc',
           transform=ax[1].transAxes)
ax[1].set(title='Zenteno+16', ylabel=r'$\rho(r)/\rho_c$', yscale='log')
ax[1].legend()

ax[2].plot(r_duffy / median_r200, nfw_profile_duffy, color='C2', label=rf'NFW, $c_{{DM}}={c_duffy:.2f}$')
ax[2].plot(r_duffy / median_r200, beta_duffy, color='C2', ls='--',
           label=rf'Best Fit, $\chi^2_\nu = {rchi2_duffy:.2f}$')
ax[2].text(0.05, 0.05, rf'$\beta = {fit_duffy[1]:.3f}\pm{err_duffy[1]:.4f}, '
                       rf'r_c = {fit_duffy[2]:.3f}\pm{err_duffy[2]:.4f}$ Mpc',
           transform=ax[2].transAxes)
ax[2].set(title='SPTcl using Duffy+08 c-M', xlabel=r'$r/r_{200}$', ylabel=r'$\rho(r)/\rho_c$', yscale='log')
ax[2].legend()
plt.show()
fig.savefig('Data/Plots/NFW_Beta_fitting/NFW_fit_Beta_individual_comparisons.pdf', format='pdf')

# %% Summary
names = [('van der Burg', c_vdb), ('Zenteno', c_zenteno), ('SPTcl with Duffy', c_duffy)]
fits = [fit_vdb, fit_zenteno, fit_duffy]
errs = [err_vdb, err_zenteno, err_duffy]
rchi2s = [rchi2_vdb, rchi2_zenteno, rchi2_duffy]
for i in range(len(names)):
    print(
        f'''{names[i][0]} with c = {names[i][1]:.2f}
Best fit: 
a = {fits[i][0]:.2f} +- {errs[i][0]:.3f}
beta = {fits[i][1]:.3f} +- {errs[i][1]:.4f}
rc = {fits[i][2]:.3f} +- {errs[i][2]:.4f} Mpc
reduced Chi2 = {rchi2s[i]:.2f}
        ''')

# Radial parameters in r500 units
rc_mpc_vdb = fit_vdb[2] * u.Mpc
rc_mpc_zenteno = fit_zenteno[2] * u.Mpc
rc_mpc_duffy = fit_duffy[2] * u.Mpc

rc_r500_vdb = rc_mpc_vdb / nfw_vdb.radius_delta(500)
rc_r500_zenteno = rc_mpc_zenteno / nfw_zenteno.radius_delta(500)
rc_r500_duffy = rc_mpc_duffy / nfw_duffy.radius_delta(500)
print(f'''rc fit parameters in r500 units
van der Burg: rc = {rc_r500_vdb:.2f} r500 ({nfw_vdb.radius_delta(500):.2f})
Zenteno: rc = {rc_r500_zenteno:.2f} r500 ({nfw_zenteno.radius_delta(500):.2f})
SPTcl with Duffy: rc = {rc_r500_duffy:.2f} r500 ({nfw_duffy.radius_delta(500):.2f})''')
