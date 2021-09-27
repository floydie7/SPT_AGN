"""
Mock_Catalog_example_plots.py
Author: Benjamin Floyd

Generates the example plots for a simulated cluster using a spatial Poisson point process.
"""

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack
from scipy import stats
# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)
# Set our cosmology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)


def poisson_point_process(model, dx, dy=None):
    """
    Uses a spatial Poisson point process to generate AGN candidate coordinates.
    :param model: The model rate used in the Poisson distribution to determine the number of points being placed.
    :param dx: Upper bound on x-axis (lower bound is set to 0).
    :param dy: Upper bound on y-axis (lower bound is set to 0).
    :return coord: 2d numpy array of (x,y) coordinates of AGN candidates.
    """

    if dy is None:
        dy = dx

    # Draw from Poisson distribution to determine how many points we will place.
    p = stats.poisson(model * dx * dy).rvs()

    # Drop `p` points with uniform x and y coordinates
    x = stats.uniform.rvs(0, dx, size=p)
    y = stats.uniform.rvs(0, dy, size=p)

    # Combine the x and y coordinates.
    coord = np.vstack((x, y))

    return coord


def model_rate(z, m, r, r500, params):
    """
    Our generating model.

    :param z: Redshift of the cluster
    :param m: M_500 mass of the cluster
    :param r: A vector of radii of objects within the cluster
    :param r500: r500 radius of the cluster
    :param params: Tuple of (theta, eta, zeta, beta, background)
    :return model: A surface density profile of objects as a function of radius
    """

    # Unpack our parameters
    theta, eta, zeta, beta, background = params

    # The cluster's core radius
    rc = 0.1 * u.Mpc

    # theta = theta * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc/u.arcmin)**2

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z)**eta * (m / (1e15 * u.Msun))**zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r / rc) ** 2) ** (-1.5 * beta + 0.5) + background * cosmo.arcsec_per_kpc_proper(z).to(u.arcmin / u.Mpc)**2

    # We impose a cut off of all objects with a radius greater than 1.5r500
    model[r / r500 > 1.5] = 0.

    return model


# Set cluster data
spt_id = 'SPT_Mock_A'
z_cl = 1.01
m500_cl = 5.63e14 * u.Msun
r500_cl = 0.86 * u.Mpc

# Set parameter values
theta_true = 0.1 / u.Mpc**2
eta_true = 1.2    # Redshift slope
beta_true = 0.5  # Radial slope
zeta_true = -1.0  # Mass slope
C_true = 0.371 / u.arcmin**2   # Background AGN surface density
# C_true = 0 / u.arcmin**2

params_true = (theta_true, eta_true, zeta_true, beta_true, C_true)

# Set up grid of radial positions (normalized by r500)
r_dist = np.linspace(0, 2, 100) * u.Mpc

# Calculate the model values for the AGN candidates in the cluster
rad_model = model_rate(z_cl, m500_cl, r_dist, r500_cl, params_true)

# Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
# the cluster.
max_rate = np.max(rad_model)

# Dx is set to 5 to mimic an IRAC image's width in arcmin.
Dx = 5. * u.arcmin #* cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin)
Dx = Dx.value

# Simulate the AGN using the spatial Poisson point process.
agn_coords = poisson_point_process(max_rate, Dx)

# Find the radius of each point placed scaled by the cluster's r500 radius
radii = (np.sqrt((agn_coords[0] - Dx / 2.) ** 2 + (agn_coords[1] - Dx / 2.) ** 2) * u.arcmin
         * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin))

# Filter the candidates through the model to establish the radial trend in the data.
rate_at_rad = model_rate(z_cl, m500_cl, radii, r500_cl, params_true)

# Our rejection rate is the model rate at the radius scaled by the maximum rate
prob_reject = rate_at_rad / max_rate

# Draw a random number for each candidate
alpha = np.random.uniform(0, 1, len(rate_at_rad))

x_final = agn_coords[0][np.where(prob_reject >= alpha)]
y_final = agn_coords[1][np.where(prob_reject >= alpha)]
print('Number of points in final selection: {}'.format(len(x_final)))

# Calculate the radii of the final AGN scaled by the cluster's r500 radius
r_final = (np.sqrt((x_final - Dx / 2.) ** 2 + (y_final - Dx / 2.) ** 2) * u.arcmin
           * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin))

fig, ax = plt.subplots()
ax.plot(r_dist / r500_cl, rad_model)
ax.set(title='Model rate', xlabel=r'$r/r_{500}$', ylabel=r'$N(r)$ [arcmin$^{-2}$]')
# fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/Model_rate.pdf', format='pdf')
# plt.show()

fig, ax = plt.subplots()
ax.scatter(agn_coords[0], agn_coords[1], edgecolor='b', facecolor='none', alpha=0.5)
ax.set_aspect(aspect=1.0)
ax.set(title=r'Spatial Poisson Point Process with $N_{{max}} = {:.2f}$'.format(max_rate),
       xlabel=r'$x$ (arcmin)', ylabel=r'$y$ (arcmin)', xlim=[0, Dx], ylim=[0, Dx])
fig.set_size_inches(5,5, forward=True)
# fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/AGN_candidates'
#             '_z{z}_m{m500.value:.2e}_r{r500.value}_t{theta.value}_e{eta}_z{zeta}_b{beta}_C{C.value}'
#             .format(z=z_cl, m500=m500_cl, r500=r500_cl,
#                     theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true).replace('.', '') +
#             '.pdf',
#             format='pdf')
# plt.show()

fig, ax = plt.subplots()
ax.scatter(x_final, y_final, edgecolor='b', facecolor='none', alpha=0.5)
ax.set_aspect(aspect=1.0)
# ax.set_aspect('equal', 'datalim')
ax.set(title='Mock AGN', xlabel=r'$x$ (arcmin)', ylabel=r'$y$ (arcmin)', xlim=[0, Dx], ylim=[0, Dx])
fig.set_size_inches(5,5, forward=True)
# fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/Final_AGN'
#             '_z{z}_m{m500.value:.2e}_r{r500.value}_t{theta.value}_e{eta}_z{zeta}_b{beta}_C{C.value}'
#             .format(z=z_cl, m500=m500_cl, r500=r500_cl,
#                     theta=theta_true, eta=eta_true, zeta=zeta_true, beta=beta_true, C=C_true).replace('.', '') +
#             '.pdf',
#             format='pdf')
# plt.show()


hist, bin_edges = np.histogram(r_final)
bins = (bin_edges[1:len(bin_edges)]-bin_edges[0:len(bin_edges)-1])/2. + bin_edges[0:len(bin_edges)-1]
plt.hist(r_final.value)
# plt.show()

# but normalise the area
area_edges = np.pi * bin_edges**2
area = area_edges[1:len(area_edges)]-area_edges[0:len(area_edges)-1]
area_arcmin2 = area * u.Mpc**2 * cosmo.arcsec_per_kpc_proper(z_cl).to(u.arcmin / u.Mpc)**2
# print(area_arcmin2)

err = np.sqrt(hist)/area_arcmin2.value

fig, ax = plt.subplots()
ax.errorbar(bins, hist/area_arcmin2.value, yerr=err, fmt='o', label='Filtered SPPP Points Normalized by Area')
ax.plot(r_dist, rad_model, color="orange", label='Model Rate')
ax.set(title='Comparison of Sampled Points to Model', xlabel=r'$r$ [Mpc]', ylabel=r'Rate [arcmin$^{-2}$]')
ax.legend()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/binned_comparison_t0.1_zcl1.01_m500_563e14_5arcmin.pdf', format='pdf')
plt.show()
