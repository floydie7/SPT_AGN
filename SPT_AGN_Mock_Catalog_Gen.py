"""
SPT_AGN_Mock_Catalog_Gen.py
Author: Benjamin Floyd

Using our Bayesian model, generates a mock catalog to use in testing the limitations of the model.
"""

from __future__ import print_function, division

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

# Set our random seed
np.random.seed(123)


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
    print('Number of points placed: ', p)

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

    # Our amplitude is determined from the cluster data
    a = theta * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc/u.arcmin)**2 * (1 + z)**eta * (m / (1e15 * u.Msun))**zeta

    # Our model rate is a surface density of objects in angular units (as we only have the background in angular units)
    model = a * (1 + (r / rc) ** 2) ** (-1.5 * beta + 0.5) + background

    # We impose a cut off of all objects with a radius greater than 1.5r500
    model[r / r500 > 1.5] = 0.

    return model


# Number of clusters to generate
n_cl = 100

# Set up grid of radial positions (normalized by r500)
r_dist = np.linspace(0, 2.5, 100) * u.Mpc

# Draw mass and redshift distribution from a uniform distribution as well.
mass_dist = np.random.uniform(0.2e15, 1.8e15, n_cl)
z_dist = np.random.uniform(0.5, 1.7, n_cl)

# Create cluster names
name_bank = ['SPT_Mock_{:03d}'.format(i) for i in range(n_cl)]
SPT_data = Table([name_bank, z_dist, mass_dist], names=['SPT_ID', 'REDSHIFT', 'M500'])

# We'll need the r500 radius for each cluster too.
SPT_data['r500'] = (3 * SPT_data['M500'] /
                    (4 * np.pi * 500 * cosmo.critical_density(SPT_data['REDSHIFT']).to(u.Msun / u.Mpc**3)))**(1/3)
SPT_data_z1 = SPT_data[np.where(SPT_data['REDSHIFT'] >= 1.)]
SPT_data_m = SPT_data[np.where(SPT_data['M500'] <= 5e14)]

# Set parameter values
# theta_true = 0.25e-5    # Amplitude (?)
theta_true = 50. / u.Mpc**2
eta_true = 1.2    # Redshift slope
beta_true = 0.5  # Radial slope
zeta_true = -1.0  # Mass slope
C_true = 0.371 / u.arcmin**2   # Background AGN surface density

params_true = (theta_true, eta_true, zeta_true, beta_true, C_true)

cluster_sample = SPT_data

AGN_cats = []
max_rate_list = []
aper_area = []
for cluster in cluster_sample:
    spt_id = cluster['SPT_ID']
    z_cl = cluster['REDSHIFT']
    m500_cl = cluster['M500'] * u.Msun
    r500_cl = cluster['r500'] * u.Mpc
    print("---\nCluster Data: z = {z:.2f}, M500 = {m:.2e}, r500 = {r:.2f}".format(z=z_cl, m=m500_cl, r=r500_cl))

    area_15r500 = np.pi * (1.5 * r500_cl * cosmo.arcsec_per_kpc_proper(z_cl).to(u.arcmin / u.Mpc))**2
    print('Aperture Area: {}'.format(area_15r500))
    aper_area.append(area_15r500.value)

    # Calculate the model values for the AGN candidates in the cluster
    rad_model = model_rate(z_cl, m500_cl, r_dist, r500_cl, params_true)

    # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
    # the cluster.
    max_rate = np.max(rad_model)
    print('Max rate: {}'.format(max_rate))
    max_rate_list.append(max_rate.value)

    # Dx is set to 5 to mimic an IRAC image's width in arcmin.
    Dx = 5.

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
    r_final_arcmin = (np.sqrt((x_final - Dx / 2.) ** 2 + (y_final - Dx / 2.) ** 2) * u.arcmin)
    r_final = r_final_arcmin * cosmo.kpc_proper_per_arcmin(z_cl).to(u.Mpc / u.arcmin)

    if r_final.size != 0:
        # Create a table of our output objects
        AGN_list = Table([r_final, r_final_arcmin], names=['radial_dist', 'radial_arcmin'])
        AGN_list['SPT_ID'] = spt_id
        AGN_list['M500'] = m500_cl
        AGN_list['REDSHIFT'] = z_cl
        AGN_list['r500'] = r500_cl

        AGN_cat = AGN_list['SPT_ID', 'REDSHIFT', 'M500', 'r500', 'radial_dist', 'radial_arcmin']
        AGN_cats.append(AGN_cat)

outAGN = vstack(AGN_cats)

# outAGN.pprint(max_width=-1)
outAGN.write('Data/MCMC/Mock_Catalog/Catalogs/new_mock_test_t50_100cl.cat', format='ascii', overwrite=True)

print('\n------\nparameters: {param}\nTotal number of clusters: {cl} \t Total number of objects: {agn}'
      .format(param=params_true, cl=len(outAGN.group_by('SPT_ID').groups.keys), agn=len(outAGN)))
print('Mean max rate : {} 1 / arcmin2'.format(np.mean(max_rate_list)))
print('Mean aperture area: {} arcmin2'.format(np.mean(aper_area)))

# fig, ax = plt.subplots()
# ax.plot(r_dist / r500_cl, rad_model)
# ax.set(title='Model rate', xlabel=r'$r/r_{500}$', ylabel=r'$N(r)$ [arcmin$^{-2}$]')
# # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/Model_rate.pdf', format='pdf')
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(agn_coords[0], agn_coords[1], edgecolor='b', facecolor='none', alpha=0.5)
# ax.set_aspect(1.0/ax.get_data_ratio())
# ax.set(title=r'Spatial Poisson Point Process with $N_{{max}} = {:.2f}$'.format(max_rate),
#        xlabel=r'$x$', ylabel=r'$y$', xlim=[0, Dx], ylim=[0, Dx])
# # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/AGN_candidates.pdf', format='pdf')
# plt.show()
#
# fig, ax = plt.subplots()
# ax.scatter(x_final, y_final, edgecolor='b', facecolor='none', alpha=0.5)
# ax.set_aspect(1.0/ax.get_data_ratio())
# # ax.set_aspect('equal', 'datalim')
# ax.set(title='Filtered SPPP', xlabel=r'$x$', ylabel=r'$y$', xlim=[0, Dx], ylim=[0, Dx])
# # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/Final_AGN.pdf', format='pdf')
# plt.show()
#
hist, bin_edges = np.histogram(r_final)
bins = (bin_edges[1:len(bin_edges)]-bin_edges[0:len(bin_edges)-1])/2. + bin_edges[0:len(bin_edges)-1]
# plt.hist(outAGN['radial_arcmin'])
# plt.show()

# but normalise the area
area_edges = np.pi * bin_edges**2
area = area_edges[1:len(area_edges)]-area_edges[0:len(area_edges)-1]
area_arcmin2 = area * u.Mpc**2 * cosmo.arcsec_per_kpc_proper(z_cl).to(u.arcmin / u.Mpc)**2
# print(area_arcmin2)

err = np.sqrt(hist)/area_arcmin2.value

fig, ax = plt.subplots()
ax.errorbar(bins/r500_cl.value, hist/area_arcmin2.value, yerr=err, fmt='o', label='Filtered SPPP Points Normalized by Area')
ax.plot(r_dist/r500_cl, rad_model, color="orange", label='Model Rate')
ax.set(title='Comparison of Sampled Points to Model', xlabel=r'$r$ [Mpc]', ylabel=r'Rate [arcmin$^{-2}$]')
ax.legend()
# fig.savefig('Data/MCMC/Mock_Catalog/Plots/Binned_data_to_Model.pdf', format='pdf')
plt.show()
