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
import scipy.optimize as op
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, Column
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


def model_rate(z, m, r_r500, params):
    """
    Our generating model.

    :param z: Redshift
    :param m: M_500
    :param r_r500: r / r_500
    :param params: Tuple of (theta, eta, zeta, beta)
    :return:
    """

    theta, eta, zeta, beta = params

    # Our amplitude is determined from the cluster data
    a = theta * (1 + z)**eta * (m / (1e15 * u.Msun))**zeta

    # For now, we assume no background contamination. All objects are in the cluster.
    model = a * (1 + r_r500**2)**(-1.5 * beta + 0.5)

    model[r_r500 > 1.5] = 0.

    return model


# Set up grid of radial positions (normalized by r500)
r_dist = np.linspace(0.1, 1.5, 50)

# Draw mass and redshift distribution from a uniform distribution as well.
mass_dist = np.random.uniform(0.2e15, 1.8e15, 300)
z_dist = np.random.uniform(0.5, 1.7, 300)

# Create cluster names
name_bank = ['SPT_Mock_{:03d}'.format(i) for i in range(300)]
SPT_data = Table([name_bank, z_dist, mass_dist], names=['SPT_ID', 'REDSHIFT', 'M500'])

# We'll need the r500 radius for each cluster too.
SPT_data['r500'] = (3 * SPT_data['M500'] /
                    (4 * np.pi * 500 * cosmo.critical_density(SPT_data['REDSHIFT']).to(u.Msun / u.Mpc**3)))**(1/3)
SPT_data_z1 = SPT_data[np.where(SPT_data['REDSHIFT'] >= 1.)]
SPT_data_m = SPT_data[np.where(SPT_data['M500'] <= 5e14)]

# Set parameter values
# theta_true = 0.25e-5    # Amplitude (?)
theta_true = 2e-4
eta_true = 1.2    # Redshift slope
beta_true = -1.5  # Radial slope
zeta_true = -1.0  # Mass slope
C_true = 0.371    # Background AGN surface density

for cluster in SPT_data_m[np.random.randint(0, len(SPT_data_m)+1, size=3)]:
    spt_id = cluster['SPT_ID']
    z_cl = cluster['REDSHIFT']
    m500_cl = cluster['M500'] * u.Msun
    r500_cl = cluster['r500'] * u.Mpc
    print("---\nCluster Data: z = {z:.2f}, M500 = {m:.2e}, r500 = {r:.2f}".format(z=z_cl, m=m500_cl, r=r500_cl))

    # Calculate the model values for the AGN candidates in the cluster
    rad_model = model_rate(z_cl, m500_cl, r_dist, (theta_true, eta_true, zeta_true, beta_true))

    # Find the maximum rate. This establishes that the number of AGN in the cluster is tied to the redshift and mass of
    # the cluster.
    max_rate = np.max(rad_model)
    print('Max rate: {}'.format(max_rate))

    # dx is set to 350 to mimic an IRAC image's width in pixels.
    dx = 350.

    # Simulate the AGN using the spatial Poisson point process.
    agn_coords = poisson_point_process(max_rate, dx)

    # Find the radius of each point placed scaled by the cluster's r500 radius
    radii = (np.sqrt((agn_coords[0] - dx/2.)**2 + (agn_coords[1] - dx/2.)**2)
             * (0.8627 * u.arcsec / cosmo.arcsec_per_kpc_proper(z_cl).to(u.arcsec / u.Mpc))) / r500_cl

    # Filter the candidates through the model to establish the radial trend in the data.
    rate_at_rad = model_rate(z_cl, m500_cl, radii, (theta_true, eta_true, zeta_true, beta_true))

    # Our rejection rate is the model rate at the radius scaled by the maximum rate
    prob_reject = rate_at_rad / max_rate
    # print(radii[0:5])
    # print(prob_reject[0:5])

    # Draw a random number for each candidate
    alpha = np.random.uniform(0, 1, len(rate_at_rad))

    x_final = agn_coords[0][np.where(prob_reject >= alpha)]
    y_final = agn_coords[1][np.where(prob_reject >= alpha)]
    print('Number of points in final selection: {}'.format(len(x_final)))

    # Calculate the radii of the final AGN scaled by the cluster's r500 radius
    r_final = (np.sqrt((x_final - dx/2.)**2 + (y_final - dx/2.)**2)
               * (0.8627 * u.arcsec / cosmo.arcsec_per_kpc_proper(z_cl).to(u.arcsec / u.Mpc))) / r500_cl

    # Create a table of our output objects
    AGN_list = Table([r_final], names=['r_r500'])
    AGN_list['SPT_ID'] = spt_id
    AGN_list['M500'] = m500_cl
    AGN_list['REDSHIFT'] = z_cl

    outAGN = AGN_list['SPT_ID', 'REDSHIFT', 'M500', 'r_r500']

    # outAGN.pprint(max_width=-1)
    outAGN.write('Data/MCMC/Mock_Catalog/Catalogs/new_mock_test.cat', format='ascii', overwrite=True)

    # fig, ax = plt.subplots()
    # ax.plot(r_dist, rad_model)
    # ax.set(title='Model rate', xlabel=r'$r/r_{500}$', ylabel=r'$N(r)$')
    # # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/Model_rate.pdf', format='pdf')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(agn_coords[0], agn_coords[1], edgecolor='b', facecolor='none', alpha=0.5)
    # ax.set(title=r'Spatial Poisson Point Process with $N_{{max}} = {:.2f}$'.format(max_rate),
    #        xlabel=r'$x$', ylabel=r'$y$', xlim=[0, dx], ylim=[0, dx])
    # # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/AGN_candidates.pdf', format='pdf')
    # plt.show()
    #
    fig, ax = plt.subplots()
    ax.scatter(x_final, y_final, edgecolor='b', facecolor='none', alpha=0.5)
    ax.set_aspect('equal', 'datalim')
    ax.set(title='Filtered SPPP', xlabel=r'$x$', ylabel=r'$y$', xlim=[0, dx], ylim=[0, dx])
    # fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Distributions/Final_AGN.pdf', format='pdf')
    plt.show()

    raise SystemExit

# Plot the canidate distributions
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
ax0.hist((1 + mock_candidates['REDSHIFT']), bins=50)
ax0.set(title='Mock Candidates', xlabel=r'$1+z$')

ax1.hist(mock_candidates['M500']/1e15, bins=50)
ax1.set(xlabel=r'$M_{500} / 10^{15} M_\odot$')

ax2.hist(mock_candidates['r_r500_radial'], bins=50)
ax2.set(xlabel=r'$r / r_{500}$')

plt.tight_layout()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Candidates_Distributions.pdf', format='pdf')



# Run the candidates through the model with the values above
N_model = np.array((1 + mock_candidates['REDSHIFT'])**eta_true
                   * (mock_candidates['M500'] / 1e15)**zeta_true
                   * (mock_candidates['r_r500_radial']) ** beta_true)

mock_candidates.add_column(Column(N_model, name='model'))

# Calculate the maximum possible model value for normalization
res = op.minimize(lambda data: -(1 + data[0]) ** eta_true * (data[1] / 1e15) ** zeta_true * (data[2]) ** beta_true,
                  x0=np.array([0., 0., 0.]), bounds=[(0.5, 1.7), (0.2e15, 1.8e15), (0.1, 1.5)])

# Our normalization value
max_model_val = -res.fun

# Normalize the model values to create probabilities
N_model_normed = N_model / max_model_val

# Draw a random number between [0,1] and if the number is smaller than the probabilities from our model, we keep the
# entry from our candidates for our mock catalog.
mock_catalog = Table(names=['SPT_ID', 'REDSHIFT', 'M500', 'r_r500_radial', 'model'],
                     dtype=['S16', 'f8', 'f8', 'f8', 'f8'])
for i in range(len(mock_candidates)):
    # Draw random number
    alpha = np.random.uniform(0, 1)

    # Check if the candidate should be added to the catalog
    if alpha < N_model_normed[i]:
        mock_catalog.add_row(mock_candidates[i])

print('Number of objects in Mock catalog: {}'.format(len(mock_catalog)))


# Plot the output distributions
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
ax0.hist((1 + mock_catalog['REDSHIFT']), bins=50)
ax0.set(title='Mock Catalog', xlabel=r'$1+z$')

ax1.hist(mock_catalog['M500']/1e15, bins=50)
ax1.set(xlabel=r'$M_{500} / 10^{15} M_\odot$')

ax2.hist(mock_catalog['r_r500_radial'], bins=50)
ax2.set(xlabel=r'$r / r_{500}$')

plt.tight_layout()
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Mock_Catalog_Distributions.pdf', format='pdf')

# Write the mock catalog to disk
mock_catalog.write('Data/MCMC/Mock_Catalog/Catalogs/mock_AGN_catalog.cat', format='ascii', overwrite=True)

sub_cat = mock_catalog[0:1500]
sub_cat.write('Data/MCMC/Mock_Catalog/Catalogs/mock_AGN_subcatalog00.cat', format='ascii', overwrite=True)

# # Split the catalog into sub-groups of ~1500 AGN to mimic our real data sample.
# for i in np.arange(1500, len(mock_catalog), 1500):
#     sub_cat = mock_catalog[i-1500:i]
#
#     print('Number of objects in subcatalog {j}: {n}'.format(j=i // 1500 - 1, n=len(sub_cat)))
#
#     sub_cat.write('Data/MCMC/Mock_Catalog/Catalogs/mock_AGN_subcatalog{j:02d}.cat'.format(j=i // 1500 - 1),
#                   format='ascii', overwrite=True)
