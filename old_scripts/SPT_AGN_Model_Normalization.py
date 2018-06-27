"""
SPT_AGN_Model_Normalization.py
Author: Benjamin Floyd

Exploring options to reconcile the model with the number of objects in the cluster.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from astropy.table import Table

# Set matplotlib parameters
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = np.sqrt(20)

subcat = Table.read('Data/MCMC/Mock_Catalog/Catalogs/mock_AGN_subcatalog00.cat', format='ascii')

subcat_grp = subcat.group_by('SPT_ID')

num_obj = []
model_sum = []
for cluster in subcat_grp.groups:
        num_obj.append(len(cluster))
        model_sum.append(np.sum(cluster['model']))

num_obj = np.array(num_obj)
model_sum = np.array(model_sum)

fig, ax = plt.subplots()
ax.plot(num_obj, model_sum, '.')
ax.set(xlabel='Number of objects in cluster', ylabel='Summed model number for cluster', title='No Normalization')
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Model_Number_Trend.pdf', format='pdf')

fig, ax = plt.subplots()
ax.plot(num_obj, model_sum/num_obj, '.')
ax.set(xlabel='Number of objects in cluster', ylabel='Summed model number for cluster',
       title='Normalized by $1/N_{agn}$')
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Model_Number_1_Nagn.pdf', format='pdf')

fig, ax = plt.subplots()
ax.plot(num_obj, model_sum/(num_obj**2), '.')
ax.set(xlabel='Number of objects in cluster', ylabel='Summed model number for cluster',
       title='Normalized by $1/N_{agn}^2$')
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Model_Number_1_Nagn2.pdf', format='pdf')

fig, ax = plt.subplots()
ax.plot(num_obj, model_sum*0.371/num_obj, '.')
ax.set(xlabel='Number of objects in cluster', ylabel='Summed model number for cluster',
       title='Normalized by $\Sigma_{SDFWS}/N_{agn}$')
fig.savefig('Data/MCMC/Mock_Catalog/Plots/Model_Number_sigma_field_Nagn.pdf', format='pdf')


# Maximize the model for the correct parameter set
def model(z, m, r):
    eta_true = 1.2
    zeta_true = -1.0
    beta_true = -1.5

    return -1.0 * (1 + z)**eta_true * (m/1e15)**zeta_true * r**beta_true


result = op.minimize(model, x0=[0., 0., 0.], bounds=[(0.5, 1.7), (0.2e15, 1.8e15), (0.1, 1.5)])
