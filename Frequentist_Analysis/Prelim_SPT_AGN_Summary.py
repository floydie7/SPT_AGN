"""
Prelim_SPT_AGN_Summary.py
Author: Benjamin Floyd

Calculates the mean AGN surface density of the non-background subtracted AGN catalogs.
"""

from __future__ import print_function, division
from astropy.table import Table
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Collect the file names
cat_files = [f for f in listdir('Data/Output/') if not f.startswith('.')]

# Read the catalogs in
catalogs = [Table.read('Data/Output/'+f, format='ascii') for f in listdir('Data/Output/') if not f.startswith('.')]

# Collect the raw number of objects, the corrected number of objects and the surface densities.
num = []
corr = []
surf_den = []
for cat in catalogs:
    num.append(len(cat))
    corr.append(cat['completeness_correction'].sum())
    surf_den.append(cat['completeness_correction'].sum() / cat['IMAGE_AREA'][0])

surf_den.append(0.0)  # To account for SPT-CLJ0154-4824 having no AGN present
surf_den.append(0.0)  # To account for SPT-CLJ2311-4203 having no AGN present

# Calculate the means
mean_num = np.mean(num)
mean_corr = np.mean(corr)
mean_surf_den = np.mean(surf_den)
std_surf_den = np.std(surf_den)

print('''Number of catalogs {cats}
Mean number of objects: {num:.3f}
Mean number of completeness corrected objects: {corr:.3f}
Mean AGN surface density: {surf:.3f} per sq. arcmin
Standard Deviation of AGN surface density: {std_surf:.3f} per sq. arcmin'''
      .format(cats=len(catalogs), num=mean_num, corr=mean_corr, surf=mean_surf_den, std_surf=std_surf_den))



surf_den_hist, bin_edges = np.histogram(surf_den, bins='auto')
mle = bin_edges[np.argmax(surf_den_hist)]+(bin_edges[np.argmax(surf_den_hist)+1] - bin_edges[np.argmax(surf_den_hist)])/2.
print(mle)

bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2.

result = skewnorm.fit(surf_den, floc=mle)
region = skewnorm.interval(.68, a=result[0], loc=result[1], scale=result[2])
print(result)

rv = skewnorm(result[0], loc=result[1], scale=result[2])

fig, ax = plt.subplots()
ax.hist(surf_den, bins='auto', align='mid')
# ax.plot(bin_centers, rv.pdf(bin_centers), 'k-')
# ax.axvline(x=region[0], linestyle='--', color='k')
# ax.axvline(x=region[1], linestyle='--', color='k')
ax.axvline(x=mle, linestyle='-', color='r', label=r'MLE = {:.3f} arcmin$^{{-2}}$'.format(mle))
ax.set(title='SPT Global Surface Densities', xlabel='$\Sigma_{\mathrm{AGN}}$ [arcmin$^{-2}$]', xlim=[0, 1.25])
ax.legend()
fig.savefig('Data/Plots/SPT_Global_Surface_Density.pdf', format='pdf')
# plt.show()

