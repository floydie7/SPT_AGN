"""
Example_cluster_images.py
Author: Benjamin Floyd

Generates a plot of two clusters found near the median redshift of our sample.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import imshow_norm, ZScaleInterval, LinearStretch
from astropy.wcs import WCS


def compass(x, y, size, color, ax):
    """Add a compass to indicate the north and east directions.

    Parameters
    ----------
    x, y : float
        Position of compass vertex in axes coordinates.
    size : float
        Size of compass in axes coordinates.

    """
    xy = x, y
    scale = ax.wcs.pixel_scale_matrix
    scale /= np.sqrt(np.abs(np.linalg.det(scale)))
    return [ax.annotate(label, xy, xy + size * n,
                        ax.transAxes, ax.transAxes,
                        color=color, fontweight='bold',
                        ha='center', va='center',
                        arrowprops=dict(arrowstyle='<-',
                                        shrinkA=0.0, shrinkB=0.0,
                                        color=color, lw=2))
            for n, label, ha, va in zip(scale, 'EN',
                                        ['right', 'center'],
                                        ['center', 'bottom'])]


# Read in the catalog
sptcl_iragn = Table.read('Data/Output/SPTcl_IRAGN.fits')

# Get the median redshift of the sample
med_z = np.median(sptcl_iragn['REDSHIFT'])

# Select clusters near the median redshift
med_z_clusters = sptcl_iragn[np.abs(sptcl_iragn['REDSHIFT'] - med_z) <= 0.05]

# Print the SPT IDs of the most and least massive cluster in our list
# Most massive cluster: SPT-CLJ0212-4657 (SPT-SZ; IRAC obs id: SPT-CLJ0212-4656)
# Least massive cluster: SPT-CLJ2314-5554 (SPTpol 100d)
max_cluster_idx, min_cluster_idx = med_z_clusters['M500'].argmax(), med_z_clusters['M500'].argmin()
max_cluster_id = med_z_clusters['SPT_ID'][max_cluster_idx]
max_cluser_mass = med_z_clusters['M500'][max_cluster_idx]
max_cluster_z = med_z_clusters['REDSHIFT'][max_cluster_idx]
min_cluster_id = med_z_clusters['SPT_ID'][min_cluster_idx]
min_cluser_mass = med_z_clusters['M500'][min_cluster_idx]
min_cluster_z = med_z_clusters['REDSHIFT'][min_cluster_idx]
print(f"""Most massive cluster
{max_cluster_id}\tM500 = {max_cluser_mass:.2e} Msun\tz = {max_cluster_z:.2f}
Least massive cluster
{min_cluster_id}\tM500 = {min_cluser_mass:.2e} Msun\tz = {min_cluster_z:.2f}""")

# Get the catalogs for each of the clusters
spt0212_cat = sptcl_iragn[sptcl_iragn['SPT_ID'] == 'SPT-CLJ0212-4657']
spt2314_cat = sptcl_iragn[sptcl_iragn['SPT_ID'] == 'SPT-CLJ2314-5554']

# Read in the 3.6um images
spt0212_img, spt0212_hdr = fits.getdata('Data/Images/I1_SPT-CLJ0212-4656_mosaic.cutout.fits', header=True)
spt2314_img, spt2314_hdr = fits.getdata('Data/SPTPol/images/cluster_cutouts/I1_SPT-CLJ2314-5554_mosaic.cutout.fits',
                                        header=True)

#%% Make plots
fig = plt.figure(figsize=(16, 8), tight_layout=dict(w_pad=2))
spt0212_ax = fig.add_subplot(121, projection=WCS(spt0212_hdr))
spt2314_ax = fig.add_subplot(122, projection=WCS(spt2314_hdr))
spt0212_ax.set(xlabel='Right Ascension', ylabel='Declination')
spt2314_ax.set(xlabel='Right Ascension', ylabel=' ')
spt2314_ax.coords[0].set_auto_axislabel(False)

# Plot the images
imshow_norm(spt0212_img, ax=spt0212_ax, origin='lower', cmap='Greys', interval=ZScaleInterval(), stretch=LinearStretch())
imshow_norm(spt2314_img, ax=spt2314_ax, origin='lower', cmap='Greys', interval=ZScaleInterval(), stretch=LinearStretch())

# Plot the AGN
spt0212_ax.scatter(spt0212_cat['ALPHA_J2000'], spt0212_cat['DELTA_J2000'], marker='s', edgecolor='w', facecolor='none',
                   s=300, linewidths=3, transform=spt0212_ax.get_transform('world'))
spt2314_ax.scatter(spt2314_cat['ALPHA_J2000'], spt2314_cat['DELTA_J2000'], marker='s', edgecolor='w', facecolor='none',
                   s=300, linewidths=3, transform=spt2314_ax.get_transform('world'))
# # Add compasses
# compass(0.9, 0.1, size=0.1, color='y', ax=spt0212_ax)
# compass(0.9, 0.1, size=0.1, color='y', ax=spt2314_ax)
# plt.tight_layout()
plt.show()
fig.savefig('Data/Plots/Example_clusters_SPT0212_SPT2314.png')
fig.savefig('Data/Plots/Example_clusters_SPT0212_SPT2314.pdf')
