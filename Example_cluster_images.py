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
from matplotlib.gridspec import GridSpec


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


# %%
# Read in the catalog
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN.fits')

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
{max_cluster_id}\tM500 = {max_cluser_mass:.2e} Msun\tz = {max_cluster_z:.2f}\trichness = {len(med_z_clusters[med_z_clusters['SPT_ID'] == max_cluster_id])}\trichness (mu > 0.5) = {len(med_z_clusters[(med_z_clusters['SPT_ID'] == max_cluster_id) & (med_z_clusters['SELECTION_MEMBERSHIP'] >= 0.5)])}
Least massive cluster
{min_cluster_id}\tM500 = {min_cluser_mass:.2e} Msun\tz = {min_cluster_z:.2f}\trichness = {len(med_z_clusters[med_z_clusters['SPT_ID'] == min_cluster_id])}\trichness (mu > 0.5) = {len(med_z_clusters[(med_z_clusters['SPT_ID'] == min_cluster_id) & (med_z_clusters['SELECTION_MEMBERSHIP'] >= 0.5)])}""")

# Get the catalogs for each of the clusters
spt0212_cat = sptcl_iragn[(sptcl_iragn['SPT_ID'] == 'SPT-CLJ0212-4657') & (sptcl_iragn['SELECTION_MEMBERSHIP'] >= 0.5)]
spt2314_cat = sptcl_iragn[(sptcl_iragn['SPT_ID'] == 'SPT-CLJ2314-5554') & (sptcl_iragn['SELECTION_MEMBERSHIP'] >= 0.5)]

# Read in the 3.6um images
spt0212_img, spt0212_hdr = fits.getdata('Data_Repository/Images/SPT/Spitzer_IRAC/SPT-SZ_2500d/'
                                        'I1_SPT-CLJ0212-4656_mosaic.cutout.fits', header=True)
spt2314_img, spt2314_hdr = fits.getdata('Data_Repository/Images/SPT/Spitzer_IRAC/SPTpol_100d/'
                                        'I1_SPT-CLJ2314-5554_mosaic.cutout.fits', header=True)

# %% Make plots
fig = plt.figure(figsize=(16, 8), tight_layout=dict(w_pad=2))
spt0212_ax = fig.add_subplot(121, projection=WCS(spt0212_hdr))
spt2314_ax = fig.add_subplot(122, projection=WCS(spt2314_hdr))
spt0212_ax.set(xlabel='Right Ascension', ylabel='Declination')
spt2314_ax.set(xlabel='Right Ascension', ylabel=' ')
spt2314_ax.coords[0].set_auto_axislabel(False)

# Plot the images
imshow_norm(spt0212_img, ax=spt0212_ax, origin='lower', cmap='Greys', interval=ZScaleInterval(),
            stretch=LinearStretch())
imshow_norm(spt2314_img, ax=spt2314_ax, origin='lower', cmap='Greys', interval=ZScaleInterval(),
            stretch=LinearStretch())

# Plot the AGN
spt0212_ax.scatter(spt0212_cat['ALPHA_J2000'], spt0212_cat['DELTA_J2000'], marker='s', edgecolor='magenta',
                   facecolor='none',
                   s=300, linewidths=3, transform=spt0212_ax.get_transform('world'))
spt2314_ax.scatter(spt2314_cat['ALPHA_J2000'], spt2314_cat['DELTA_J2000'], marker='s', edgecolor='magenta',
                   facecolor='none',
                   s=300, linewidths=3, transform=spt2314_ax.get_transform('world'))
# # Add compasses
# compass(0.9, 0.1, size=0.1, color='y', ax=spt0212_ax)
# compass(0.9, 0.1, size=0.1, color='y', ax=spt2314_ax)
# plt.tight_layout()
plt.show()
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/Example_clusters_SPT0212_SPT2314_magenta.png')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/Example_clusters_SPT0212_SPT2314_magenta_mu0.5.png')
# fig.savefig('Data/Plots/Example_clusters_SPT0212_SPT2314.pdf')

# %% Plot both images of Phoenix Cluster to use as data representatives
I1_img, I1_hdr = fits.getdata('Data_Repository/Images/SPT/Spitzer_IRAC/SPT-SZ_2500d/'
                              'I1_SPT-CLJ2344-4243_mosaic.cutout.fits', header=True)
I2_img, I2_hdr = fits.getdata('Data_Repository/Images/SPT/Spitzer_IRAC/SPT-SZ_2500d/'
                              'I2_SPT-CLJ2344-4243_mosaic.cutout.fits', header=True)

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121, projection=WCS(I1_hdr))
ax2 = fig.add_subplot(122, projection=WCS(I2_hdr), sharey=ax1)
ax1.coords[0].set_axislabel('Right Ascension (J2000)')
ax1.coords[1].set_axislabel('Declination (J2000)')
ax2.coords[1].set_ticklabel_visible(False)
ax2.coords[1].set_auto_axislabel(False)
ax2.coords[0].set_axislabel('Right Ascension (J2000)')
imshow_norm(I1_img, ax=ax1, origin='lower', interval=ZScaleInterval(), stretch=LinearStretch(), cmap='Greys')
imshow_norm(I2_img, ax=ax2, origin='lower', interval=ZScaleInterval(), stretch=LinearStretch(), cmap='Greys')
# ax1.add_patch(Rectangle((170.8157 - w/2, 186.80723 - h/2), 100.8495, 77.665708, fill=False, lw=3, edgecolor='w'))
# ax2.add_patch(Rectangle((170.8157 - w/2, 186.80723 - h/2), 100.8495, 77.665708, fill=False, lw=3, edgecolor='w'))
ax1.text(0.05, 0.05, s=r'3.6 $\mu$m', fontsize=30, color='magenta', transform=ax1.transAxes, zorder=5)
ax2.text(0.05, 0.05, s=r'4.5 $\mu$m', fontsize=30, color='magenta', transform=ax2.transAxes, zorder=5)
plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.05, wspace=0.05)
# plt.show()
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/Example_Phoenix_SPT-CLJ2344-4243_I1_I2.pdf')

# %% Plot all bands for SPT-CLJ2314-5554
pisco_g = fits.getdata('Data_Repository/Images/SPT/PISCO/SPT-CLJ2314-5554/sptclj2314_5554.g.cutoutlarge.coaddwhtd.fits')
pisco_r = fits.getdata('Data_Repository/Images/SPT/PISCO/SPT-CLJ2314-5554/sptclj2314_5554.r.cutoutlarge.coaddwhtd.fits')
pisco_i = fits.getdata('Data_Repository/Images/SPT/PISCO/SPT-CLJ2314-5554/sptclj2314_5554.i.cutoutlarge.coaddwhtd.fits')
pisco_z = fits.getdata('Data_Repository/Images/SPT/PISCO/SPT-CLJ2314-5554/sptclj2314_5554.z.cutoutlarge.coaddwhtd.fits')
irac_1 = fits.getdata('Data_Repository/Images/SPT/Spitzer_IRAC/SPTpol_100d/I1_SPT-CLJ2314-5554_mosaic.cutout.fits')
irac_2 = fits.getdata('Data_Repository/Images/SPT/Spitzer_IRAC/SPTpol_100d/I2_SPT-CLJ2314-5554_mosaic.cutout.fits')

pisco_wcs = WCS('Data_Repository/Images/SPT/PISCO/SPT-CLJ2314-5554/sptclj2314_5554.g.cutoutlarge.coaddwhtd.fits')
irac_wcs = WCS('Data_Repository/Images/SPT/Spitzer_IRAC/SPTpol_100d/I1_SPT-CLJ2314-5554_mosaic.cutout.fits')

fig = plt.figure(figsize=(32, 16))
gs = GridSpec(nrows=2, ncols=4, height_ratios=[1.5, 1],
              left=0.035, right=0.98, top=1, bottom=0.05, wspace=0.05, hspace=0.05)
pisco_axes = [fig.add_subplot(gs[0, i], projection=pisco_wcs) for i in range(4)]
irac_axes = [fig.add_subplot(gs[1, i], projection=irac_wcs) for i in range(1, 3)]

for ax, data in zip([*pisco_axes, *irac_axes], [pisco_g, pisco_r, pisco_i, pisco_z, irac_1, irac_2]):
    imshow_norm(data, ax=ax, interval=ZScaleInterval(), stretch=LinearStretch(), cmap='Greys')
for ax in [*pisco_axes[1:], irac_axes[1]]:
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[1].set_auto_axislabel(False)
for ax in [*pisco_axes, *irac_axes]:
    ax.coords[0].set_axislabel('Right Ascension (J2000)')
for ax, band in zip(pisco_axes, ['g', 'r', 'i', 'z']):
    ax.text(0.9, 0.05, s=band, fontsize='30', fontstyle='italic', color='tab:red', transform=ax.transAxes, zorder=5)
for ax, band in zip(irac_axes, [r'3.6 $\mu$m', r'4.5 $\mu$m']):
    ax.text(0.75, 0.05, s=band, fontsize='30', color='tab:red', transform=ax.transAxes, zorder=5)
pisco_axes[0].coords[1].set_axislabel('Declination (J2000)')
irac_axes[0].coords[1].set_axislabel('Declination (J2000)')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/Misc_Plots/Example_SPT-CLJ2314-5554_PISCO_IRAC.pdf')
plt.show()

# %% Plot HST SNAP images overlayed and split from a single cluster
spt2148_f110w = fits.getdata('Data_Repository/Images/SPT/HST_snap/'
                             'SPT-CLJ2148-4843_F110W_0.03g1.0_cr2.5_0.7_drz_sci.fits.bz2')
spt2148_f200lp = fits.getdata('Data_Repository/Images/SPT/HST_snap/'
                              'SPT-CLJ2148-4843_F200LP_0.03g0.8_cr1.2_0.7_drc_sci.fits.bz2')
w = WCS('Data_Repository/Images/SPT/HST_snap/SPT-CLJ2148-4843_F110W_0.03g1.0_cr2.5_0.7_drz_sci.fits.bz2')

# Set up parameter for the split over the images
combined = np.ones_like(spt2148_f110w)
angle = -np.pi / 3.0
lower_intersection = 0.18
line_width = 50

y, x = spt2148_f110w.shape
yy, xx = np.mgrid[:y, :x]

# Find the pixels from each image that will be plotted on the combined version
f110w_positions = (xx - lower_intersection * x) * np.tan(angle) - line_width // 2 > (yy - y)
f200lp_positions = (xx - lower_intersection * x) * np.tan(angle) + line_width // 2 < (yy - y)

# Merge the images
combined[f110w_positions] = spt2148_f110w[f110w_positions]
combined[f200lp_positions] = spt2148_f200lp[f200lp_positions]

# %%
fig, ax = plt.subplots(subplot_kw=dict(projection=w))
imshow_norm(combined, ax=ax, origin='lower', cmap='Greys', interval=ZScaleInterval(), stretch=LinearStretch())
ax.coords[0].set_axislabel('Right Ascension (J2000)')
ax.coords[1].set_axislabel('Declination (J2000)')
ax.text(0.01, 0.02, s='F110W', fontsize=15, color='k', transform=ax.transAxes, zorder=5)
ax.text(0.8, 0.92, s='F200LP', fontsize=15, color='k', transform=ax.transAxes, zorder=5)
plt.show()
