"""
Problem_Mask_Clusters_visualization.py
Author: Benjamin Floyd

Creates visualizations of the clusters that have suspicious masks used in the real data and mock catalogs.
"""
import glob
import numpy as np
from astropy.io import fits
from astropy.visualization import imshow_norm, ZScaleInterval, LinearStretch
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import warnings
from astropy.utils.exceptions import AstropyWarning  # For suppressing the astropy warnings.

warnings.simplefilter('ignore', category=AstropyWarning)

zscale = ZScaleInterval(nsamples=600, contrast=0.25)

spt_id = ['SPT-CLJ0002-5557',
          'SPT-CLJ0048-6416',
          'SPT-CLJ0058-6145',
          'SPT-CLJ0107-5833',
          'SPT-CLJ0131-5604',
          'SPT-CLJ0131-5921',
          'SPT-CLJ0147-5622',
          'SPT-CLJ0151-5954',
          'SPT-CLJ0156-5541',
          'SPT-CLJ0202-5401',
          'SPT-CLJ0203-5651',
          'SPT-CLJ0208-4425',
          'SPT-CLJ0238-4904',
          'SPT-CLJ0231-5403',
          'SPT-CLJ0258-5355',
          'SPT-CLJ0313-5334',
          'SPT-CLJ0329-4029',
          'SPT-CLJ0415-4621',
          'SPT-CLJ0428-6049',
          'SPT-CLJ0500-4713',
          'SPT-CLJ0536-6109',
          'SPT-CLJ0552-4937',
          'SPT-CLJ0559-6022',
          'SPT-CLJ0637-4327',
          'SPT-CLJ0642-6310']
cluster_dict = {}
for cluster in spt_id:
    cluster_dict[cluster] = {}
    for file in glob.glob('Data/Images/*{}*.fits'.format(cluster)):
        if 'I1' in file and '_cov' not in file:
            cluster_dict[cluster]['I1_sci'] = fits.getdata(file, ignore_missing_end=True)
        elif 'I1' in file and '_cov' in file:
            cluster_dict[cluster]['I1_cov'] = fits.getdata(file, ignore_missing_end=True)
        elif 'I2' in file and '_cov' not in file:
            cluster_dict[cluster]['I2_sci'] = fits.getdata(file, ignore_missing_end=True)
        else:
            cluster_dict[cluster]['I2_cov'] = fits.getdata(file, ignore_missing_end=True)

    for file in glob.glob('Data/Masks/*{}*.fits'.format(cluster)):
        cluster_dict[cluster]['mask'] = fits.getdata(file, ignore_missing_end=True)
        cluster_dict[cluster]['wcs'] = WCS(file)

for cluster_id in cluster_dict.keys():
    print(cluster_id)

    fig, axarr = plt.subplots(2, 4, sharex='col', sharey='row', subplot_kw={'projection': cluster_dict[cluster_id]['wcs']},
                              figsize=(11, 5.8))

    gs = axarr[0, 0].get_gridspec()
    for ax in axarr[0:, 0:2].flatten():
        ax.remove()

    ax_mask = fig.add_subplot(gs[0:, 0:2], projection=cluster_dict[cluster_id]['wcs'])
    ax_mask.imshow(cluster_dict[cluster_id]['mask'], origin='lower', cmap='gray_r')
    imshow_norm(cluster_dict[cluster_id]['I1_sci'], axarr[0, 2], origin='lower', cmap='gray_r', interval=zscale, stretch=LinearStretch())
    imshow_norm(cluster_dict[cluster_id]['I2_sci'], axarr[0, 3], origin='lower', cmap='gray_r', interval=zscale, stretch=LinearStretch())
    imshow_norm(cluster_dict[cluster_id]['I1_cov'], axarr[1, 2], origin='lower', cmap='gray_r', interval=zscale, stretch=LinearStretch())
    imshow_norm(cluster_dict[cluster_id]['I2_cov'], axarr[1, 3], origin='lower', cmap='gray_r', interval=zscale, stretch=LinearStretch())

    ra_mask = ax_mask.coords['ra']
    dec_mask = ax_mask.coords['dec']
    ra_mask.set_ticks_visible(False)
    ra_mask.set_ticklabel_visible(False)
    dec_mask.set_ticks_visible(False)
    dec_mask.set_ticklabel_visible(False)
    for ax in axarr.flatten():
        ra = ax.coords['ra']
        dec = ax.coords['dec']
        ra.set_ticks_visible(False)
        ra.set_ticklabel_visible(False)
        dec.set_ticks_visible(False)
        dec.set_ticklabel_visible(False)

    fig.suptitle('{}'.format(cluster_id))
    ax_mask.set(title='Mask')
    axarr[0, 2].set(title='IRAC [3.6]')
    axarr[0, 3].set(title='IRAC [4.5]')

    fig.savefig('Data/MCMC/Mock_Catalog/Plots/Poisson_Likelihood/pre-final_tests/mask_generation_checks/{}_masking_check.pdf'.format(cluster_id), format='pdf')
