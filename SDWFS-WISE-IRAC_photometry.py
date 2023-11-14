"""
SDWFS-WISE-IRAC_photometry.py
Author: Benjamin Floyd

Matches the IRAC and CatWISE catalogs for SDWFS and compares the colors within the photometric ranges used for IR-AGN.
"""

import arviz as az
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import imshow_norm, ZScaleInterval, LinearStretch
from astropy.wcs import WCS

# Read in the IRAC catalog
irac_catalog = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/'
                          'ch2v33_sdwfs_2009mar3_apcorr_matched_ap4_Main_v0.4.cat.gz',
                          names=['ID', 'ALPHA_J2000', 'DELTA_J2000',
                                 'B_APFLUX4', 'R_APFLUX4', 'I_APFLUX4',
                                 'B_APFLUXERR4', 'R_APFLUXERR4', 'I_APFLUXERR4',
                                 'B_APMAG4', 'R_APMAG4', 'I_APMAG4',
                                 'B_APMAGERR4', 'R_APMAGERR4', 'I_APMAGERR4',
                                 'I1_FLUX_APER4', 'I2_FLUX_APER4', 'I3_FLUX_APER4', 'I4_FLUX_APER4',
                                 'I1_FLUXERR_APER4', 'I2_FLUXERR_APER4', 'I3_FLUXERR_APER4', 'I4_FLUXERR_APER4',
                                 'I1_FLUX_APER4_BROWN', 'I2_FLUX_APER4_BROWN', 'I3_FLUX_APER4_BROWN',
                                 'I4_FLUX_APER4_BROWN',
                                 'I1_MAG_APER4', 'I2_MAG_APER4', 'I3_MAG_APER4', 'I4_MAG_APER4',
                                 'I1_MAGERR_APER4', 'I2_MAGERR_APER4', 'I3_MAGERR_APER4', 'I4_MAGERR_APER4',
                                 'I1_MAGERR_APER4_BROWN', 'I2_MAGERR_APER4_BROWN', 'I3_MAGERR_APER4_BROWN',
                                 'I4_MAGERR_APER4_BROWN',
                                 'STARS_COLOR', 'STARS_MORPH', 'CLASS_STAR', 'MBZ_FLAG_4_4_4'], format='ascii')

# Read in the WISE catalog
wise_catalog = Table.read('Data_Repository/Catalogs/Bootes/SDWFS/SDWFS_catWISE.ecsv')

# Select SNR >= 5 objects
irac_catalog_snr = irac_catalog[(irac_catalog['I1_FLUX_APER4'] / irac_catalog['I1_FLUXERR_APER4'] >= 5) &
                                (irac_catalog['I2_FLUX_APER4'] / irac_catalog['I2_FLUXERR_APER4'] >= 5)]
wise_catalog_snr = wise_catalog[(wise_catalog['w1flux'] / wise_catalog['w1sigflux'] >= 5) &
                                (wise_catalog['w2flux'] / wise_catalog['w2sigflux'] >= 5)]

#%% Quick plot to check the relative magnitude turn overs
turnover_mag_bins = np.arange(10., 20., 0.25)
fig, ax = plt.subplots()
ax.hist(irac_catalog_snr['I2_MAG_APER4'], bins=turnover_mag_bins, label='IRAC', alpha=0.4)
ax.hist(wise_catalog_snr['w2mpro'], bins=turnover_mag_bins, label='WISE', alpha=0.4)
ax.axvline(x=17.48, ls='--', c='k')
ax.legend()
ax.set(title='SDWFS Galaxies', xlabel='[4.5] (W2) (Vega)', ylabel='number', yscale='log')
# fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/SDWFS/SDWFS_IRAC-WISE_mag_turnover.pdf')
plt.show()

#%% Select objects within our magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

# irac_catalog = irac_catalog[(ch1_bright_mag < irac_catalog['I1_MAG_APER4']) &
#                             (irac_catalog['I1_MAG_APER4'] <= ch1_faint_mag) &
#                             (ch2_bright_mag < irac_catalog['I2_MAG_APER4']) &
#                             (irac_catalog['I2_MAG_APER4'] <= ch2_faint_mag)]
# wise_catalog = wise_catalog[(ch1_bright_mag < wise_catalog['w1mpro']) & (wise_catalog['w1mpro'] <= ch1_faint_mag) &
#                             (ch2_bright_mag < wise_catalog['w2mpro']) & (wise_catalog['w2mpro'] <= ch2_faint_mag)]

# Match the catalogs together
irac_coords = SkyCoord(irac_catalog['ALPHA_J2000'], irac_catalog['DELTA_J2000'], unit=u.deg)
wise_coords = SkyCoord(wise_catalog['ra'], wise_catalog['dec'], unit=u.deg)

wise_idx, sep, _ = irac_coords.match_to_catalog_sky(wise_coords)

# Check the separation distribution
fig, ax = plt.subplots()
ax.hist(sep.to_value(u.arcsec), bins='auto')
ax.set(xlabel='separation [arcsec]', xlim=[0, 3])
plt.show()

#%% Select the matched objects
max_separation = 1 * u.arcsec
sep_constraint = sep < max_separation
irac_matches = irac_catalog[sep_constraint]
wise_matches = wise_catalog[wise_idx[sep_constraint]]

#%% Matched Magnitude turnovers

# snr cuts
irac_matches_snr = irac_matches[(irac_matches['I1_FLUX_APER4'] / irac_matches['I1_FLUXERR_APER4'] >= 5) &
                                (irac_matches['I2_FLUX_APER4'] / irac_matches['I2_FLUXERR_APER4'] >= 5)]
wise_matches_snr = wise_matches[(wise_matches['w1flux'] / wise_matches['w1sigflux'] >= 5) &
                                (wise_matches['w2flux'] / wise_matches['w2sigflux'] >= 5)]

turnover_mag_bins = np.arange(10., 20., 0.25)
fig, ax = plt.subplots()
ax.hist(irac_matches_snr['I2_MAG_APER4'], bins=turnover_mag_bins, label='IRAC', alpha=0.4)
ax.hist(wise_matches_snr['w2mpro'], bins=turnover_mag_bins, label='WISE', alpha=0.4)
ax.axvline(x=17.48, ls='--', c='k')
ax.legend()
ax.set(title='SDWFS Galaxies', xlabel='[4.5] or W2 (Vega)', ylabel='number', yscale='log')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/SDWFS/SDWFS_IRAC-WISE_mag_turnover_matches_snr5.pdf')
plt.show()

#%% Plot the galaxies on the Boötes footprint
i1_img, i1_hdr = fits.getdata('Data_Repository/Images/Bootes/SDWFS/I1_bootes.v32.fits', header=True)
i1_wcs = WCS(i1_hdr)

#%%
fig, ax = plt.subplots(figsize=(9, 12), subplot_kw={'projection': i1_wcs})
imshow_norm(i1_img, ax=ax, origin='lower', cmap='Greys', interval=ZScaleInterval(), stretch=LinearStretch())
ax.scatter(irac_catalog_snr['ALPHA_J2000'], irac_catalog_snr['DELTA_J2000'], marker='o', s=10, fc='none', ec='tab:blue',
           alpha=0.2, label='IRAC Galaxies', transform=ax.get_transform('world'))
ax.scatter(wise_catalog_snr['ra'], wise_catalog_snr['dec'], marker='s', s=10, fc='none', ec='tab:orange', alpha=0.2,
           label='WISE Galaxies',
           transform=ax.get_transform('world'))
ax.legend(loc='lower left')
ax.set(xlabel='Right Ascension', ylabel='Declination')
plt.tight_layout()
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/SDWFS/SDWFS_IRAC-WISE_footprint.png')
plt.show()

#%% Plot the magnitude differences
fig, (ax, bx) = plt.subplots(ncols=2, sharey='row', figsize=(6.4*2, 4.8))
# ax.scatter(irac_matches['I1_MAG_APER4'], irac_matches['I1_MAG_APER4'] - wise_matches['w1mpro'], marker='.')
# bx.scatter(irac_matches['I2_MAG_APER4'], irac_matches['I2_MAG_APER4'] - wise_matches['w2mpro'], marker='.')
sns.kdeplot(x=irac_matches['I1_MAG_APER4'], y=irac_matches['I1_MAG_APER4'] - wise_matches['w1mpro'], ax=ax)
sns.kdeplot(x=irac_matches['I2_MAG_APER4'], y=irac_matches['I2_MAG_APER4'] - wise_matches['w2mpro'], ax=bx)
ax.axhline(0.0, ls='--', c='k')
bx.axhline(0.0, ls='--', c='k')
ax.set(xlabel=r'[3.6 $\mu$m]', ylabel=r'[3.6 $\mu$m] - $W1$')
bx.set(xlabel=r'[4.5 $\mu$m]', ylabel=r'[4.5 $\mu$m] - $W2$')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/SDWFS/SDWFS_IRAC-WISE_mag_diffs_cont.pdf')
plt.show()

#%% Plot the color differences
color_diff = ((irac_matches['I1_MAG_APER4'] - irac_matches['I2_MAG_APER4'])
              - (wise_matches['w1mpro'] - wise_matches['w2mpro']))
hdi_color_diff = az.hdi(color_diff, hdi_prob=0.68)

irac_color_err = np.sqrt((2.5 * irac_matches['I1_FLUXERR_APER4'] / (irac_matches['I1_FLUX_APER4'] * np.log(10))) ** 2 +
                         (2.5 * irac_matches['I2_FLUXERR_APER4'] / (irac_matches['I2_FLUX_APER4'] * np.log(10))) ** 2)
wise_color_err = np.sqrt((2.5 * wise_matches['w1sigflux'] / (wise_matches['w1flux'] * np.log(10))) ** 2 +
                         (2.5 * wise_matches['w2sigflux'] / (wise_matches['w2flux'] * np.log(10))) ** 2)



#%%
fig, ax = plt.subplots()
hist, bins, _ = ax.hist(color_diff, bins='auto', histtype='step')
mode_color_diff = bins[np.argmax(hist)]
hdi_lvls = np.diff([hdi_color_diff[0], mode_color_diff, hdi_color_diff[1]])
# sns.kdeplot(color_diff, ax=ax)
ax.axvline(0.0, ls='--', c='k')
ax.axvline(mode_color_diff, ls='-', label=fr'${mode_color_diff:.2f}_{{-{hdi_lvls[0]:.2f}}}^{{+{hdi_lvls[1]:.2f}}}$')
ax.axvspan(hdi_color_diff[0], hdi_color_diff[1], alpha=0.3)
ax.legend()
ax.set(xlabel=r'IRAC - WISE Colors')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/SDWFS/SDWFS_IRAC-WISE_color_diffs_hist.pdf')
plt.show()

#%%
fig, ax = plt.subplots()
# ax.scatter(irac_matches['I1_MAG_APER4'] - irac_matches['I2_MAG_APER4'], color_diff, marker='.', alpha=0.2)
sns.kdeplot(x=irac_matches['I1_MAG_APER4'] - irac_matches['I2_MAG_APER4'], y=color_diff, ax=ax)
ax.axhline(0.0, ls='--', c='k')
ax.set(xlabel=r'[3.6 $\mu$m] - [4.5 $\mu$m]', ylabel='IRAC - WISE Colors')
fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/plots/SDWFS/SDWFS_IRAC-WISE_color_diffs_cont.pdf')
plt.show()
