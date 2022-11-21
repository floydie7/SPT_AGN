"""
SPTSZ-SPTpol_catalog_comparison.py
Author: Benjamin Floyd

Compares the targeted vs SSDF catalogs by applying both common footprint and magnitude cuts to investigate differences
in number counts of the two catalogs.
"""

import glob
import json
import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, setdiff, vstack
from astropy.visualization import imshow_norm, LinearStretch, ZScaleInterval
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d


def sanitize_fluxes(channel, targeted_cat, ssdf_cat):
    # To handle non-detections in I1, we will set all "-99.0" entries to the 1-sigma flux error.
    targeted_flux_limit = targeted_I1_flux_err if channel == 1 else targeted_I2_flux_err
    ssdf_flux_limit = ssdf_I1_flux_err if channel == 1 else ssdf_I2_flux_err
    targeted_cat[f'I{channel}_FLUX_APER4'] = np.where((targeted_cat[f'I{channel}_FLUX_APER4'] == -99.) |
                                                      (targeted_cat[f'I{channel}_FLUX_APER4'] == 0.),
                                                      targeted_flux_limit, targeted_cat[f'I{channel}_FLUX_APER4'])
    ssdf_cat[f'I{channel}_FLUX_APER4'] = np.where((ssdf_cat[f'I{channel}_FLUX_APER4'] == -99.) |
                                                  (ssdf_cat[f'I{channel}_FLUX_APER4'] == 0.),
                                                  ssdf_flux_limit, ssdf_cat[f'I{channel}_FLUX_APER4'])
    return targeted_cat, ssdf_cat


def sanitize_fluxes_merged(channel, merged_catalog, targeted_flux_limit, ssdf_flux_limit, err=False):
    error_string = 'ERR' if err else ''

    merged_catalog[f'TARGETED_I{channel}_MAG{error_string}'] = np.where(
        (merged_catalog[f'TARGETED_I{channel}_MAG{error_string}'] == -99.) |
        (merged_catalog[f'TARGETED_I{channel}_MAG{error_string}'] == 0.),
        targeted_flux_limit,
        merged_catalog[f'TARGETED_I{channel}_MAG{error_string}'])
    merged_catalog[f'SSDF_I{channel}_MAG{error_string}'] = np.where(
        (merged_catalog[f'SSDF_I{channel}_MAG{error_string}'] == -99.) |
        (merged_catalog[f'SSDF_I{channel}_MAG{error_string}'] == 0.),
        ssdf_flux_limit, merged_catalog[f'SSDF_I{channel}_MAG{error_string}'])

    return merged_catalog


def add_vega_axes(ax, channel, xaxis=True, color=False, mag_diff=False):
    if not color and not mag_diff:
        # Add a magnitude scale at the top
        f_v0 = 280.9 * u.Jy if channel == 1 else 179.7 * u.Jy  # Zero-point flux
        zpt_mag = 2.5 * np.log10(f_v0.to_value(u.uJy))
    else:
        zpt_mag = 0.

    if color:
        label = '[3.6] - [4.5] (Vega)'
    elif mag_diff:
        label = (fr'[{"3.6" if channel == 1 else "4.5"}$_{{\rm Targeted}}$]'
                 fr'- [{"3.6" if channel == 1 else "4.5"}$_{{\rm SSDF}}$]')
    else:
        label = 'Vega Magnitude'

    if xaxis:
        ax_mag_top = ax.twiny()
        x_logflux_min, x_logflux_max = ax.get_xlim()
        ax_mag_top.set_xlim(-2.5 * x_logflux_min + zpt_mag, -2.5 * x_logflux_max + zpt_mag)
        ax_mag_top.xaxis.set_major_locator(MultipleLocator(1))
        ax_mag_top.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax_mag_top.set(xlabel=label)
        return ax_mag_top
    else:
        ax_mag_right = ax.twinx()
        y_logflux_min, y_logflux_max = ax.get_ylim()
        ax_mag_right.set_ylim(-2.5 * y_logflux_min + zpt_mag, -2.5 * y_logflux_max + zpt_mag)
        ax_mag_right.yaxis.set_major_locator(MultipleLocator(1))
        ax_mag_right.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax_mag_right.set(ylabel=label)
        return ax_mag_right


def source_count_plot(ax, channel, targeted_cat, ssdf_cat, sigma_levels=None, targeted_agn_cat=None, ssdf_agn_cat=None,
                      cumulative=-1):
    # Sanitize the data
    targeted_cat, ssdf_cat = sanitize_fluxes(channel=channel, targeted_cat=targeted_cat, ssdf_cat=ssdf_cat)

    stacked_data = np.hstack((np.log10(targeted_cat[f'I{channel}_FLUX_APER4']),
                              np.log10(ssdf_cat[f'I{channel}_FLUX_APER4'])))
    bins = np.histogram_bin_edges(stacked_data, bins='auto')
    # bins = np.arange(np.min(stacked_data), np.max(stacked_data), 0.25)
    ax.hist(np.log10(targeted_cat[f'I{channel}_FLUX_APER4']), bins=bins, cumulative=cumulative, histtype='step',
            log=True, color='C0', label='Targeted')
    ax.hist(np.log10(ssdf_cat[f'I{channel}_FLUX_APER4']), bins=bins, cumulative=cumulative, histtype='step', log=True,
            color='C1', label='SSDF')
    if targeted_agn_cat is not None:
        ax.hist(np.log10(targeted_agn_cat[f'I{channel}_FLUX_APER4']), bins=bins, cumulative=cumulative, histtype='bar',
                log=True, color='C0', label='Targeted AGN', alpha=0.6)
    if ssdf_agn_cat is not None:
        ax.hist(np.log10(ssdf_agn_cat[f'I{channel}_FLUX_APER4']), bins=bins, cumulative=cumulative, histtype='bar',
                log=True, color='C1', label='SSDF AGN', alpha=0.4)
    if sigma_levels is not None:
        sigma_3, sigma_5, sigma_10 = np.log10(sigma_levels)
        ax.axvline(x=sigma_3, color='k', ls='dotted')
        ax.axvline(x=sigma_5, color='k', ls='dashdot')
        ax.axvline(x=sigma_10, color='k', ls='dashed')
    ax.legend()

    if channel == 1:
        ax.set(xlabel=r'$\log\/S_{{3.6\mu\rm m}}\/[\mu\rm Jy]$', ylabel=r'$N(>S_{{3.6\mu\rm m}})$')
    else:
        ax.set(xlabel=r'$\log\/S_{{4.5\mu\rm m}}\/[\mu\rm Jy]$', ylabel=r'$N(>S_{{4.5\mu\rm m}})$')

    # Add Vega magnitude axes
    add_vega_axes(ax=ax, channel=channel)


def plot_image(fig, targeted_cat, ssdf_cat, targeted_agn_cat=None, ssdf_agn_cat=None):
    # Read in the targeted I2 image
    image_name = ''.join(s for s in image_names if off_to_obs_ids[cluster_id] in s)
    image, header = fits.getdata(image_name, header=True)
    wcs = WCS(header)

    # Separate the AGN from galaxies
    if targeted_agn_cat:
        targeted_galaxies = setdiff(targeted_cat, targeted_agn_cat)
    else:
        targeted_galaxies = targeted_cat
    if ssdf_agn_cat:
        ssdf_galaxies = setdiff(ssdf_cat, ssdf_agn_cat)
    else:
        ssdf_galaxies = ssdf_cat

    # Plot the objects on the deep targeted image
    ax = fig.add_subplot(projection=wcs)
    imshow_norm(image, ax=ax, origin='lower', cmap='Greys', interval=ZScaleInterval(), stretch=LinearStretch())

    # Plot the galaxies (excluding AGN if provided)
    ax.scatter(targeted_galaxies['ALPHA_J2000'], targeted_galaxies['DELTA_J2000'], marker='s', edgecolor='w',
               facecolor='none', s=55, linewidths=1, transform=ax.get_transform('world'), label='Targeted')
    ax.scatter(ssdf_galaxies['ALPHA_J2000'], ssdf_galaxies['DELTA_J2000'], marker='o', edgecolor='r',
               facecolor='none', s=50, linewidths=1, transform=ax.get_transform('world'), label='SSDF')

    # Plot the AGN separately if provided
    if targeted_agn_cat:
        ax.scatter(targeted_agn_cat['ALPHA_J2000'], targeted_agn_cat['DELTA_J2000'], marker='*', edgecolor='w',
                   facecolor='none',
                   s=65, linewidths=1, transform=ax.get_transform('world'), label='Targeted AGN')
    if ssdf_agn_cat:
        ax.scatter(ssdf_agn_cat['ALPHA_J2000'], ssdf_agn_cat['DELTA_J2000'], marker='*', edgecolor='r',
                   facecolor='none', s=60,
                   linewidths=1, transform=ax.get_transform('world'), label='SSDF AGN')

    # Plot the common footprint boundary used for matching.
    ax.add_patch(SphericalCircle(center=(center['RA_Huang'] * u.deg, center['DEC_Huang'] * u.deg),
                                 radius=1.5 * u.arcmin, edgecolor='green', facecolor='none', linestyle='--',
                                 transform=ax.get_transform('world')))
    ax.legend()
    ax.set(title=fr'Targeted $4.5 \mu\rm m$ image of {cluster_id}', xlabel='Right Ascension', ylabel='Declination')


def flux_comparison_plot(ax, channel, targeted_cat, ssdf_cat):
    # Merge the catalogs into a unified catalog
    merged_catalog, ssdf_nonmatches, targeted_matches, targeted_nonmatches = merge_catalogs(channel, targeted_cat,
                                                                                            ssdf_cat,
                                                                                            matches_nonmatches=True)

    # Do the same as well for the colors where we prefer the colors from the targeted catalog for common matches
    merged_catalog['I1-I2_COLOR'] = np.concatenate([targeted_matches['I1_MAG_APER4'] - targeted_matches['I2_MAG_APER4'],
                                                    targeted_nonmatches['I1_MAG_APER4'] - targeted_nonmatches[
                                                        'I2_MAG_APER4'],
                                                    ssdf_nonmatches['I1_MAG_APER4'] - ssdf_nonmatches['I2_MAG_APER4']])

    # To handle non-detections/non-matches, we will set all "-99.0" entries to a magnitude that is outside our cuts.
    targeted_mag_limit = 20.
    ssdf_mag_limit = 20.
    merged_catalog = sanitize_fluxes_merged(channel=channel, merged_catalog=merged_catalog,
                                            targeted_flux_limit=targeted_mag_limit, ssdf_flux_limit=ssdf_mag_limit,
                                            err=False)

    # We need a similar fix for the mag errors as well. Here just set them to zero
    merged_catalog = sanitize_fluxes_merged(channel=channel, merged_catalog=merged_catalog,
                                            targeted_flux_limit=0., ssdf_flux_limit=0.,
                                            err=True)

    ax.axline(xy1=[0, 0], slope=1, ls='--', c='k', alpha=0.4)
    cm = ax.scatter(merged_catalog[f'TARGETED_I{channel}_MAG'],
                    merged_catalog[f'SSDF_I{channel}_MAG'],
                    c=merged_catalog['I1-I2_COLOR'], cmap='RdBu_r',
                    norm=TwoSlopeNorm(vcenter=0.7, vmin=0, vmax=3), zorder=3)
    ax.errorbar(merged_catalog[f'TARGETED_I{channel}_MAG'],
                merged_catalog[f'SSDF_I{channel}_MAG'],
                xerr=merged_catalog[f'TARGETED_I{channel}_MAGERR'],
                yerr=merged_catalog[f'SSDF_I{channel}_MAGERR'], fmt='none', zorder=0)

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))

    # # Add a magnitude scale at the top
    # add_vega_axes(ax=ax, channel=channel, xaxis=True)
    #
    # # And one to the right
    # add_vega_axes(ax=ax, channel=channel, xaxis=False)

    # Set the axes labels
    if channel == 1:
        ax.set(xlabel=r'Targeted $[3.6\mu\rm m]\/(Vega)$',
               ylabel=r'SSDF $[3.6\mu\rm m]\/(Vega)$')
    else:
        ax.set(xlabel=r'Targeted $[4.5\mu\rm m]\/(Vega)$',
               ylabel=r'SSDF $[4.5\mu\rm m]\/(Vega)$')
    ax.set(xlim=[20.5, 9.5], ylim=[20.5, 9.5])

    return cm


def merge_catalogs(channel, targeted_cat, ssdf_cat, matches_nonmatches=False):
    targeted_coord = SkyCoord(targeted_cat['ALPHA_J2000'], targeted_cat['DELTA_J2000'], unit=u.deg)
    ssdf_coord = SkyCoord(ssdf_cat['ALPHA_J2000'], ssdf_cat['DELTA_J2000'], unit=u.deg)
    # Match the catalogs
    idx, sep, _ = targeted_coord.match_to_catalog_sky(ssdf_coord)
    sep_constraint = sep <= 1 * u.arcsec
    targeted_matches = targeted_cat[sep_constraint]
    ssdf_matches = ssdf_cat[idx[sep_constraint]]
    # Also identify the non-matches
    targeted_nonmatches = setdiff(targeted_cat, targeted_matches)
    ssdf_nonmatches = setdiff(ssdf_cat, ssdf_matches)
    # Concatenate the mag columns using the scheme: (common matches, targeted non-matches, ssdf non-matches)
    targeted_mag = np.concatenate([targeted_matches[f'I{channel}_MAG_APER4'],
                                    targeted_nonmatches[f'I{channel}_MAG_APER4'],
                                    np.full_like(ssdf_nonmatches[f'I{channel}_MAG_APER4'], -99.)])
    ssdf_mag = np.concatenate([ssdf_matches[f'I{channel}_MAG_APER4'],
                                np.full_like(targeted_nonmatches[f'I{channel}_MAG_APER4'], -99.),
                                ssdf_nonmatches[f'I{channel}_MAG_APER4']])
    # Do the same for the mag errors
    targeted_mag_err = np.concatenate([targeted_matches[f'I{channel}_MAGERR_APER4'],
                                        targeted_nonmatches[f'I{channel}_MAGERR_APER4'],
                                        np.full_like(ssdf_nonmatches[f'I{channel}_MAGERR_APER4'], -99.)])
    ssdf_mag_err = np.concatenate([ssdf_matches[f'I{channel}_MAGERR_APER4'],
                                    np.full_like(targeted_nonmatches[f'I{channel}_MAGERR_APER4'], -99.),
                                    ssdf_nonmatches[f'I{channel}_MAGERR_APER4']])
    # Finally, collect everything into a single table
    merged_catalog = Table([targeted_mag, targeted_mag_err, ssdf_mag, ssdf_mag_err],
                           names=[f'TARGETED_I{channel}_MAG', f'TARGETED_I{channel}_MAGERR', f'SSDF_I{channel}_MAG',
                                  f'SSDF_I{channel}_MAGERR'])
    if matches_nonmatches:
        return merged_catalog, ssdf_nonmatches, targeted_matches, targeted_nonmatches
    else:
        return merged_catalog


def stacked_flux_comparison_plot(ax, channel, targeted_cat, ssdf_cat):
    # Merge the catalogs into a unified catalog
    merged_catalog = merge_catalogs(channel, targeted_cat, ssdf_cat)

    # To handle non-detections/non-matches, we will set all "-99.0" entries to a log-flux of 0
    merged_catalog = sanitize_fluxes_merged(channel=channel, merged_catalog=merged_catalog,
                                            targeted_flux_limit=20., ssdf_flux_limit=20., err=False)

    # We need a similar fix for the flux errors as well
    merged_catalog = sanitize_fluxes_merged(channel=channel, merged_catalog=merged_catalog,
                                            targeted_flux_limit=0., ssdf_flux_limit=0., err=True)

    # Plot the points with error bars
    _, _, (xebar, yebar) = ax.errorbar(merged_catalog[f'TARGETED_I{channel}_MAG'],
                                       merged_catalog[f'SSDF_I{channel}_MAG'],
                                       xerr=merged_catalog[f'TARGETED_I{channel}_MAGERR'],
                                       yerr=merged_catalog[f'SSDF_I{channel}_MAGERR'], fmt='.', color='k')
    xebar.set_alpha(0.5)
    yebar.set_alpha(0.5)

    # Draw the 1:1 line for reference
    ax.axline(xy1=[0, 0], slope=1, ls='--', c='k', alpha=0.4)

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))

    # # Add a magnitude scale at the top
    # ax_mag_top = add_vega_axes(ax=ax, channel=channel, xaxis=True)
    #
    # # And the same on the right
    # ax_mag_right = add_vega_axes(ax=ax, channel=channel, xaxis=False)

    # Plot the mag limits using the magnitude axes and set the axes labels for the main axes
    if channel == 1:
        ax.axvline(x=10., ls='--', color='k', alpha=0.4)
        ax.axhline(y=10., ls='--', color='k', alpha=0.4)
        ax.axvline(x=18.3, ls='--', color='k', alpha=0.4)
        ax.axhline(y=18.3, ls='--', color='k', alpha=0.4)

        ax.set(xlabel=r'Targeted $[3.6\mu\rm m]\/(Vega)$',
               ylabel=r'SSDF $[3.6\mu\rm m]\/(Vega)$')
    else:
        ax.axvline(x=10.45, ls='--', color='k', alpha=0.4)
        ax.axhline(y=10.45, ls='--', color='k', alpha=0.4)
        ax.axvline(x=17.46, ls='--', color='k', alpha=0.4)
        ax.axhline(y=17.46, ls='--', color='k', alpha=0.4)

        ax.set(xlabel=r'Targeted $[4.5\mu\rm m]\/(Vega)$',
               ylabel=r'SSDF $[4.5\mu\rm m]\/(Vega)$')
    ax.set(xlim=[20.5, 9.5], ylim=[20.5, 9.5])


def color_mag_plot(ax, channel, targeted_cat, ssdf_cat, targeted_agn_cat=None, ssdf_agn_cat=None, color_threshold=None,
                   errors=False):
    # Separate the AGN from galaxies
    if targeted_agn_cat:
        targeted_galaxies = setdiff(targeted_cat, targeted_agn_cat)
    else:
        targeted_galaxies = targeted_cat
    if ssdf_agn_cat:
        ssdf_galaxies = setdiff(ssdf_cat, ssdf_agn_cat)
    else:
        ssdf_galaxies = ssdf_cat

    if errors:
        targeted_gal_color_err = np.sqrt((targeted_galaxies['I1_MAGERR_APER4']) ** 2 +
                                         (targeted_galaxies['I2_MAGERR_APER4']) ** 2)
        ssdf_gal_color_err = np.sqrt((ssdf_galaxies['I1_MAGERR_APER4']) ** 2 +
                                     (ssdf_galaxies['I2_MAGERR_APER4']) ** 2)
        if targeted_agn_cat:
            targeted_agn_color_err = np.sqrt((targeted_agn_cat['I1_MAGERR_APER4']) ** 2 +
                                             (targeted_agn_cat['I2_MAGERR_APER4']) ** 2)
        if ssdf_agn_cat:
            ssdf_agn_color_err = np.sqrt((ssdf_agn_cat['I1_MAGERR_APER4']) ** 2 +
                                         (ssdf_agn_cat['I2_MAGERR_APER4']) ** 2)
    else:
        targeted_gal_color_err = None
        ssdf_gal_color_err = None
        targeted_agn_color_err = None
        ssdf_agn_color_err = None

    # Plot the galaxies, excluding the AGN if provided
    ax.errorbar(targeted_galaxies[f'I{channel}_MAG_APER4'],
                targeted_galaxies['I1_MAG_APER4'] - targeted_galaxies['I2_MAG_APER4'],
                yerr=targeted_gal_color_err,
                ls='', marker='s', color='C0',
                label='Targeted')
    ax.errorbar(ssdf_galaxies[f'I{channel}_MAG_APER4'],
                ssdf_galaxies['I1_MAG_APER4'] - ssdf_galaxies['I2_MAG_APER4'],
                yerr=ssdf_gal_color_err,
                ls='', marker='o', color='C1', label='SSDF')

    # Plot the AGN if provided
    if targeted_agn_cat:
        ax.errorbar(targeted_agn_cat[f'I{channel}_MAG_APER4'],
                    targeted_agn_cat['I1_MAG_APER4'] - targeted_agn_cat['I2_MAG_APER4'],
                    yerr=targeted_agn_color_err,
                    ls='', marker='s', markerfacecolor='none',
                    markeredgecolor='C0', ecolor='C0', label='Targeted AGN')
    if ssdf_agn_cat:
        ax.errorbar(ssdf_agn_cat[f'I{channel}_MAG_APER4'],
                    ssdf_agn_cat['I1_MAG_APER4'] - ssdf_agn_cat['I2_MAG_APER4'],
                    yerr=ssdf_agn_color_err,
                    ls='', marker='o', markerfacecolor='none',
                    markeredgecolor='C1', ecolor='C1', label='SSDF AGN')

    # Add the color threshold line
    if color_threshold is not None:
        ax.axhline(y=color_threshold, ls='--', c='k', alpha=0.6)

    ax.legend()
    ax.set(ylabel='[3.6] - [4.5] (Vega)')
    if channel == 1:
        ax.set(xlabel='[3.6] (Vega)')
    else:
        ax.set(xlabel='[4.5] (Vega)')


def color_mag_snr_plot(ax, channel, targeted_cat, ssdf_cat, color_threshold):
    # Compute the SNRs for all objects
    targeted_snr = targeted_cat[f'I{channel}_FLUX_APER4'] / targeted_cat[f'I{channel}_FLUXERR_APER4']
    ssdf_snr = ssdf_cat[f'I{channel}_FLUX_APER4'] / ssdf_cat[f'I{channel}_FLUXERR_APER4']

    # Plot the scatter points
    targeted_cm = ax.scatter(targeted_cat[f'I{channel}_MAG_APER4'],
                             targeted_cat['I1_MAG_APER4'] - targeted_cat['I2_MAG_APER4'],
                             c=targeted_snr, cmap='Reds_r', vmin=5, vmax=30, marker='s', label='Targeted')
    ssdf_cm = ax.scatter(ssdf_cat[f'I{channel}_MAG_APER4'],
                         ssdf_cat['I1_MAG_APER4'] - ssdf_cat['I2_MAG_APER4'],
                         c=ssdf_snr, cmap='Blues_r', vmin=5, vmax=30, marker='o', label='SSDF')

    # Plot a line indicating the AGN selection threshold
    ax.axhline(y=color_threshold, ls='--', c='k', alpha=0.6)

    ax.legend()
    ax.set(ylabel='[3.6] - [4.5] (Vega)')

    # Add the color bars
    targeted_cbar = plt.colorbar(targeted_cm, ax=[ax], location='bottom', extend='both')
    targeted_cbar.set_label(f'Targeted I{channel} SNR')
    ssdf_cbar = plt.colorbar(ssdf_cm, ax=[ax], location='left' if channel == 1 else 'right', extend='both')
    ssdf_cbar.set_label(f'SSDF I{channel} SNR')


def flux_err_plot(ax, channel, targeted_cat, ssdf_cat):
    ax.scatter(targeted_cat[f'I{channel}_FLUX_APER4'], targeted_cat[f'I{channel}_FLUXERR_APER4'], marker='s',
               label='Targeted')
    ax.scatter(ssdf_cat[f'I{channel}_FLUX_APER4'], ssdf_cat[f'I{channel}_FLUXERR_APER4'], marker='o', label='SSDF')

    ax.legend()

    if channel == 1:
        ax.set(xlabel=r'$S_{3.6\mu\rm m} [\mu\rm Jy]$', ylabel=r'$\delta S_{3.6\mu\rm m} [\mu\rm Jy]$')
    else:
        ax.set(xlabel=r'$S_{4.5\mu\rm m} [\mu\rm Jy]$', ylabel=r'$\delta S_{4.5\mu\rm m} [\mu\rm Jy]$')


def mag_err_plot(ax, channel, targeted_cat, ssdf_cat):
    ax.scatter(targeted_cat[f'I{channel}_MAG_APER4'], targeted_cat[f'I{channel}_MAGERR_APER4'], marker='s',
               label='Targeted')
    ax.scatter(ssdf_cat[f'I{channel}_MAG_APER4'], ssdf_cat[f'I{channel}_MAGERR_APER4'], marker='o', label='SSDF')
    ax.legend()

    if channel == 1:
        ax.set(xlabel='[3.6] (Vega)', ylabel=r'$\delta[3.6]$ (Vega)')
    else:
        ax.set(xlabel='[4.5] (Vega)', ylabel=r'$\delta[4.5]$ (Vega)')


def color_color_err_plot(ax, targeted_cat, ssdf_cat, targeted_agn_cat=None, ssdf_agn_cat=None,
                         color_threshold=None):
    # Separate the AGN from galaxies
    if targeted_agn_cat:
        targeted_galaxies = setdiff(targeted_cat, targeted_agn_cat)
    else:
        targeted_galaxies = targeted_cat
    if ssdf_agn_cat:
        ssdf_galaxies = setdiff(ssdf_cat, ssdf_agn_cat)
    else:
        ssdf_galaxies = ssdf_cat

    # Compute the color errors
    targeted_gal_color_err = np.sqrt((targeted_galaxies['I1_MAGERR_APER4']) ** 2 +
                                     (targeted_galaxies['I2_MAGERR_APER4']) ** 2)
    ssdf_gal_color_err = np.sqrt((ssdf_galaxies['I1_MAGERR_APER4']) ** 2 +
                                 (ssdf_galaxies['I2_MAGERR_APER4']) ** 2)

    ax.scatter(targeted_galaxies['I1_MAG_APER4'] - targeted_galaxies['I2_MAG_APER4'], targeted_gal_color_err,
               marker='s', c='C0', label='Targeted')
    ax.scatter(ssdf_galaxies['I1_MAG_APER4'] - ssdf_galaxies['I2_MAG_APER4'], ssdf_gal_color_err,
               marker='o', c='C1', label='SSDF')
    if targeted_agn_cat:
        targeted_agn_color_err = np.sqrt((targeted_agn_cat['I1_MAGERR_APER4']) ** 2 +
                                         (targeted_agn_cat['I2_MAGERR_APER4']) ** 2)
        ax.scatter(targeted_agn_cat['I1_MAG_APER4'] - targeted_agn_cat['I2_MAG_APER4'], targeted_agn_color_err,
                   marker='s', edgecolor='C0', facecolor='none', label='Targeted AGN')
    if ssdf_agn_cat:
        ssdf_agn_color_err = np.sqrt((ssdf_agn_cat['I1_MAGERR_APER4']) ** 2 +
                                     (ssdf_agn_cat['I2_MAGERR_APER4']) ** 2)
        ax.scatter(ssdf_agn_cat['I1_MAG_APER4'] - ssdf_agn_cat['I2_MAG_APER4'], ssdf_agn_color_err,
                   marker='o', edgecolor='C1', facecolor='none', label='SSDF AGN')

    if color_threshold is not None:
        ax.axvline(x=color_threshold, ls='--', c='k', alpha=0.6)

    ax.legend()
    ax.set(xlabel='[3.6] - [4.5] (Vega)', ylabel=r'$\delta$([3.6] - [4.5]) (Vega)')


def mag_diff_plot(ax, channel, targeted_cat, ssdf_cat):
    # Merge the catalogs
    merged_catalog = merge_catalogs(channel, targeted_cat, ssdf_cat)

    # To handle non-detections/non-matches, we will set all "-99.0" entries to a magnitude outside our cuts.
    targeted_flux_limit = 20.
    ssdf_flux_limit = 20.
    merged_catalog = sanitize_fluxes_merged(channel=channel, merged_catalog=merged_catalog,
                                            targeted_flux_limit=targeted_flux_limit, ssdf_flux_limit=ssdf_flux_limit,
                                            err=False)

    # # Since we're primarily working in flux space, we'll do a log flux ratio for the mag difference
    # log_flux_ratio = np.log10(merged_catalog[f'TARGETED_I{channel}_FLUX'] / merged_catalog[f'SSDF_I{channel}_FLUX'])

    # Compute the mag difference
    mag_diff = merged_catalog[f'TARGETED_I{channel}_MAG'] - merged_catalog[f'SSDF_I{channel}_MAG']

    # We want to find the mean of the matched data only
    mean_log_flux_ratio = np.mean(mag_diff[np.abs(mag_diff) < 1.])

    # Make a plot of log-flux ratio vs log-flux
    ax.scatter(merged_catalog[f'TARGETED_I{channel}_MAG'], mag_diff,
               label=rf'mean = {mean_log_flux_ratio:.2e}  Vega Magnitude')
    ax.axhline(y=0., ls='--', c='k', alpha=0.6)
    ax.legend(loc='upper left', frameon=False)

    # Add Vega axes (Magnitudes on x-axis, color on y-axis)
    # add_vega_axes(ax=ax, channel=channel, xaxis=True, mag_diff=False)
    # add_vega_axes(ax=ax, channel=channel, xaxis=False, mag_diff=True)

    # Set the axes labels
    if channel == 1:
        ax.set(xlabel=r'Targeted $[3.6\mu\rm m]\/(Vega)$',
               ylabel=r'$[3.6\mu\rm m]_{{\rm Targeted}} - [3.6\mu\rm m]_{{\rm SSDF}}$')
    else:
        ax.set(xlabel=r'Targeted $[4.5\mu\rm m]\/(Vega)$',
               ylabel=r'$[4.5\mu\rm m]_{{\rm Targeted}} - [4.5\mu\rm m]_{{\rm SSDF}}$')


# Read in the catalog of all clusters common to both the targeted and SSDF programs
common_cluster_cat = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Misc/common_clusters_2500d_100d.fits')

# Remove clusters that don't have IRAC imaging in both surveys (1 = targeted, 2 = SSDF, 3 = both)
common_cluster_cat = common_cluster_cat[common_cluster_cat['IRAC_imaging'] == 3]

# Read in the sky error catalogs
targeted_sky_errors = Table.read(
    'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPT-SZ_2500d/SPT-SZ_sky_errors_qphot.fits')
ssdf_sky_errors = Table.read(
    'Data_Repository/Project_Data/SPT-IRAGN/sky_errors/SPTpol_100d/SPTpol_sky_errors_qphot.fits')

# Read in the Observed-Official ID lookup tables for the targeted sample.
with open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_observed_to_official_ids.json', 'r') as f, \
        open('Data_Repository/Project_Data/SPT-IRAGN/Misc/SPT-SZ_official_to_observed_ids.json', 'r') as g:
    obs_to_off_ids = json.load(f)
    off_to_obs_ids = json.load(g)

# Read in the purity and surface density files
with (open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.46.json', 'r') as f,
      open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_background_prior_distributions.json',
           'r') as g):
    sdwfs_purity_data = json.load(f)
    sdwfs_prior_data = json.load(g)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
threshold_bins = sdwfs_prior_data['color_thresholds'][:-1]

# Set up interpolators
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Get list of photometric catalogs
targeted_cat_names = glob.glob('Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPT-SZ_2500d/*.cat')
ssdf_cat_names = glob.glob('Data_Repository/Catalogs/SPT/Spitzer_catalogs/SPTpol_100d/*.cat')

# Also read in the targeted IRAC image
image_names = glob.glob('Data_Repository/Images/SPT/Spitzer_IRAC/SPT-SZ_2500d/I2*_mosaic.cutout.fits')

spt_id = re.compile(r'SPT-CLJ\d+-\d+')
targeted_catalogs, ssdf_catalogs = [], []
for cluster in common_cluster_cat:
    cluster_id = cluster['SPT_ID']
    cluster_z = cluster['REDSHIFT_Bocquet']

    # Pull the sky error values
    targeted_I1_flux_err = targeted_sky_errors['I1_flux_error'][
        targeted_sky_errors['SPT_ID'] == off_to_obs_ids[cluster_id]]
    targeted_I2_flux_err = targeted_sky_errors['I2_flux_error'][
        targeted_sky_errors['SPT_ID'] == off_to_obs_ids[cluster_id]]
    ssdf_I1_flux_err = ssdf_sky_errors['I1_flux_error'][ssdf_sky_errors['SPT_ID'] == cluster_id]
    ssdf_I2_flux_err = ssdf_sky_errors['I2_flux_error'][ssdf_sky_errors['SPT_ID'] == cluster_id]

    # Extract the catalog names
    targeted_catalog_name = ''.join(s for s in targeted_cat_names if off_to_obs_ids[cluster_id] in s)
    ssdf_catalog_name = ''.join(s for s in ssdf_cat_names if cluster_id in s)

    # Read in the catalogs
    targeted_catalog = Table.read(targeted_catalog_name, format='ascii')
    ssdf_catalog = Table.read(ssdf_catalog_name, format='ascii')

    # Fix a problem that arises later with column-names
    if 'TILE' in targeted_catalog.colnames:
        targeted_catalog.rename_column('TILE', 'TARGET')

    # To create a common footprint, we will select all objects within 1.5 arcmin of the Huang+20 (SPTpol) center.
    center = common_cluster_cat['RA_Huang', 'DEC_Huang'][common_cluster_cat['SPT_ID'] == cluster_id]
    center_coord = SkyCoord(center['RA_Huang'], center['DEC_Huang'], unit=u.deg)
    targeted_coords = SkyCoord(targeted_catalog['ALPHA_J2000'], targeted_catalog['DELTA_J2000'], unit=u.deg)
    ssdf_coords = SkyCoord(ssdf_catalog['ALPHA_J2000'], ssdf_catalog['DELTA_J2000'], unit=u.deg)

    targeted_catalog = targeted_catalog[targeted_coords.separation(center_coord) <= 1.5 * u.arcmin]
    ssdf_catalog = ssdf_catalog[ssdf_coords.separation(center_coord) <= 1.5 * u.arcmin]

    # Add columns to store the sky flux errors
    targeted_catalog['I1_flux_error'] = targeted_I1_flux_err
    targeted_catalog['I2_flux_error'] = targeted_I2_flux_err
    ssdf_catalog['I1_flux_error'] = ssdf_I1_flux_err
    ssdf_catalog['I2_flux_error'] = ssdf_I2_flux_err

    # Make magnitude cuts
    targeted_catalog = targeted_catalog[(10.0 <= targeted_catalog['I1_MAG_APER4']) &
                                        (targeted_catalog['I1_MAG_APER4'] <= 18.3) &
                                        (10.45 <= targeted_catalog['I2_MAG_APER4']) &
                                        (targeted_catalog['I2_MAG_APER4'] <= 17.46)]
    ssdf_catalog = ssdf_catalog[(10.0 <= ssdf_catalog['I1_MAG_APER4']) &
                                (ssdf_catalog['I1_MAG_APER4'] <= 18.3) &
                                (10.45 <= ssdf_catalog['I2_MAG_APER4']) &
                                (ssdf_catalog['I2_MAG_APER4'] <= 17.46)]
    targeted_catalogs.append(targeted_catalog)
    ssdf_catalogs.append(ssdf_catalog)

    # Make AGN color cut
    selection_color = agn_purity_color(cluster_z)
    targeted_agn = targeted_catalog[
        targeted_catalog['I1_MAG_APER4'] - targeted_catalog['I2_MAG_APER4'] >= selection_color]
    ssdf_agn = ssdf_catalog[ssdf_catalog['I1_MAG_APER4'] - ssdf_catalog['I2_MAG_APER4'] >= selection_color]

    # Make the source count plots
    fig, (ax_I1, ax_I2) = plt.subplots(ncols=2, figsize=(16, 8))
    source_count_plot(ax=ax_I1, channel=1, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog,
                      targeted_agn_cat=targeted_agn, ssdf_agn_cat=ssdf_agn)
    source_count_plot(ax=ax_I2, channel=2, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog,
                      targeted_agn_cat=targeted_agn, ssdf_agn_cat=ssdf_agn)
    fig.suptitle(f'{cluster_id}')
    plt.tight_layout()
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs'
                f'/source_counts/{cluster_id}_source_count.pdf')

    # Plot the objects on the targeted I2 image
    fig = plt.figure(figsize=(8, 8))
    plot_image(fig, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog,
               targeted_agn_cat=targeted_agn, ssdf_agn_cat=ssdf_agn, )
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
                f'images/{cluster_id}_image.pdf')

    # Plot the flux photometry comparison
    fig, axarr = plt.subplots(ncols=2, figsize=(17, 8), constrained_layout=True)
    ax_I1, ax_I2 = axarr
    flux_comparison_plot(ax=ax_I1, channel=1, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    cm = flux_comparison_plot(ax=ax_I2, channel=2, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    cbar = plt.colorbar(cm, ax=axarr, extend='both')
    cbar.set_label('[3.6] - [4.5]')
    fig.suptitle(f'{cluster_id}')
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
                f'phot_comparison/{cluster_id}_phot_comparison.pdf')

    # Make the mag difference plot
    fig, (ax_I1, ax_I2) = plt.subplots(ncols=2, sharey='row', figsize=(17, 8), constrained_layout=True)
    mag_diff_plot(ax=ax_I1, channel=1, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    mag_diff_plot(ax=ax_I2, channel=2, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    fig.suptitle(f'{cluster_id}')
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
                f'mag_diff/{cluster_id}_mag_diff.pdf')

    # Make the flux-flux err plot
    fig, (ax_I1, ax_I2) = plt.subplots(ncols=2, figsize=(16, 8))
    flux_err_plot(ax=ax_I1, channel=1, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    flux_err_plot(ax=ax_I2, channel=2, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    fig.suptitle(f'{cluster_id}')
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
                f'flux_error/{cluster_id}_flux_error.pdf')

    # Make the mag-mag err plot
    fig, (ax_I1, ax_I2) = plt.subplots(ncols=2, figsize=(16, 8))
    mag_err_plot(ax=ax_I1, channel=1, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    mag_err_plot(ax=ax_I2, channel=2, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    fig.suptitle(f'{cluster_id}')
    fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
                f'mag_error/{cluster_id}_mag_error.pdf')

    # Make the color-mag plot
    fig, (ax_I1, ax_I2) = plt.subplots(ncols=2, figsize=(18, 9), constrained_layout=True)
    color_mag_plot(ax=ax_I1, channel=1, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog,
                   targeted_agn_cat=targeted_agn, ssdf_agn_cat=ssdf_agn, color_threshold=selection_color, errors=True)
    color_mag_plot(ax=ax_I2, channel=2, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog,
                   targeted_agn_cat=targeted_agn, ssdf_agn_cat=ssdf_agn, color_threshold=selection_color, errors=True)
    fig.suptitle(f'{cluster_id}')
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
                f'color_mag/{cluster_id}_color_mag.pdf')

    # # Make the color-mag-snr plot
    # fig, (ax_I1, ax_I2) = plt.subplots(ncols=2, figsize=(18, 9), constrained_layout=True)
    # color_mag_snr_plot(ax=ax_I1, channel=1, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    # color_mag_snr_plot(ax=ax_I2, channel=2, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog)
    # fig.suptitle(f'{cluster_id}')
    # fig.savefig('Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/Source_Counts/'
    #             f'{cat_type}/color_mag_snr/{cluster_id}_{cat_type}_color_mag_snr.pdf')

    # Make color-color error plot
    fig, ax = plt.subplots(figsize=(9, 9))
    color_color_err_plot(ax=ax, targeted_cat=targeted_catalog, ssdf_cat=ssdf_catalog, targeted_agn_cat=targeted_agn,
                         ssdf_agn_cat=ssdf_agn, color_threshold=selection_color)
    ax.set(title=f'{cluster_id}')
    fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
                f'color_color_err/{cluster_id}_color_color_err.pdf')

    plt.close('all')

# For the remaining plots we will stack the catalogs together
targeted_stacked = vstack(targeted_catalogs, metadata_conflicts='silent')
ssdf_stacked = vstack(ssdf_catalogs, metadata_conflicts='silent')

# targeted_stacked_agn = targeted_stacked[targeted_stacked['I1_MAG_APER4'] - targeted_stacked['I2_MAG_APER4'] >= 0.7]
# ssdf_stacked_agn = ssdf_stacked[ssdf_stacked['I1_MAG_APER4'] - ssdf_stacked['I2_MAG_APER4'] >= 0.7]

# Create the stacked, differential source count plot for I2
fig, ax = plt.subplots()
source_count_plot(ax=ax, channel=2, targeted_cat=targeted_stacked, ssdf_cat=ssdf_stacked,
                  targeted_agn_cat=None, ssdf_agn_cat=None,
                  cumulative=False)
ax.set(title='All Common Clusters', xlabel=r'$\log\/S_{{4.5\mu\rm m}}\/[\mu\rm Jy]$', ylabel='Number')
plt.tight_layout()
fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
            f'common_clusters_stacked_noncum_hist.pdf')

# Create the stacked flux comparison plot
fig, (ax_I1, ax_I2) = plt.subplots(ncols=2, figsize=(16, 8))
stacked_flux_comparison_plot(ax=ax_I1, channel=1, targeted_cat=targeted_stacked, ssdf_cat=ssdf_stacked)
stacked_flux_comparison_plot(ax=ax_I2, channel=2, targeted_cat=targeted_stacked, ssdf_cat=ssdf_stacked)
fig.suptitle('All Common Clusters')
plt.tight_layout()
fig.savefig(f'Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/SSDFv10_catalogs/'
            f'common_clusters_stacked_phot_comparison.pdf')

plt.close('all')
