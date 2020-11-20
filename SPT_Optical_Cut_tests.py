"""
SPT_Optical_Cut_tests.py
Author: Benjamin Floyd

Explores any optical selections that could be applied to help refine the AGN selection using the PISCO catalogs.
"""

import glob
import json

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack, vstack

# Grab the PISCO catalogs

pisco_cat_names = glob.glob('Data/Data_Repository/Catalogs/SPT/PISCO_catalogs/*.sav')

# Get the two version of the IRAC catalogs
irac_sz_cat_names = glob.glob('Data/Catalogs/*.cat')
irac_pol_cat_names = glob.glob('Data/SPTPol/catalogs/cluster_cutouts/*.cat')

# Because the SPT-SZ images use the older IDs, we will need to use a conversion dictionary from "official" to "observed"
with open('Data/SPT-SZ_official_to_observed_ids.json', 'r') as f:
    sptsz_official_to_obs_id = json.load(f)

# A list of common clusters between the IRAC samples that also have PISCO observations
common_ids_with_pisco = ['SPT-CLJ0001-5440', 'SPT-CLJ2300-5331', 'SPT-CLJ2301-5546',
                         'SPT-CLJ2311-5820', 'SPT-CLJ2337-5912', 'SPT-CLJ2358-5229']
# %%
catalog_dict = {}
for cluster_id in common_ids_with_pisco:
    # Get the catalog names
    pisco_cat_name = ''.join(s for s in pisco_cat_names if cluster_id in s)
    irac_sz_cat_name = ''.join(s for s in irac_sz_cat_names if sptsz_official_to_obs_id[cluster_id] in s)
    irac_pol_cat_name = ''.join(s for s in irac_pol_cat_names if cluster_id in s)

    # Read in the catalogs
    pisco_catalog = Table(sio.readsav(pisco_cat_name)['cat'])
    irac_sz_catalog = Table.read(irac_sz_cat_name, format='ascii')
    irac_pol_catalog = Table.read(irac_pol_cat_name, format='ascii')

    # Fix the SPT ID column for the irac catalogs
    if 'TILE' in irac_sz_catalog.colnames:
        irac_sz_catalog.rename_column('TILE', 'SPT_ID')
    else:
        irac_sz_catalog.rename_column('TARGET', 'SPT_ID')
    irac_pol_catalog['SPT_ID'] = cluster_id

    # Apply SNR cuts to the PISCO catalog F/dF = -2.5 / (ln(10) * dm)
    SNR_cut = 5
    dm_snr_prefactor = 2.5 / np.log(10)
    pisco_catalog = pisco_catalog[(dm_snr_prefactor / np.abs(pisco_catalog['RERR']) >= SNR_cut) &
                                  (dm_snr_prefactor / np.abs(pisco_catalog['RERR']) >= SNR_cut) &
                                  (dm_snr_prefactor / np.abs(pisco_catalog['IERR']) >= SNR_cut) &
                                  (dm_snr_prefactor / np.abs(pisco_catalog['ZERR']) >= SNR_cut)]

    # Remove non-detections from the PISCO catalog
    pisco_catalog = pisco_catalog[(-20.0 < pisco_catalog['G']) & (pisco_catalog['G'] < 99.0) &
                                  (-20.0 < pisco_catalog['R']) & (pisco_catalog['R'] < 99.0) &
                                  (-20.0 < pisco_catalog['I']) & (pisco_catalog['I'] < 99.0) &
                                  (-20.0 < pisco_catalog['Z']) & (pisco_catalog['Z'] < 99.0)]

    # Add columns for the IRAC band in AB for later. We will still make selections in Vega mags
    irac_sz_catalog['I1_MAG_APER4_AB'] = irac_sz_catalog['I1_MAG_APER4'] + 2.79
    irac_sz_catalog['I2_MAG_APER4_AB'] = irac_sz_catalog['I2_MAG_APER4'] + 3.26
    irac_pol_catalog['I1_MAG_APER4_AB'] = irac_pol_catalog['I1_MAG_APER4'] + 2.79
    irac_pol_catalog['I2_MAG_APER4_AB'] = irac_pol_catalog['I2_MAG_APER4'] + 3.26

    # Make the IRAC magnitude cuts to select for AGN
    irac_sz_catalog = irac_sz_catalog[(irac_sz_catalog['I1_MAG_APER4'] >= 10.0) &
                                      (irac_sz_catalog['I2_MAG_APER4'] >= 10.45) &
                                      (irac_sz_catalog['I2_MAG_APER4'] <= 17.46)]
    irac_pol_catalog = irac_pol_catalog[(irac_pol_catalog['I1_MAG_APER4'] >= 10.0) &
                                        (irac_pol_catalog['I2_MAG_APER4'] >= 10.45) &
                                        (irac_pol_catalog['I2_MAG_APER4'] <= 17.46)]

    # Create SkyCoords for all three catalogs
    pisco_coords = SkyCoord(pisco_catalog['OBJ_RA'], pisco_catalog['OBJ_DEC'], unit=u.deg)
    irac_sz_coords = SkyCoord(irac_sz_catalog['ALPHA_J2000'], irac_sz_catalog['DELTA_J2000'], unit=u.deg)
    irac_pol_coords = SkyCoord(irac_pol_catalog['ALPHA_J2000'], irac_pol_catalog['DELTA_J2000'], unit=u.deg)

    # Match the PISCO catalogs to both the IRAC catalogs
    pisco_sz_idx, pisco_sz_sep, _ = irac_sz_coords.match_to_catalog_sky(pisco_coords)
    pisco_pol_idx, pisco_pol_sep, _ = irac_pol_coords.match_to_catalog_sky(pisco_coords)

    # Match the irac to pisco coordinates taking the closest match and removing objects further than 3 arcsec away
    initial_sep_constraint_sz = pisco_sz_sep <= 3 * u.arcsec
    initial_sep_constraint_pol = pisco_pol_sep <= 3 * u.arcsec
    pisco_coords_sz_matches = pisco_coords[pisco_sz_idx[initial_sep_constraint_sz]]
    irac_coords_sz_matches = irac_sz_coords[initial_sep_constraint_sz]
    pisco_coords_pol_matches = pisco_coords[pisco_pol_idx[initial_sep_constraint_pol]]
    irac_coords_pol_matches = irac_pol_coords[initial_sep_constraint_pol]

    # Find the offset coordinates from the IRAC to the PISCO coordinates of the matches above
    dra_ddec_sz = pisco_coords_sz_matches.spherical_offsets_to(irac_coords_sz_matches)
    dra_ddec_pol = pisco_coords_pol_matches.spherical_offsets_to(irac_coords_pol_matches)

    # Find the centroid of the offsets
    offset_centroid_sz = np.median(dra_ddec_sz, axis=1) * u.deg
    offset_centroid_pol = np.median(dra_ddec_pol, axis=1) * u.deg

    # Cast the offset centroids into SkyCoords
    origin = SkyCoord(0, 0, unit=u.deg)
    offset_coord_sz = SkyCoord(offset_centroid_sz[0], offset_centroid_sz[1])
    offset_coord_pol = SkyCoord(offset_centroid_pol[0], offset_centroid_pol[1])

    # Find the position angle and separation of the centroids relative to the origin
    offset_sz_pa, offset_sz_sep = origin.position_angle(offset_coord_sz), origin.separation(offset_coord_sz)
    offset_pol_pa, offset_pol_sep = origin.position_angle(offset_coord_pol), origin.separation(offset_coord_pol)

    # Now that we know the offset corrections apply the correction to the PISCO coordinates
    pisco_coords_corrected_sz = pisco_coords.directional_offset_by(offset_sz_pa, offset_sz_sep)
    pisco_coords_corrected_pol = pisco_coords.directional_offset_by(offset_pol_pa, offset_pol_sep)

    # Rematch the corrected PISCO coordinates to the IRAC coordinates
    pisco_sz_corr_idx, pisco_sz_corr_sep, _ = irac_sz_coords.match_to_catalog_sky(pisco_coords_corrected_sz)
    pisco_pol_corr_idx, pisco_pol_corr_sep, _ = irac_pol_coords.match_to_catalog_sky(pisco_coords_corrected_pol)

    # Analysis of separation histograms show that ~0.5 arcsec is a reasonable cutoff
    max_sep = 0.5 * u.arcsec

    # Select matches for PISCO-SPT-SZ and merge the catalogs
    irac_sz_matches = irac_sz_catalog[pisco_sz_corr_sep <= max_sep]
    pisco_sz_matches = pisco_catalog[pisco_sz_corr_idx[pisco_sz_corr_sep <= max_sep]]
    pisco_irac_sz_catalog = hstack([pisco_sz_matches, irac_sz_matches])

    # Select matches for PISCO-SPTpol and merge the catalogs
    irac_pol_matches = irac_pol_catalog[pisco_pol_corr_sep <= max_sep]
    pisco_pol_matches = pisco_catalog[pisco_pol_corr_idx[pisco_pol_corr_sep <= max_sep]]
    pisco_irac_pol_catalog = hstack([pisco_pol_matches, irac_pol_matches])

    # # Make IRAC color cut to make the final AGN selection
    pisco_irac_sz_agn = pisco_irac_sz_catalog[pisco_irac_sz_catalog['I1_MAG_APER4']
                                              - pisco_irac_sz_catalog['I2_MAG_APER4'] >= 0.7]
    pisco_irac_pol_agn = pisco_irac_pol_catalog[pisco_irac_pol_catalog['I1_MAG_APER4']
                                                - pisco_irac_pol_catalog['I2_MAG_APER4'] >= 0.7]

    # Store the catalogs in the dictionary for later use
    catalog_dict[cluster_id] = {'pisco_irac_sz_catalog': pisco_irac_sz_catalog,
                                'pisco_irac_pol_catalog': pisco_irac_pol_catalog,
                                'pisco_irac_sz_agn': pisco_irac_sz_agn,
                                'pisco_irac_pol_agn': pisco_irac_pol_agn}

    # # <editor-fold desc="Diagnostic Plots">
    # # Offsets with centroid
    # fig, (ax, bx) = plt.subplots(ncols=2, sharey='row', figsize=(4.8 * 2, 4.8))
    # ax.scatter(dra_ddec_sz[0].arcsec, dra_ddec_sz[1].arcsec, marker='.', alpha=0.4)
    # ax.scatter(offset_centroid_sz.to(u.arcsec)[0], offset_centroid_sz.to(u.arcsec)[1], marker='+', color='r')
    # ax.axvline(0.0, ls='--', c='k', alpha=0.2)
    # ax.axhline(0.0, ls='--', c='k', alpha=0.2)
    # ax.add_artist(Circle((0, 0), radius=1, linestyle='--', color='k', alpha=0.2, fill=False))
    # ax.set(title='SPT-SZ/targeted', xlabel=r'$(\alpha_{\rm IRAC}-\alpha_{\rm PISCO})\cos\delta_{\rm IRAC}$ (arcsec)',
    #        ylabel=r'$\delta_{\rm IRAC}-\delta_{\rm PISCO}$ (arcsec)', xlim=[-2, 2], ylim=[-2, 2])
    # ax.set_aspect('equal')
    #
    # bx.scatter(dra_ddec_pol[0].arcsec, dra_ddec_pol[1].arcsec, marker='.', alpha=0.4)
    # bx.scatter(offset_centroid_pol.to(u.arcsec)[0], offset_centroid_pol.to(u.arcsec)[1], marker='+', color='r')
    # bx.axvline(0.0, ls='--', c='k', alpha=0.2)
    # bx.axhline(0.0, ls='--', c='k', alpha=0.2)
    # bx.add_artist(Circle((0, 0), radius=1, linestyle='--', color='k', alpha=0.2, fill=False))
    # bx.set(title='SPTpol/SSDF', xlabel=r'$(\alpha_{\rm IRAC}-\alpha_{\rm PISCO})\cos\delta_{\rm IRAC}$ (arcsec)',
    #        xlim=[-2, 2], ylim=[-2, 2])
    # bx.set_aspect('equal')
    # fig.suptitle(cluster_id)
    # fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/PISCO_photometry/'
    #             f'Offsets/{cluster_id}_2arcsec.pdf')
    #
    # # Corrected offset
    # pisco_coords_corrected_sz_matches = pisco_coords_corrected_sz[pisco_sz_corr_idx[pisco_sz_corr_sep <= 3 * u.arcsec]]
    # irac_coords_corrected_sz_matches = irac_sz_coords[pisco_sz_corr_sep <= 3 * u.arcsec]
    # pisco_coords_corrected_pol_matches = pisco_coords_corrected_pol[pisco_pol_corr_idx[pisco_pol_corr_sep <= 3 * u.arcsec]]
    # irac_coords_corrected_pol_matches = irac_pol_coords[pisco_pol_corr_sep <= 3 * u.arcsec]
    # dra_ddec_sz_corr = pisco_coords_corrected_sz_matches.spherical_offsets_to(irac_coords_corrected_sz_matches)
    # dra_ddec_pol_corr = pisco_coords_corrected_pol_matches.spherical_offsets_to(irac_coords_corrected_pol_matches)
    #
    # fig, (ax, bx) = plt.subplots(ncols=2, sharey='row', figsize=(4.8 * 2, 4.8))
    # ax.scatter(dra_ddec_sz_corr[0].arcsec, dra_ddec_sz_corr[1].arcsec, marker='.', alpha=0.4)
    # ax.scatter(np.median(dra_ddec_sz_corr[0].arcsec), np.median(dra_ddec_sz_corr[1].arcsec),
    #            marker='+', color='r')
    # ax.axvline(0.0, ls='--', c='k', alpha=0.2)
    # ax.axhline(0.0, ls='--', c='k', alpha=0.2)
    # ax.add_artist(Circle((0, 0), radius=1, linestyle='--', color='k', alpha=0.2, fill=False))
    # ax.set(title='SPT-SZ/targeted',
    #        xlabel=r'$(\alpha_{\rm IRAC}-\alpha_{\rm PISCO})\cos\delta_{\rm IRAC}$ Corrected (arcsec)',
    #        ylabel=r'$\delta_{\rm IRAC}-\delta_{\rm PISCO}$ Corrected (arcsec)', xlim=[-2, 2], ylim=[-2, 2])
    # ax.set_aspect('equal')
    #
    # bx.scatter(dra_ddec_pol_corr[0].arcsec, dra_ddec_pol_corr[1].arcsec, marker='.', alpha=0.4)
    # bx.scatter(np.median(dra_ddec_pol_corr[0].arcsec), np.median(dra_ddec_pol_corr[1].arcsec),
    #            marker='+', color='r')
    # bx.axvline(0.0, ls='--', c='k', alpha=0.2)
    # bx.axhline(0.0, ls='--', c='k', alpha=0.2)
    # bx.add_artist(Circle((0, 0), radius=1, linestyle='--', color='k', alpha=0.2, fill=False))
    # bx.set(title='SPTpol/SSDF',
    #        xlabel=r'$(\alpha_{\rm IRAC}-\alpha_{\rm PISCO})\cos\delta_{\rm IRAC}$ Corrected (arcsec)',
    #        xlim=[-2, 2], ylim=[-2, 2])
    # bx.set_aspect('equal')
    # fig.suptitle(cluster_id)
    # fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/PISCO_photometry/'
    #             f'Offsets/{cluster_id}_2arcsec_corrected.pdf')
    #
    # # Angular Separation (Corrected)
    # fig, ax = plt.subplots()
    # sep_bins = np.arange(0., 2.25, 0.25)
    # ax.hist(pisco_sz_corr_sep.arcsec, bins=sep_bins, label='SPT-SZ/targeted')
    # ax.hist(pisco_pol_corr_sep.arcsec, bins=sep_bins, label='SPTpol/SSDF', alpha=0.5)
    # ax.legend()
    # ax.set(title=f'{cluster_id}', xlabel='Corrected Separation (arcsec)')
    # fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/PISCO_photometry/'
    #             f'Offsets/{cluster_id}_corrected_angular_sep.pdf')
    # # </editor-fold>

#%% Stack the catalogs from the clusters together
pisco_irac_sz_catalog_stacked = vstack([cat['pisco_irac_sz_catalog'] for cat in catalog_dict.values()])
pisco_irac_pol_catalog_stacked = vstack([cat['pisco_irac_pol_catalog'] for cat in catalog_dict.values()])
pisco_irac_sz_agn_stacked = vstack([cat['pisco_irac_sz_agn'] for cat in catalog_dict.values()])
pisco_irac_pol_agn_stacked = vstack([cat['pisco_irac_pol_agn'] for cat in catalog_dict.values()])

# Grab all the columns (just to make thing easier to plot)
g_sz, g_pol = pisco_irac_sz_catalog_stacked['G'], pisco_irac_pol_catalog_stacked['G']
r_sz, r_pol = pisco_irac_sz_catalog_stacked['R'], pisco_irac_pol_catalog_stacked['R']
i_sz, i_pol = pisco_irac_sz_catalog_stacked['I'], pisco_irac_pol_catalog_stacked['I']
z_sz, z_pol = pisco_irac_sz_catalog_stacked['Z'], pisco_irac_pol_catalog_stacked['Z']
I1_sz, I1_pol = pisco_irac_sz_catalog_stacked['I1_MAG_APER4_AB'], pisco_irac_pol_catalog_stacked['I1_MAG_APER4_AB']
I2_sz, I2_pol = pisco_irac_sz_catalog_stacked['I2_MAG_APER4_AB'], pisco_irac_pol_catalog_stacked['I2_MAG_APER4_AB']
pisco_sz_bands = [g_sz, r_sz, i_sz, z_sz]
pisco_pol_bands = [g_pol, r_pol, i_pol, z_pol]

# And for the AGN
g_sz_agn, g_pol_agn = pisco_irac_sz_agn_stacked['G'], pisco_irac_pol_agn_stacked['G']
r_sz_agn, r_pol_agn = pisco_irac_sz_agn_stacked['R'], pisco_irac_pol_agn_stacked['R']
i_sz_agn, i_pol_agn = pisco_irac_sz_agn_stacked['I'], pisco_irac_pol_agn_stacked['I']
z_sz_agn, z_pol_agn = pisco_irac_sz_agn_stacked['Z'], pisco_irac_pol_agn_stacked['Z']
I1_sz_agn, I1_pol_agn = pisco_irac_sz_agn_stacked['I1_MAG_APER4_AB'], pisco_irac_pol_agn_stacked['I1_MAG_APER4_AB']
I2_sz_agn, I2_pol_agn = pisco_irac_sz_agn_stacked['I2_MAG_APER4_AB'], pisco_irac_pol_agn_stacked['I2_MAG_APER4_AB']

# Point-like sources
pisco_irac_sz_point_sources = pisco_irac_sz_catalog_stacked[pisco_irac_sz_catalog_stacked['Z_STARGAL'] > 0.95]
pisco_irac_pol_point_sources = pisco_irac_pol_catalog_stacked[pisco_irac_pol_catalog_stacked['Z_STARGAL'] > 0.95]
g_sz_stars, g_pol_stars = pisco_irac_sz_point_sources['G'], pisco_irac_pol_point_sources['G']
r_sz_stars, r_pol_stars = pisco_irac_sz_point_sources['R'], pisco_irac_pol_point_sources['R']
i_sz_stars, i_pol_stars = pisco_irac_sz_point_sources['I'], pisco_irac_pol_point_sources['I']
z_sz_stars, z_pol_stars = pisco_irac_sz_point_sources['Z'], pisco_irac_pol_point_sources['Z']
I1_sz_stars, I1_pol_stars = pisco_irac_sz_point_sources['I1_MAG_APER4_AB'], pisco_irac_pol_point_sources['I1_MAG_APER4_AB']
I2_sz_stars, I2_pol_stars = pisco_irac_sz_point_sources['I2_MAG_APER4_AB'], pisco_irac_pol_point_sources['I2_MAG_APER4_AB']

# %% PISCO mag -- [3.6]-[4.5]
for sz_band, pol_band, band_name in zip(pisco_sz_bands, pisco_pol_bands, ["g'", "r'", "i'", "z'"]):
    fig, ax = plt.subplots()
    ax.scatter(sz_band, I1_sz - I2_sz, marker='.', label='SPT-SZ/targeted')
    ax.scatter(pol_band, I1_pol - I2_pol, marker='.', label='SPTpol/SDSS', alpha=0.4)
    # ax.scatter(g_sz_agn, I1_sz_agn - I2_sz_agn, marker='.', label='IRAC AGN')
    ax.axhline(0.7 + (2.79 - 3.26), ls='--', c='k')
    ax.legend()
    ax.set(title='SPT clusters (stacked by survey)', xlabel=f"PISCO {band_name} (AB)", ylabel='[3.6] - [4.5] (AB)')
    fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/'
                f'PISCO_photometry/Color-Mag/SPT_stacked_PISCO_{band_name}-IRAC_color.pdf')

    # PISCO mag-[3.6] -- [3.6]-[4.5]
    fig, ax = plt.subplots()
    ax.scatter(sz_band - I1_sz, I1_sz - I2_sz, marker='.', label='SPT-SZ/targeted')
    ax.scatter(pol_band - I1_pol, I1_pol - I2_pol, marker='.', label='SPTpol/SDSS', alpha=0.4)
    # ax.scatter(g_sz_agn - I1_sz_agn, I1_sz_agn - I2_sz_agn, marker='.', label='IRAC AGN')
    ax.axhline(0.7 + (2.79 - 3.26), ls='--', c='k')
    ax.legend()
    ax.set(title='SPT clusters (stacked by survey)', xlabel=f"PISCO {band_name} - [3.6] (AB)",
           ylabel='[3.6] - [4.5] (AB)')
    fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/'
                f'PISCO_photometry/Color-Color/SPT_stacked_PISCO_{band_name}-I1-IRAC_color.pdf')

    # PISCO mag-[4.5] -- [3.6]-[4.5]
    fig, ax = plt.subplots()
    ax.scatter(sz_band - I2_sz, I1_sz - I2_sz, marker='.', label='SPT-SZ/targeted')
    ax.scatter(pol_band - I2_pol, I1_pol - I2_pol, marker='.', label='SPTpol/SDSS', alpha=0.4)
    # ax.scatter(g_sz_agn - I2_sz_agn, I1_sz_agn - I2_sz_agn, marker='.', label='IRAC AGN')
    ax.axhline(0.7 + (2.79 - 3.26), ls='--', c='k')
    ax.legend()
    ax.set(title='SPT clusters (stacked by survey)', xlabel=f"PISCO {band_name} - [4.5] (AB)",
           ylabel='[3.6] - [4.5] (AB)')
    fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/'
                f'PISCO_photometry/Color-Color/SPT_stacked_PISCO_{band_name}-I2-IRAC_color.pdf')
    plt.close('all')

#%% PISCO z-band CLASS_STAR -- z
fig, ax = plt.subplots()
ax.scatter(pisco_irac_sz_catalog_stacked['Z'], pisco_irac_sz_catalog_stacked['Z_STARGAL'], marker='.', label='SZ/targetd')
ax.set(xlabel="PISCO z' (AB)", ylabel='CLASS_STAR (z-band)')
fig.savefig("Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/PISCO_photometry/"
            "SPT_stacked_PISCO_z'_CLASS_STAR.pdf")
# plt.show()

#%%  g - i -- i - [3.6]
fig, ax = plt.subplots()
ax.scatter(i_sz - I1_sz, g_sz - i_sz, marker='.', color='k', alpha=0.4)
ax.scatter(i_pol - I1_pol, g_pol - i_pol, marker='.', color='k', alpha=0.4)
ax.scatter(i_sz_stars - I1_sz_stars, g_sz_stars - i_sz_stars, marker='.', color='C0', label='Point Sources')
ax.scatter(i_pol_stars - I1_pol_stars, g_pol_stars - i_pol_stars, marker='.', color='C0')
ax.scatter(i_sz_agn - I1_sz_agn, g_sz_agn - i_sz_agn, marker='.', color='C1', label='SPT-SZ/targeted')
ax.scatter(i_pol_agn - I1_pol_agn, g_pol_agn - i_pol_agn, marker='o', edgecolor='C1', facecolor='none', label='SPTpol/SSDF')
ax.legend()
ax.set(xlabel="PISCO i' - [3.6] (AB)", ylabel="PISCO g' - i' (AB)")
fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/PISCO_photometry/Color-Color/'
            "SPT_stacked_PISCO_g'-i'-I1_color_PISCO_z'_CLASS_STAR.pdf")
# plt.show()

#%%  r - z -- z - [3.6]
fig, ax = plt.subplots()
ax.scatter(z_sz - I1_sz, r_sz - z_sz, marker='.', color='k', alpha=0.4)
ax.scatter(z_pol - I1_pol, r_pol - z_pol, marker='.', color='k', alpha=0.4)
ax.scatter(z_sz_stars - I1_sz_stars, r_sz_stars - z_sz_stars, marker='.', color='C0', label='Point Sources')
ax.scatter(z_pol_stars - I1_pol_stars, r_pol_stars - z_pol_stars, marker='.', color='C0')
ax.scatter(z_sz_agn - I1_sz_agn, r_sz_agn - z_sz_agn, marker='.', color='C1', label='SPT-SZ/targeted')
ax.scatter(z_pol_agn - I1_pol_agn, r_pol_agn - z_pol_agn, marker='o', edgecolor='C1', facecolor='none', label='SPTpol/SSDF')
ax.legend()
ax.set(xlabel="PISCO z' - [3.6] (AB)", ylabel="PISCO r' - z' (AB)")
fig.savefig('Data/Data_Repository/Project_Data/SPT-IRAGN/SPTSZ_SPTpol_photometry_comparison/PISCO_photometry/Color-Color/'
            "SPT_stacked_PISCO_r'-z'-I1_color_PISCO_z'_CLASS_STAR.pdf")
plt.show()
