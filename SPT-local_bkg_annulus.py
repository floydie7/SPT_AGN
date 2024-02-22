"""
SPT-local_bkg_annulus.py
Author: Benjamin Floyd

Determines the necessary annulus outer radius used for the background measurement around the clusters. The annulus is
determined iteratively for each cluster until the Poisson fractional error is approximately 5%.
"""
import glob
import json
import re
from itertools import groupby

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable, setdiff
from tqdm import tqdm

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
cluster_id = re.compile(r'SPT-CLJ\d+-\d+')

# Set our selection magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

# Set our correction factors needed to be able to use IRAC magnitude cuts
w1_correction = -0.11 * u.mag
w2_correction = -0.07 * u.mag

# Set our background annulus ranges in terms of r200 radii
inner_radius_factor = 3

# Read in and process the SPT WISE galaxy catalogs
spt_wise_gal_data = {}
wise_catalog_names = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/*_wise_local_bkg.ecsv')
gaia_catalog_names = glob.glob('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/catalogs/gaia/*_bkgs_gaia.fits')
catalog_names = sorted([*wise_catalog_names, *gaia_catalog_names], key=lambda s: cluster_id.search(s).group(0))
catalog_names = {cluster_name: list(names)
                 for cluster_name, names in groupby(catalog_names, key=lambda s: cluster_id.search(s).group(0))}

for cluster_name, (wise_catalog_name, gaia_catalog_name) in tqdm(catalog_names.items(),
                                                                 desc='Finding background annulus radii'):
    spt_wise_gal = QTable.read(wise_catalog_name)
    spt_gaia_cat = QTable.read(gaia_catalog_name)
    spt_gaia_stars = spt_gaia_cat[~(spt_gaia_cat['in_qso_candidates'] | spt_gaia_cat['in_galaxy_candidates'])]

    # Apply photometric correction factors
    spt_wise_gal['w1mpro'] = spt_wise_gal['w1mpro'] + w1_correction
    spt_wise_gal['w2mpro'] = spt_wise_gal['w2mpro'] + w2_correction

    # Select only objects within the magnitude ranges
    spt_wise_gal = spt_wise_gal[(ch1_bright_mag < spt_wise_gal['w1mpro'].value) &
                                (spt_wise_gal['w1mpro'].value <= ch1_faint_mag) &
                                (ch2_bright_mag < spt_wise_gal['w2mpro'].value) &
                                (spt_wise_gal['w2mpro'].value <= ch2_faint_mag)]

    # Excise the cluster and only select objects in our chosen annulus
    spt_wise_gal_cluster_coord = SkyCoord(spt_wise_gal['SZ_RA'][0], spt_wise_gal['SZ_DEC'][0], unit=u.deg)
    spt_wise_gal_coords = SkyCoord(spt_wise_gal['ra'], spt_wise_gal['dec'], unit=u.deg)
    spt_gaia_star_coords = SkyCoord(spt_gaia_stars['ra'], spt_gaia_stars['dec'], unit=u.deg)

    # Remove stars
    spt_wise_gal_idx, spt_wise_gal_sep, _ = spt_gaia_star_coords.match_to_catalog_sky(spt_wise_gal_coords)
    spt_wise_stars = spt_wise_gal[spt_wise_gal_idx[spt_wise_gal_sep < 1 * u.arcsec]]
    spt_wise_gal = setdiff(spt_wise_gal, spt_wise_stars, keys=['ra', 'dec'])

    spt_wise_gal_coords = SkyCoord(spt_wise_gal['ra'], spt_wise_gal['dec'], unit=u.deg)
    spt_wise_gal_sep_deg = spt_wise_gal_cluster_coord.separation(spt_wise_gal_coords)

    inner_radius_mpc = inner_radius_factor * spt_wise_gal['R200'][0]
    inner_radius_deg = inner_radius_mpc * cosmo.arcsec_per_kpc_proper(spt_wise_gal['REDSHIFT'][0]).to(u.deg / u.Mpc)

    outer_radius_deg = inner_radius_deg + 5 * u.arcsec
    while True:
        temp_cat = spt_wise_gal[(inner_radius_deg < spt_wise_gal_sep_deg) & (spt_wise_gal_sep_deg <= outer_radius_deg)]
        frac_err = np.sqrt(len(temp_cat)) / len(temp_cat)

        if np.isclose(frac_err, 0.05, atol=0.01, rtol=0.0):
            break
        else:
            outer_radius_deg += 5 * u.arcsec

    # Also compute the annulus area
    spt_bkg_area = np.pi * (outer_radius_deg ** 2 - inner_radius_deg ** 2)

    spt_wise_gal_data[cluster_name] = {'inner_radius_deg': inner_radius_deg.value,
                                       'outer_radius_deg': outer_radius_deg.value,
                                       'annulus_area': spt_bkg_area.value,
                                       'frac_err': frac_err}

with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/SPTcl-local_bkg_annulus_no_stars.json', 'w') as f:
    json.dump(spt_wise_gal_data, f)