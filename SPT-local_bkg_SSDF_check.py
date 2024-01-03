"""
SPT-local_bkg_SSDF_check.py
Author: Benjamin Floyd

This script provides a direct check against the SPT WISE galaxies--SDWFS IRAC AGN estimation of the background AGN in
an annulus around the cluster. It also provides direct measurements of the IRAC AGN background for all clusters in the
SSDF footprint (i.e., the SPTpol 100d clusters).
"""
import json
import re
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astro_compendium.utils.json_helpers import NumpyArrayEncoder
from astro_compendium.utils.small_poisson import small_poisson
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable, vstack
from colossus.cosmology import cosmology
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from tqdm import tqdm

cluster_id = re.compile(r'SPT-CLJ\d+-\d+')

# Set up cosmology objects
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048, Tcmb0=2.7255, Neff=3)
colo_cosmo = cosmology.fromAstropy(cosmo, sigma8=0.8, ns=0.96, cosmo_name='concordance')

# Set our selection magnitude ranges
ch1_bright_mag = 10.0  # Bright-end 3.6 um magnitude
ch1_faint_mag = 18.3  # Faint-end 3.6 um magnitude
ch2_bright_mag = 10.45  # Bright-end 4.5 um magnitude
ch2_faint_mag = 17.48  # Faint-end 4.5 um magnitude

# Set our magnitude binning
mag_bin_width = 0.25
magnitude_bins = np.arange(ch2_bright_mag, ch2_faint_mag, mag_bin_width)
magnitude_bin_centers = magnitude_bins[:-1] + np.diff(magnitude_bins) / 2

# Set our background annulus ranges in terms of r200 radii
inner_radius_factor = 3


@dataclass
class ClusterInfo:
    inner_radius: u.Quantity
    outer_radius: u.Quantity
    annulus_area: u.Quantity
    cluster_data: QTable = None
    bkg_catalog: QTable = None


# Read in the SSDF catalog
ssdf = QTable.read('Data_Repository/Catalogs/SSDF/SSDF2.20170125.v10.public.cat', format='ascii')

# Read in the SPTpol 100d IR-AGN catalog (this way we only work on clusters that reside within the SSDF footprint)
sptpol_iragn = QTable.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTpol_100d_IRAGN.fits')

# Group to only have cluster information
sptcl_iragn_grp = sptpol_iragn.group_by('SPT_ID')
sptcl_clusters = vstack([QTable(cluster['SPT_ID', 'SZ_RA', 'SZ_DEC', 'REDSHIFT', 'M500', 'R500'][0])
                         for cluster in sptcl_iragn_grp.groups])

# Add units
sptcl_clusters['M500'].unit = u.Msun

# Read in the annulus information that we previously calculated
with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/SPTcl-local_bkg_annulus.json', 'r') as f:
    spt_wise_annuli_data = json.load(f)


# Extract the coordinates of the galaxies in SSDF
ssdf_coords = SkyCoord(ssdf['ALPHA_J2000'], ssdf['DELTA_J2000'], unit=u.deg)

# From the full SSDF catalog, create subcatalogs of the background annulus region
spt_bkg_gal_data = {}
for cluster in tqdm(sptcl_clusters, desc='Creating Background Catalogs in SSDF'):
    cluster_center = SkyCoord(cluster['SZ_RA'], cluster['SZ_DEC'], unit=u.deg)
    inner_radius = spt_wise_annuli_data[cluster['SPT_ID']]['inner_radius_deg'] * u.deg
    outer_radius = spt_wise_annuli_data[cluster['SPT_ID']]['outer_radius_deg'] * u.deg
    annulus_area = spt_wise_annuli_data[cluster['SPT_ID']]['annulus_area'] * u.deg ** 2

    # Select for galaxies within the background annulus
    sep = cluster_center.separation(ssdf_coords)
    background_ssdf = ssdf[(inner_radius <= sep) & (sep < outer_radius)]

    spt_bkg_gal_data[cluster['SPT_ID']] = ClusterInfo(inner_radius=inner_radius, outer_radius=outer_radius,
                                                      annulus_area=annulus_area, cluster_data=cluster,
                                                      bkg_catalog=background_ssdf)

# Read in the color threshold--redshift relations
with open('Data_Repository/Project_Data/SPT-IRAGN/SDWFS_background/SDWFS_purity_color_4.5_17.48.json', 'r') as f:
    sdwfs_purity_data = json.load(f)
z_bins = sdwfs_purity_data['redshift_bins'][:-1]
agn_purity_color = interp1d(z_bins, sdwfs_purity_data['purity_90_colors'], kind='previous')

# Compute the number count distributions of the IRAC AGN in the SSDF annuli
for cluster_data in tqdm(spt_bkg_gal_data.values(), desc='Computing dN/dm distributions'):
    catalog = cluster_data.bkg_catalog
    spt_bkg_area = cluster_data.annulus_area
    cluster_z = cluster_data.cluster_data['REDSHIFT']

    # Calculate our weighting factor
    dndm_weight = spt_bkg_area.value * mag_bin_width

    # Select for AGN in the SSDF catalog
    catalog = catalog[(ch1_bright_mag < catalog['MAG_APER_1'].value) & (catalog['MAG_APER_1'].value <= ch1_faint_mag) &
                      (ch2_bright_mag < catalog['MAG_APER_2'].value) & (catalog['MAG_APER_2'].value <= ch2_faint_mag)]
    catalog = catalog[catalog['MAG_APER_1'].value - catalog['MAG_APER_2'].value >= agn_purity_color(cluster_z)]

    # Create histogram
    spt_bkg_dndm, _ = np.histogram(catalog['MAG_APER_2'].value, bins=magnitude_bins)
    spt_bkg_dndm_weighted = spt_bkg_dndm / dndm_weight

    # Compute the errors
    spt_bkg_dndm_err = tuple(err / dndm_weight for err in small_poisson(spt_bkg_dndm))[::-1]

    # Store the number count distribution and errors for later
    cluster_data.bkg_dndm = spt_bkg_dndm_weighted
    cluster_data.bkg_dndm_err = spt_bkg_dndm_err

# For each cluster, integrate the SSDF background number count distribution over the selection magnitude range to find
# the true IRAC AGN local background for these clusters
spt_ssdf_local_bkg_agn_surf_den = {cluster_name: simpson(cluster_data.bkg_dndm, magnitude_bin_centers)
                                   for cluster_name, cluster_data in tqdm(spt_bkg_gal_data.items(),
                                                                          desc='Integrating SSDF dN/dm')}

with open('Data_Repository/Project_Data/SPT-IRAGN/local_backgrounds/SPT-SSDF_local_bkg.json', 'w') as f:
    json.dump(spt_ssdf_local_bkg_agn_surf_den, f, cls=NumpyArrayEncoder)
