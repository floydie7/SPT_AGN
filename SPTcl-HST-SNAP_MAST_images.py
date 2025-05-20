"""
SPTcl-HST-SNAP_MAST_images.py
Author: Benjamin Floyd

Downloads all HST-SNAP program images from MAST for all SPT cluster in the SPTcl-IRAGN sample.
"""

from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.mast import Observations
from tqdm.contrib import tzip

# Read in spt agn catalog
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN_no-stars.fits')

# Just cluster information
cluster_info = vstack([cluster[0] for cluster in sptcl_iragn.group_by('SPT_ID').groups])

# Get coordinates
cluster_coords = SkyCoord(cluster_info['SZ_RA'], cluster_info['SZ_DEC'], unit='deg')

# Login to MAST using token
my_session = Observations.login(token='607b9fb61bdf49a3a78d29042de4dc79')

for cluster_id, cluster_coord in tzip(cluster_info['SPT_ID'], cluster_coords):
    # Make initial query to MAST
    obs_table = Observations.query_criteria(coordinates=cluster_coord, radius='0.5m', obs_collection='HST',
                                            filters=['F200LP', 'F110W'], intentType='science', provenance_name='HAP-*')

    if not obs_table:
        print(f'No HST-SNAP images found for cluster {cluster_id}')
        continue

    # Create a directory to store the downloaded data
    download_dir = Path(f'Data_Repository/Images/SPT/HST-SNAP2/{cluster_id}')
    download_dir.mkdir(parents=True, exist_ok=True)

    # Get list of available data products
    data_products = Observations.get_product_list(obs_table)

    # Filter the products list to only get science images with Multi-visit Mosaics.
    science_products = Observations.filter_products(data_products, productType=['SCIENCE'], calib_level=[3],
                                                    project='HAP-SVM', filters=['F200LP', 'F110W'], mrp_only=False)

    # Download the images
    manifest = Observations.download_products(science_products, download_dir=download_dir,
                                              extension=['drz.fits', 'drc.fits'], flat=True)
my_session = Observations.logout()
