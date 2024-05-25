"""
SDWFS_fake_cluster_catalog.py
Author: Benjamin Floyd
Description: Randomly assigns cluster-level data from the SPT cluster catalogs to the SDWFS data and applies color cuts.
"""

from astropy.table import Table

# Read in the SDWFS IRAGN (cutout) catalog
sdwfs_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')

# Read in the cluster catalogs
