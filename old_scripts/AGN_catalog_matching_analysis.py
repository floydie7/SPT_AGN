"""
AGN_catalog_matching_analysis.py
Author: Benjamin Floyd

Examining the issues that have been discovered in the AGN catalog generation related to the matching of the IRAC images
to the Bleem+15 SPT-SZ catalog.
"""

from astropy.table import Table, unique
from astropy.io import fits
import numpy as np
from Pipeline_functions import file_pairing
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt


# Load in external files as we would in the main script.
Bleem = Table(fits.getdata('Data/2500d_cluster_sample_fiducial_cosmology.fits'))  # Bleem+15 Master SPT Cluster catalog

# Clean the table of the rows without mass data. These are unconfirmed cluster candidates.
Bleem = Bleem[np.where(Bleem['M500'] != 0.0)]

# Run the file pairing method.
cluster_list = file_pairing('Data/Catalogs/', 'Data/Images/')
print("File pairing complete, Clusters in directory: {num_clusters}".format(num_clusters=len(cluster_list)))

# The following lines are taken from the `catalog_image_match` method.
catalog = Bleem
cat_ra_col, cat_dec_col = 'RA', 'DEC'

# Create astropy skycoord object from the catalog columns.
cat_coords = SkyCoord(catalog[cat_ra_col], catalog[cat_dec_col], unit=u.degree)

for cluster in cluster_list:
    # Array element names
    irac_ch1_sci = cluster['ch1_sci_path']

    # Get the RA and Dec of the center pixel in the image.
    img_ra = fits.getval(irac_ch1_sci, 'CRVAL1')
    img_dec = fits.getval(irac_ch1_sci, 'CRVAL2')

    # Create astropy skycoord object for the center pixel of the image.
    img_coord = SkyCoord(img_ra, img_dec, unit=u.degree)

    # Preform the catalog matching.
    idx, sep, _ = img_coord.match_to_catalog_sky(cat_coords)

    # Add the (nearest) catalog id and separation (in arcsec) to the output array.
    cluster.update({'Bleem_idx': idx, 'center_sep': sep.arcmin.item(), 'SPT_ID': catalog[idx]['SPT_ID']})

# If there are any duplicate matches in the sample remaining we need to remove the match that is the poorer
# match. We will only keep the closest matches.
# First set up a table of the index of the cluster dictionaries in cluster_list, the recorded Bleem index, and
# the recorded separation.
match_info = Table(names=['list_idx', 'Bleem_idx', 'center_sep', 'SPT_ID', 'Image_ID'], dtype=['i8', 'i8', 'f8', 'S16', 'S37'])
for i in range(len(cluster_list)):
    match_info.add_row([i, cluster_list[i]['Bleem_idx'], cluster_list[i]['center_sep'], cluster_list[i]['SPT_ID'], cluster_list[i]['sex_cat_path'][-23:-7]])

# Sort the table by the Bleem index.
match_info.sort(['Bleem_idx', 'center_sep'])

# Use Astropy's unique function to remove the duplicate rows. Because the table rows will be subsorted by the
# separation column we only need to keep the first incidence of the Bleem index as our best match.
match_info = unique(match_info, keys='Bleem_idx', keep='first')
#
# # Resort the table by the list index (not sure if this is necessary).
# match_info.sort('list_idx')
#
# # Generate the output list using the remaining indices in the table.
# cluster_list = [cluster_list[i] for i in match_info['list_idx']]

# match_info.pprint(max_lines=-1)

match_info_6 = match_info[np.where((match_info['center_sep'] <= 6) & (match_info['center_sep'] > 1))]
# match_info_6.write(sys.stdout, format='latex')

# Using the matched lists, plot a histogram of the separations.
# separations = np.array([cluster['center_sep'] for cluster in cluster_list])
separations = match_info['center_sep']

print('min:{min}, max:{max}'.format(min=np.min(separations), max=np.max(separations)))
print('number of pairs with separations larger than 1 arcmin: {}'.format(len(separations[np.where(separations > 1)])))
print('Number of pairs with separations 1 < sep <= 6: {}'.format(len(separations[np.where((separations > 1) & (separations <= 6))])))

fig, ax = plt.subplots()
ax.hist(separations, bins='auto', range=(0, 6))
ax.set(title='Matching Ch1 image CRVAL to Bleem+15 SZ Center', xlabel='separation (arcmin)')
fig.savefig('Data/Plots/SZ_to_CRVAL_separations.pdf', format='pdf')
