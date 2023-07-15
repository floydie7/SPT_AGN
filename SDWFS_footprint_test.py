"""
SDWFS_footprint_test.py
Author: Benjamin Floyd

Removes SDWFS footprints from the background sample if they have any associated structures in-image.
"""
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, setdiff, unique
from astropy.wcs import WCS
import astropy.units as u
import shapely.geometry as sg
from shapely import STRtree

cosmo = FlatLambdaCDM(H0=10, Om0=0.3)

# Read in the SDWFS image WCS
I1_SDWFS_wcs = WCS('Data_Repository/Images/Bootes/SDWFS/I1_bootes.v32.fits')
I1_SDWFS_pixel_scale = I1_SDWFS_wcs.proj_plane_pixel_scales()[0]

# Read in the current cutout catalog
sdwfs_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN.fits')
sdwfs_iragn_grp = sdwfs_iragn.group_by('CUTOUT_ID')

# Read in the ISCS cluster catalog
iscs_clusters = Table.read('Data_Repository/Catalogs/Bootes/ISCS/cluster.list.v10.Nsp1_official.clean_with_Zest.fits')

# Get the mask WCSs
mask_wcss = [WCS(cutout['MASK_NAME'][0]) for cutout in sdwfs_iragn_grp.groups]

# Get the cutout centers
cutout_centers = [(cutout['SZ_RA'][0] * u.deg, cutout['SZ_DEC'][0] * u.deg) for cutout in sdwfs_iragn_grp.groups]

# Create the boundaries of the cutouts
cutout_anchors = [np.array(I1_SDWFS_wcs.wcs_world2pix(cutout_center[0].value, cutout_center[1].value, 0))
                  - np.array(mask_wcs.pixel_shape) / 2
                  for cutout_center, mask_wcs in zip(cutout_centers, mask_wcss)]
cutout_bounds = [[anchor[0], anchor[1], anchor[0] + mask_wcs.pixel_shape[0], anchor[1] + mask_wcs.pixel_shape[1]]
                 for anchor, mask_wcs in zip(cutout_anchors, mask_wcss)]

# Create the cutout footprint polygons
footprint_boxes = [sg.box(*bounds) for bounds in cutout_bounds]

# Create the cluster boundaries
cluster_circles = []
for cluster in iscs_clusters:
    cluster_center = I1_SDWFS_wcs.wcs_world2pix(cluster['RA'], cluster['DEC'], 0)
    cluster_radius = (1 * u.Mpc * cosmo.arcsec_per_kpc_proper(cluster['z2']).to(I1_SDWFS_pixel_scale.unit / u.Mpc)
                      / I1_SDWFS_pixel_scale)
    cluster_circles.append(sg.Point(cluster_center).buffer(cluster_radius.value))

# Create a Sort-Tile-Recursive tree to optimize our intersection searches
cutout_tree = STRtree(footprint_boxes)

# Find all cutout footprints that intersect with the cluster regions
cutouts_with_structure = np.unique(cutout_tree.query(cluster_circles, predicate='intersects')[1])

# Write out a modified IRAGN catalog omitting the cutouts with structure
sdwfs_cutouts_with_structure = sdwfs_iragn_grp.groups[cutouts_with_structure]
sdwfs_cutouts_with_structure.group_by('CUTOUT_ID').groups.keys.pprint(max_lines=-1)
# sdwfs_iragn_no_structure = setdiff(sdwfs_iragn, sdwfs_cutouts_with_structure, keys='CUTOUT_ID')
# sdwfs_iragn_no_structure.write('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN_no_structure.fits')

manual_id_list = ['000', '001', '002', '003', '004', '005', '006', '007', '009',
                  '010', '011', '012', '013', '014', '015', '016', '019', '020',
                  '022', '024', '027', '029', '032', '035', '036', '037', '038',
                  '039', '040', '041', '042', '045', '047', '048', '050', '051',
                  '053', '054', '058', '059', '061', '063', '064', '065', '068',
                  '069', '070', '071', '073', '074', '076', '077', '079', '080',
                  '082', '085', '086', '087', '089', '090', '093', '094', '095',
                  '096', '097', '098', '099']
manual_id_list = [f'SDWFS_cutout_{id_num}' for id_num in manual_id_list]
sdwfs_iragn_no_structure = sdwfs_iragn[~np.isin(sdwfs_iragn['CUTOUT_ID'], manual_id_list)]
sdwfs_iragn_no_structure.write('Data_Repository/Project_Data/SPT-IRAGN/Output/SDWFS_cutout_IRAGN_no_structure.fits',
                               overwrite=True)
