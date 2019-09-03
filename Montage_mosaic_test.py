"""
Montage_mosaic_test.py
Author: Benjamin Floyd

Integration test for the montage_mosaic.py API to see if it can complete the mosaicking process from start to finish.
"""

from montage_mosaic import montage_mosaic

hcc_prefix = '/work/mei/bfloyd/SPT_AGN/'

# Start by testing making a single I1 science mosaic for tile SSDF0.2 and 0.3.
I1_sci_files = [hcc_prefix+'Data/SPTPol/images/ssdf_tiles/I1_SSDF0.2_mosaic.fits',
                hcc_prefix+'Data/SPTPol/images/ssdf_tiles/I1_SSDF0.3_mosaic.fits']

out_dir = hcc_prefix+'Data/SPTPol/images/mosaic_tiles/'

montage_mosaic(I1_sci_files, out_file=out_dir+'I1_SSDF0.2_0.3_mosaic.fits',
               workdir=out_dir+'I1_SSDF0.2_0.3_sci', clean_workdir=False)
