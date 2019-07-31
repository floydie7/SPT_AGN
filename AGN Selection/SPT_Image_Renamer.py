"""
SPT_Image_Renamer.py
Author: Benjamin Floyd
Short script to unify the file naming scheme of the SPT cluster images so they can be more easily used for analysis.
"""

import glob
import os
import shutil

orig_image_dir = '/Users/btfkwd/Desktop/Files_to_add/'
orig_cat_dir = '/Users/btfkwd/Desktop/Files_to_add/'
new_image_dir = '/Users/btfkwd/Documents/SPT_AGN/Data/Images/'
new_cat_dir = '/Users/btfkwd/Documents/SPT_AGN/Data/Catalogs.old/'

# Images
for f in glob.glob(orig_image_dir+'*.fits'):
    fname = f.replace(orig_image_dir, '')

    # Change the suffix
    if f.endswith('_cutout.fits'):
        new_f = f.replace('_cutout.fits', '.cutout.fits')
        os.rename(f, new_f)
        f = new_f

    # Change the prefix
    if 'I1_CLJ'in fname or 'I2_CLJ' in fname:
        shutil.move(f, new_image_dir+fname.replace('CLJ', 'SPT-CLJ'))
    elif 'I1_J' in fname or 'I2_J' in fname:
        shutil.move(f, new_image_dir+fname.replace('J', 'SPT-CLJ'))
    elif 'I1_SPT' in fname or 'I2_SPT' in fname:
        shutil.move(f, new_image_dir+fname.replace('SPT', 'SPT-CLJ'))

# Catalogs
for f in glob.glob(orig_cat_dir+'*.cat'):
    fname = f.replace(orig_cat_dir, '')

    # Change the prefix
    if fname.startswith('CLJ'):
        shutil.move(f, new_cat_dir+fname.replace('CLJ', 'SPT-CLJ'))
    elif fname.startswith('J'):
        shutil.move(f, new_cat_dir+fname.replace('J', 'SPT-CLJ'))
    elif fname.startswith('SPT'):
        shutil.move(f, new_cat_dir+fname.replace('SPT', 'SPT-CLJ'))