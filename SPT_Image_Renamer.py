"""
SPT_Image_Renamer.py
Author: Benjamin Floyd
Short script to unify the file naming scheme of the SPT cluster images so they can be more easily used for analysis.
"""

import os

# Images
filedir = 'Data/Images/'

os.chdir(filedir)

for f in os.listdir('.'):
    os.rename(f, f.replace('SPT-CLJ-CLJ-CLJ-CLJ-CLJ-CLJ-CLJ', 'SPT-CLJ'))

# for f in os.listdir('.'):
#     if f.startswith('I1_CLJ') or f.startswith('I2_CLJ'):
#         os.rename(f, f.replace('CLJ', 'SPT-CLJ'))
#     elif f.startswith('I1_J') or f.startswith('I2_J'):
#         os.rename(f, f.replace('J', 'SPT-CLJ'))
#     elif f.startswith('I1_SPT') or f.startswith('I2_SPT'):
#         os.rename(f, f.replace('SPT', 'SPT-CLJ'))
#
# # Catalogs
# filedir = '../Catalogs/'
#
# os.chdir(filedir)
#
# for f in os.listdir('.'):
#     if f.startswith('CLJ'):
#         os.rename(f, f.replace('CLJ', 'SPT-CLJ'))
#     elif f.startswith('J'):
#         os.rename(f, f.replace('J', 'SPT-CLJ'))
#     elif f.startswith('SPT'):
#         os.rename(f, f.replace('SPT', 'SPT-CLJ'))