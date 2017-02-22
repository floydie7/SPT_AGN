"""
Catalog_fixer.py
Author: Benjamin Floyd
Some of the SExtractor catalogs have a tab character which interferes with astropy reading the tables in correctly.
"""

import os
from astropy.io import ascii


for files in os.listdir('Data/test/'):
    with open('Data/test/' + files, 'r+') as sexcat:
        for line in sexcat:
            sexcat.write(line.replace('\t', ' '))

# ascii.read('Data/test/SPT-CLJ0000-4356.v2.cat', format='sextractor')