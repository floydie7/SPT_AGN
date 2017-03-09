"""
Catalog_fixer.py
Author: Benjamin Floyd
Some of the SExtractor catalogs have a tab character which interferes with astropy reading the tables in correctly.
"""

import os
import numpy as np
from astropy.io import ascii


for files in os.listdir('Data/Catalogs.old/'):
    if not files.startswith('.'):
        print("Opening: " + files)
        with open('Data/Catalogs.old/' + files, 'r') as sexcat:
            # Read all the lines in.
            lines = sexcat.readlines()

            i = 0
            header = []
            # From the top lines grab the header of the catalog.
            while lines[i][0] == '#':
                header.append(lines[i])
                i += 1

            # For the rest of the file get all the data and replace the tab character with a space.
            data = []
            for j in np.arange(i, len(lines)):
                data.append(lines[j].replace('\t', ' '))

        # Write the corrected file out.
        print("Writing: " + files)
        with open('Data/Catalogs.old/' + files, 'w') as outcat:
            outcat.writelines(header)
            outcat.writelines(data)

        # Read the catalog back in, this time using Astropy, and rename the columns.
        tempcat = ascii.read('Data/Catalogs.old/' + files)
        col_names = ['SPT_ID', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'KRON_RADIUS', 'BACKGROUND',
                     'FLUX_RADIUS', 'ALPHAPEAK_J2000', 'DELTAPEAK_J2000', 'X2_IMAGE', 'Y2_IMAGE', 'XY_IMAGE', 'A_IMAGE',
                     'B_IMAGE', 'THETA_IMAGE', 'A_WORLD', 'B_WORLD', 'THETA_WORLD', 'CLASS_STAR', 'FLAGS', 'I1_MAG_AUTO',
                     'I1_MAGERR_AUTO', 'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I1_MAG_APER6', 'I1_MAGERR_APER6', 'I1_FLUX_AUTO',
                     'I1_FLUXERR_AUTO', 'I1_FLUX_APER4', 'I1_FLUXERR_APER4', 'I1_FLUX_APER6', 'I1_FLUXERR_APER6',
                     'I2_MAG_AUTO', 'I2_MAGERR_AUTO', 'I2_MAG_APER4', 'I2_MAGERR_APER4', 'I2_MAG_APER6', 'I2_MAGERR_APER6',
                     'I2_FLUX_AUTO', 'I2_FLUXERR_AUTO', 'I2_FLUX_APER4', 'I2_FLUXERR_APER4', 'I2_FLUX_APER6',
                     'I2_FLUXERR_APER6']
        i = 0
        for col in tempcat.keys():
            tempcat.rename_column(col, col_names[i])
            i += 1

        # Finally, write the corrected catalog out now that it has the correct formatting.
        ascii.write(tempcat, 'Data/Catalogs/' + files)
