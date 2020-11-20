"""
JWST_K15_SED_prep.py
Author: Benjamin Floyd

Reads in the Kirkpatrick+15 AGN SEDs and converts the luminosity column to flux and writes out the converted file for
use with the JWST ETC.
"""

import glob
import re

import astropy.units as u
import numpy as np
from astropy.table import Table

# Collect the SED filenames
sed_files = glob.glob('Data/Data_Repository/SEDs/Kirkpatrick/MIR-Based/*.txt')

# Extract the SED name from the filename
sed_name_pattern = re.compile(r'/(MIR\d.\d).txt')

# Read in the files
seds = {sed_name_pattern.search(f).group(1): Table.read(f, format='ascii', data_start=3,
                                                        names=['wavelength', 'luminosity', 'dLnu']) for f in sed_files}

for sed_name, sed in seds.items():
    # Attach units to the columns
    sed['wavelength'].unit = u.um
    sed['luminosity'].unit = u.W / u.Hz
    sed['dLnu'].unit = u.W / u.Hz

    # Convert the luminosity density to a flux density
    sed['flux'] = (sed['luminosity'] / (4 * np.pi * (10 * u.pc) ** 2)).to(u.mJy, equivalencies=u.spectral_density(
        sed['wavelength']))

