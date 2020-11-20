"""
k_correction_raw_test.py
Author: Benjamin Floyd

Performs the same test against the CWW E SED that the second half of `abs_mag_test.py` does but done more by hand.
"""

import numpy as np
from astropy.table import Table
from scipy.integrate import quadrature
from scipy.interpolate import interp1d

# Set the redshift
z = 0.2

# Read in the SED
cww_e_data = np.genfromtxt('../../Data/Data_Repository/SEDs/CWW/CWW_E_ext.sed').T

# Read in the SDSS u-band filter
sdss_u_data = Table.read('Data/Data_Repository/filter_curves/SDSS/filter_curves.fits', hdu=1)

# Read in the Fukugita et al. (1995) K-corrections for comparision
fukugita95 = Table.read('Data/fukugita_fig20_sdss_ug_kcorr.csv', data_start=2,
                        names=['redshift_u', 'k_correction_u', 'redshift_g', 'k_correction_g'])

# Create interpolation functions
cww_e_sed = interp1d(cww_e_data[0], cww_e_data[1])
sdss_u_filter = interp1d(sdss_u_data['wavelength'], sdss_u_data['respt'])
fukugita95_u = interp1d(fukugita95['redshift_u'], fukugita95['k_correction_u'])

# Compose the integrands
lam_f_R = lambda lam: lam * cww_e_sed(lam) * sdss_u_filter(lam)
lam_f_z1_R = lambda lam, z_obs: lam * cww_e_sed((1 + z_obs) * lam) * sdss_u_filter(lam)

# Our integration axis is the range of wavelengths the filter covers
wavelengths = sdss_u_data['wavelength']  # In Angstroms

# Perform the integrations
int_lam_f_R = quadrature(lam_f_R, a=np.min(wavelengths), b=np.max(wavelengths), maxiter=10000)[0]
int_lam_f_z1_R = quadrature(lam_f_z1_R, a=np.min(wavelengths), b=np.max(wavelengths), args=(z,), maxiter=1000)[0]

# Compose the K-correction
k_corr = -2.5 * np.log10((1 / (1 + z)) * int_lam_f_R / int_lam_f_z1_R)

# Check against the Fukugita+95 value at the same redshift
fukugita_k_corr = fukugita95_u(z)

print(f'My K-correction:{k_corr:.2f}\nFukugita+95 K-correction: {fukugita_k_corr:.2f}')