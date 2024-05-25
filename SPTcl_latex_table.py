"""
SPTcl_latex_table.py
Author: Benjamin Floyd
Writes a LaTeX table of SPTcl-IRAGN catalog.
"""

from astropy.io import ascii
from astropy.table import Table

# Read in the table
# sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/Output/SPTcl_IRAGN_no-stars.fits')
sptcl_iragn = Table.read('Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/local_backgrounds/eta-zeta_slopes/targeted_snr/mock_AGN_catalog_t0.0053_e4.00_z-1.00_b1.00_rc0.100_C0.181_maxr5.00_seed3775_308x1.0_tez_grid.fits')

# Select only AGN
sptcl_iragn = sptcl_iragn[sptcl_iragn['SELECTION_MEMBERSHIP'] >= 0.5]

# Reorganize the columns
# new_col_order = ['SPT_ID', 'SZ_RA', 'SZ_DEC', 'REDSHIFT', 'REDSHIFT_UNC', 'M500', 'M500_uerr', 'M500_lerr', 'R500',
#                  'ALPHA_J2000', 'DELTA_J2000',
#                  'I1_MAG_APER4', 'I1_MAGERR_APER4', 'I1_FLUX_APER4', 'I1_FLUXERR_APER4',
#                  'I2_MAG_APER4', 'I2_MAGERR_APER4', 'I2_FLUX_APER4', 'I2_FLUXERR_APER4',
#                  'J_ABS_MAG',
#                  'RADIAL_SEP_R500', 'RADIAL_SEP_ARCMIN',
#                  'SELECTION_MEMBERSHIP',
#                  'COMPLETENESS_CORRECTION',
#                  'MASK_NAME']
new_col_order = ['SPT_ID', 'SZ_RA', 'SZ_DEC', 'REDSHIFT', 'M500', 'R500',
                 'RA', 'DEC',
                 'J_ABS_MAG',
                 'RADIAL_SEP_R500',
                 'SELECTION_MEMBERSHIP',
                 'COMPLETENESS_CORRECTION']
sptcl_iragn = sptcl_iragn[new_col_order]

# Factor out the cluster masses
sptcl_iragn['M500'] /= 1e14
# sptcl_iragn['M500_uerr'] /= 1e14
# sptcl_iragn['M500_lerr'] /= 1e14

# Coordinate formatting
# for col in ['SZ_RA', 'SZ_DEC', 'ALPHA_J2000', 'DELTA_J2000']:
for col in ['SZ_RA', 'SZ_DEC', 'RA', 'DEC']:
    sptcl_iragn[col].format = '{:.4f}'

# Redshift formatting
# for col in ['REDSHIFT', 'REDSHIFT_UNC']:
for col in ['REDSHIFT']:
    sptcl_iragn[col].format = '{:.2f}'

# Mass formatting
# for col in ['M500', 'M500_uerr', 'M500_lerr']:
for col in ['M500']:
    sptcl_iragn[col].format = '{:.2f}'

# r500 formatting
sptcl_iragn['R500'].format = '{:.2f}'

# Photometry formatting
sptcl_iragn['J_ABS_MAG'].format = '{:.3f}'

# Radial formatting
# for col in ['RADIAL_SEP_R500', 'RADIAL_SEP_ARCMIN']:
for col in ['RADIAL_SEP_R500']:
    sptcl_iragn[col].format = '{:.2f}'

# Ancillary data formating
for col in ['SELECTION_MEMBERSHIP', 'COMPLETENESS_CORRECTION']:
    sptcl_iragn[col].format = '{:.2f}'

# Write to disk
sptcl_iragn.write('Data_Repository/Project_Data/SPT-IRAGN/Publication_Plots/mock_catalog.tex',
                  format='ascii.latex', latexdict=ascii.latex.latexdicts['AA'], overwrite=True)
