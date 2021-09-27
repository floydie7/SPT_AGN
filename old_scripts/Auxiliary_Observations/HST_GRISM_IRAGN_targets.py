"""
HST_GRISM_IRAGN_targets.py
Author: Benjamin Floyd

Selects for IR-bright AGN for use on HST GRISM proposal (2018)
"""

from astropy.table import Table, vstack

# Read in tables
spt0313 = Table.read('Data/Output/SPT-CLJ0313-5334_AGN.cat', format='ascii')
spt0459 = Table.read('Data/Output/SPT-CLJ0459-4947_AGN.cat', format='ascii')
spt0607 = Table.read('Data/Output/SPT-CLJ0607-4448_AGN.cat', format='ascii')
spt2040 = Table.read('Data/Output/SPT-CLJ2040-4451_AGN.cat', format='ascii')
spt0446 = Table.read('Data/Output/SPT-CLJ0446-4606_AGN.cat', format='ascii')
spt0421 = Table.read('Data/Output/SPT-CLJ0421-4845_AGN.cat', format='ascii')

# Sort the tables by their RA
spt0313.sort('ALPHA_J2000')
spt0459.sort('ALPHA_J2000')
spt0607.sort('ALPHA_J2000')
spt2040.sort('ALPHA_J2000')
spt0446.sort('ALPHA_J2000')
spt0421.sort('ALPHA_J2000')

# Combine all the tables
ir_agn = vstack([spt0313, spt0421, spt0446, spt0459, spt0607, spt2040])

# Select for only AGN within 1.5 arcmin of the SZ center
ir_agn15 = ir_agn[ir_agn['RADIAL_DIST'] <= 1.5]

# Only want SPT_ID, RA, and Dec for columns
ir_agn_targets = ir_agn15['SPT_ID', 'ALPHA_J2000', 'DELTA_J2000']
ir_agn_targets['ALPHA_J2000'].name = 'RA'
ir_agn_targets['DELTA_J2000'].name = 'DEC'

# Write target list to file
ir_agn_targets.write('Data/HST_GRISM_Observation/SPT_IR_AGN_HST-GRISM_target_list.cat', format='ascii')
