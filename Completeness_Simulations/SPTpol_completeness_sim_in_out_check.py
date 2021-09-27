"""
SPTpol_completeness_sim_in_out_check.py
Author: Benjamin Floyd

Examines the diagnostic catalogs of the SPTpol completeness simulations to see any discrepancies with the photometry.
"""
import re

import numpy as np
from astropy.table import Table
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Get the catalogs
inout_cats = glob.glob('Data/Comp_Sim/SPTpol/Input_Output_catalogs/*.fits')

for catalog in inout_cats:
    spt_id = re.search(r'SPT-CLJ\d+-\d+', catalog).group(0)
    # Read in the catalog
    full_cat = Table.read(catalog, masked=True)
    full_cat['MAG_APER'].mask = np.isnan(full_cat['MAG_APER'])
    full_cat['MAG_APER'].fill_value = -99
    full_cat = full_cat.filled()

    # Filter for only the objects we would have counted as 'recovered' (using a magnitude difference of 0.2 mag)
    recovered = full_cat[np.abs(full_cat['selection_band'] - full_cat['MAG_APER']) <= 0.2]

    # Make a histogram
    fig, ax = plt.subplots()
    ax.hist(recovered['selection_band'] - recovered['MAG_APER'], bins='auto')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set(title=f'{spt_id} Completeness Sim (recovered)', xlabel='MAG_IN - MAG_OUT', ylabel='Recovered Objects',
           xlim=[-0.2, 0.2])
    fig.savefig(f'Data/Comp_Sim/SPTpol/Plots/Input_Output_Checks/{spt_id}_inout.pdf', format='pdf')
    plt.close()
