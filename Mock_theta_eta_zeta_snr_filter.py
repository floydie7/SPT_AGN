"""
Mock_theta_eta_zeta_snr_filter.py
Author: Benjamin Floyd

Check the cluster-to-background SNR and identifies the catalogs with good SNR.
"""

import glob
import re
import shutil

import numpy as np
from astropy.table import Table

param_pattern = re.compile(r'(?:[tezbCx]|rc)(-*\d+.\d+|\d+)')
eta_zeta_snr_pattern = re.compile(r'(?:[ez]|snr)(-*\d+.\d+|\d+)')
theta_eta_zeta_snr_pattern = re.compile(r'(?:[tez]|snr)(-*\d+.\d+|\d+)')

main_dir = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/eta_zeta_slopes/raw_grid/308cl_catalogs/'
new_dir = 'Data_Repository/Project_Data/SPT-IRAGN/MCMC/Mock_Catalog/Catalogs/Port_Rebuild_Tests/eta_zeta_slopes/targeted_snr/308cl/snr_5/'

# Get list of all catalogs
catalog_list = glob.glob(f'{main_dir}/*_tez_grid.fits')

# Define our grid for the parameters we want.
eta_range = [-5., -3., 0., 3., 4., 5.]
zeta_range = [-2., -1., 0., 1., 2.]
eta_zeta_grid = np.array(np.meshgrid(eta_range, zeta_range)).T.reshape(-1, 2)

target_snr = 5.
snr_tol = 1.

cat_dict, catalogs_to_test = set(), set()
for catalog_file in catalog_list:
    # Extract the parameter set
    params = np.array(param_pattern.findall(catalog_file), dtype=float)

    # Read in the catalog
    catalog = Table.read(catalog_file)

    # Separate the cluster and background objects
    cluster_agn = catalog[catalog['CLUSTER_AGN'].astype(bool)]
    background_agn = catalog[~catalog['CLUSTER_AGN'].astype(bool)]

    # Compute the corrected number of objects in the catalogs
    num_cl_agn = cluster_agn['COMPLETENESS_CORRECTION'].sum()
    num_bkg_agn = background_agn['COMPLETENESS_CORRECTION'].sum()

    # Signal-to-Noise ratio is defined as the numbers of cluster / background AGN
    snr = num_cl_agn / num_bkg_agn

    # print(f'{params = }: {snr = :.2f}')

    # First filter for only catalogs that are roughly our target SNR
    # if np.abs(snr - target_snr) <= snr_tol:
    # Now check if our parameter set is in the output dictionary already
    theta, eta, zeta = params[:3]
    current_keys = [np.array(theta_eta_zeta_snr_pattern.findall(k), dtype=float)
                    for k in cat_dict if f'e{eta:.3f}_z{zeta:.3f}' in k]

    # If we have already stored this parameter set check if the SNR is better
    # "Better" here is defined as closest to the target SNR.
    if current_keys:
        current_snrs = current_keys[0][-1]
        current_theta = current_keys[0][0]
        if np.abs(np.array([snr, current_snrs]) - target_snr).argmin() == 0:
        # if np.argmax([snr, current_snrs]) == 0:
            # Remove the old catalog and replace with the better option
            cat_dict.remove(f't{current_theta:.3f}_e{eta:.3f}_z{zeta:.3f}_snr{current_snrs:.4f}')
            cat_dict.update({f't{theta:.3f}_e{eta:.3f}_z{zeta:.3f}_snr{snr:.4f}'})

            # Also update the running list of files
            catalogs_to_test.remove([old_file for old_file in catalogs_to_test if f'e{eta:.2f}_z{zeta:.2f}' in old_file][0])
            catalogs_to_test.update({catalog_file})
    else:
        cat_dict.update({f't{theta:.3f}_e{eta:.3f}_z{zeta:.3f}_snr{snr:.4f}'})
        catalogs_to_test.update({catalog_file})

tez_snr = np.array([np.array(theta_eta_zeta_snr_pattern.findall(k), dtype=float) for k in cat_dict])
tez_snr = tez_snr[tez_snr[:, 2].argsort()]
tez_snr = tez_snr[tez_snr[:, 1].argsort(kind='mergesort')]
t = Table(tez_snr, names=['theta', 'eta', 'zeta', 'SNR'])

try:
    assert tez_snr.shape[0] == 30

    for filename in catalogs_to_test:
        shutil.copy(filename, filename.replace(main_dir, new_dir))
except AssertionError:
    print(f'Missing catalogs in grid. {tez_snr.shape[0] = } should be 30.')
    print(f'Using a targeted SNR: {target_snr} +- {snr_tol}')
    print(tez_snr)
