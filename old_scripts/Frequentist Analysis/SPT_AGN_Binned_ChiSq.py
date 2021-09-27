"""
SPT_AGN_Binned_ChiSq.py
Author: Benjamin Floyd

Computes a chi-squared fit of the binned values with the constant field level.
"""

import numpy as np
from scipy.stats import chi2


def rchisq(ydata, ymod, deg=2, sd=None, both=False):
    # Chi-square statistic
    if sd is None:
        chisq = np.sum((ydata - ymod) ** 2)
    else:
        chisq = np.sum(((ydata - ymod) / sd) ** 2)

        # Number of degrees of freedom assuming 2 free parameters
    nu = ydata.size - 1 - deg

    if both:
        return chisq, chisq / nu

    return chisq / nu


# Read in the radial and mass data files
radial = np.load('Data/Binned_Analysis/Radial_bin_data_cumulative.npy', encoding='latin1').item()
mass = np.load('Data/Binned_Analysis/Mass_1.5r500_bin_data.npy', encoding='latin1').item()

# Extract the radial values
rad_bin_cent = radial['radial_bins']
rad_z05_065_surf_den = radial['mid_low_z_rad_surf_den']
rad_z05_065_err = radial['mid_low_z_rad_err']
rad_z065_075_surf_den = radial['mid_mid_z_rad_surf_den']
rad_z065_075_err = radial['mid_mid_z_rad_err']
rad_z075_1_surf_den = radial['mid_high_z_rad_surf_den']
rad_z075_1_err = radial['mid_high_z_rad_err']
rad_z1_surf_den = radial['high_z_rad_surf_den']
rad_z1_err = radial['high_z_rad_err']
rad_allz_surf_den = radial['all_z_rad_surf_den']
rad_allz_err = radial['all_z_rad_err']

# Extract the mass values within 1.5r500
mass_bin_cent = mass['mass_bin_cent']
mass_z05_065_surf_den = np.array(mass['mid_low_z_mass_surf_den'])
mass_z05_065_err = np.array(mass['mid_low_z_mass_surf_den_err'])
mass_z065_075_surf_den = np.array(mass['mid_mid_z_mass_surf_den'])
mass_z065_075_err = np.array(mass['mid_mid_z_mass_surf_den_err'])
mass_z075_1_surf_den = np.array(mass['mid_high_z_mass_surf_den'])
mass_z075_1_err = np.array(mass['mid_high_z_mass_surf_den_err'])
mass_z1_surf_den = np.array(mass['high_z_mass_surf_den'])
mass_z1_err = np.array(mass['high_z_mass_surf_den_err'])
mass_allz_surf_den = np.array(mass['all_z_mass_surf_den'])
mass_allz_err = np.array(mass['all_z_mass_surf_den_err'])

# The field surface density is 0 / Mpc^2 for both the radial and mass bins
rad_field = np.zeros(len(rad_bin_cent))
mass_field = np.zeros(len(mass_bin_cent))

# Compute reduced chi-squared statistic for all radial values
rad_z05_065_chisq, rad_z05_065_rchisq = rchisq(rad_z05_065_surf_den, rad_field, deg=1, sd=rad_z05_065_err, both=True)
rad_z065_075_chisq, rad_z065_075_rchisq = rchisq(rad_z065_075_surf_den, rad_field, deg=1, sd=rad_z065_075_err, both=True)
rad_z075_1_chisq, rad_z075_1_rchisq = rchisq(rad_z075_1_surf_den, rad_field, deg=1, sd=rad_z075_1_err, both=True)
rad_z1_chisq, rad_z1_rchisq = rchisq(rad_z1_surf_den, rad_field, deg=1, sd=rad_z1_err, both=True)
rad_allz_chisq, rad_allz_rchisq = rchisq(rad_allz_surf_den, rad_field, deg=1, sd=rad_allz_err, both=True)

# Compute the probability to exceed of these chi-squared values
rad_z05_065_pte = chi2.sf(rad_z05_065_chisq, 1)
rad_z065_075_pte = chi2.sf(rad_z065_075_chisq, 1)
rad_z075_1_pte = chi2.sf(rad_z075_1_chisq, 1)
rad_z1_pte = chi2.sf(rad_z1_chisq, 1)
rad_allz_pte = chi2.sf(rad_allz_chisq, 1)

print("""Radial Chi-Sq:
0.5 < z < 0.65: {z05:.2f},\t {z05_pte:.2e}
0.65 < z < 0.75: {z065:.2f},\t {z065_pte:.2e}
0.75 < z < 1.0: {z075:.2f},\t {z075_pte:.2e}
z > 1: {z1:.2f},\t {z1_pte:.2e}
all z: {zall:.2f},\t {zall_pte:.2e}""".format(z05=rad_z05_065_rchisq, z05_pte=rad_z05_065_pte,
                                              z065=rad_z065_075_rchisq, z065_pte=rad_z065_075_pte,
                                              z075=rad_z075_1_rchisq, z075_pte=rad_z075_1_pte,
                                              z1=rad_z1_rchisq, z1_pte=rad_z1_pte,
                                              zall=rad_allz_rchisq, zall_pte=rad_allz_pte))

np.save('Data/Binned_Analysis/Radial_bin_chisq_pte', np.array([[rad_z05_065_rchisq, rad_z05_065_pte],
                                                               [rad_z065_075_rchisq, rad_z065_075_pte],
                                                               [rad_z075_1_rchisq, rad_z075_1_pte],
                                                               [rad_z1_rchisq, rad_z1_pte],
                                                               [rad_allz_rchisq, rad_allz_pte]]))

# Compute reduced chi-squared statistic for all mass values
mass_z05_065_chisq, mass_z05_065_rchisq = rchisq(mass_z05_065_surf_den, mass_field, deg=1, sd=mass_z05_065_err, both=True)
mass_z065_075_chisq, mass_z065_075_rchisq = rchisq(mass_z065_075_surf_den, mass_field, deg=1, sd=mass_z065_075_err, both=True)
mass_z075_1_chisq, mass_z075_1_rchisq = rchisq(mass_z075_1_surf_den, mass_field, deg=1, sd=mass_z075_1_err, both=True)
mass_z1_chisq, mass_z1_rchisq = rchisq(mass_z1_surf_den, mass_field, deg=1, sd=mass_z1_err, both=True)
mass_allz_chisq, mass_allz_rchisq = rchisq(mass_allz_surf_den, mass_field, deg=1, sd=mass_allz_err, both=True)

# Compute the probability to exceed of these chi-squared values
mass_z05_065_pte = chi2.sf(mass_z05_065_chisq, 1)
mass_z065_075_pte = chi2.sf(mass_z065_075_chisq, 1)
mass_z075_1_pte = chi2.sf(mass_z075_1_chisq, 1)
mass_z1_pte = chi2.sf(mass_z1_chisq, 1)
mass_allz_pte = chi2.sf(mass_allz_chisq, 1)

np.save('Data/Binned_Analysis/Mass_1.5r500_bin_chisq_pte', np.array([[mass_z05_065_rchisq, mass_z05_065_pte],
                                                                     [mass_z065_075_rchisq, mass_z065_075_pte],
                                                                     [mass_z075_1_rchisq, mass_z075_1_pte],
                                                                     [mass_z1_rchisq, mass_z1_pte],
                                                                     [mass_allz_rchisq, mass_allz_pte]]))

print("""Mass Chi-Sq:
0.5 < z < 0.65: {z05:.2f},\t {z05_pte:.2e}
0.65 < z < 0.75: {z065:.2f},\t {z065_pte:.2e}
0.75 < z < 1.0: {z075:.2f},\t {z075_pte:.2e}
z > 1: {z1:.2f},\t {z1_pte:.2e}
all z: {zall:.2f},\t {zall_pte:.2e}""".format(z05=mass_z05_065_rchisq, z05_pte=mass_z05_065_pte,
                                              z065=mass_z065_075_rchisq, z065_pte=mass_z065_075_pte,
                                              z075=mass_z075_1_rchisq, z075_pte=mass_z075_1_pte,
                                              z1=mass_z1_rchisq, z1_pte=mass_z1_pte,
                                              zall=mass_allz_rchisq, zall_pte=mass_allz_pte))
