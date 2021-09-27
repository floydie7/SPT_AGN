"""
SSDF_background_noise.py
Author: Benjamin Floyd

Reads in background estimate files from SPOT to process.
"""

import re
import numpy as np
import matplotlib.pyplot as plt


def file_buffer(f):
    buffer = []
    for ln in f:
        if 'MOVING_TARGET' in ln:
            if buffer:
                yield buffer
            buffer = [ln]
        else:
            buffer.append(ln)
    yield buffer


def parse_backgrounds(filename):
    with open(filename, 'r') as f:
        target_info = list(file_buffer(f))

    target_backgrounds = {}
    for block in target_info[1:]:
        for ln in block:
            name_line = re.match(r'\s+ TARGET_NAME:\s+ (\S+)\n', ln)
            if name_line:
                name = name_line.group(1)

            background_line = re.search(r'TOTAL=(\d+.\d+)\n', ln)
            if background_line:
                background = float(background_line.group(1))

        target_backgrounds[name] = background

    return target_backgrounds


# Read the file
SSDF_jul11_file = 'Data/Background_Noises/SSDF_backgrounds_july2011'
SSDF_jan12_file = 'Data/Background_Noises/SSDF_backgrounds_jan2012'
SPT_cycle6_file = 'Data/Background_Noises/SPTcycle6_backgrounds'
SPT_cycle7_file = 'Data/Background_Noises/SPTcycle7_backgrounds'
SPT_cycle8_file = 'Data/Background_Noises/SPTcycle8_backgrounds'
SPT_cycle10_file = 'Data/Background_Noises/SPTcycle10_backgrounds'
SPT_cycle11_file = 'Data/Background_Noises/SPTcycle11_backgrounds'
SPT_cycle12_file = 'Data/Background_Noises/SPTcycle12_backgrounds'

SSDF_jul11_target_bkgs = parse_backgrounds(SSDF_jul11_file)
SSDF_jan12_target_bkgs = parse_backgrounds(SSDF_jan12_file)
SPT_cycle6_target_bkgs = parse_backgrounds(SPT_cycle6_file)
SPT_cycle7_target_bkgs = parse_backgrounds(SPT_cycle7_file)
SPT_cycle8_target_bkgs = parse_backgrounds(SPT_cycle8_file)
SPT_cycle10_target_bkgs = parse_backgrounds(SPT_cycle10_file)
SPT_cycle11_target_bkgs = parse_backgrounds(SPT_cycle11_file)
SPT_cycle12_target_bkgs = parse_backgrounds(SPT_cycle12_file)

SSDF_jul11_bkg_values = list(SSDF_jul11_target_bkgs.values())
SSDF_jan12_bkg_values = list(SSDF_jan12_target_bkgs.values())
SPT_cycle6_bkg_values = list(SPT_cycle6_target_bkgs.values())
SPT_cycle7_bkg_values = list(SPT_cycle7_target_bkgs.values())
SPT_cycle8_bkg_values = list(SPT_cycle8_target_bkgs.values())
SPT_cycle10_bkg_values = list(SPT_cycle10_target_bkgs.values())
SPT_cycle11_bkg_values = list(SPT_cycle11_target_bkgs.values())
SPT_cycle12_bkg_values = list(SPT_cycle12_target_bkgs.values())

SPT_all_cycles_bkg_values = [*SPT_cycle6_bkg_values, *SPT_cycle7_bkg_values, *SPT_cycle8_bkg_values,
                             *SPT_cycle10_bkg_values, *SPT_cycle11_bkg_values, *SPT_cycle12_bkg_values]

bins = np.arange(0.14, 0.32, 0.005)
alpha = 0.6
fig, ax = plt.subplots()
# ax.hist(SSDF_jul11_bkg_values, bins=bins, alpha=alpha+0.4, label='SSDF (Jul 2011)')
# ax.hist(SSDF_jan12_bkg_values, bins=bins, alpha=alpha+0.4, label='SSDF (Jan 2012)')

# ax.hist(SPT_all_cycles_bkg_values, bins=bins, alpha=alpha, label='SPT cycles 6-8, 10-12')

ax.hist(SPT_cycle6_bkg_values, bins=bins, alpha=alpha, label='SPT cycle 6')
ax.hist(SPT_cycle7_bkg_values, bins=bins, alpha=alpha, label='SPT cycle 7')
ax.hist(SPT_cycle8_bkg_values, bins=bins, alpha=alpha, label='SPT cycle 8')
ax.hist(SPT_cycle10_bkg_values, bins=bins, alpha=alpha, label='SPT cycle 10')
ax.hist(SPT_cycle11_bkg_values, bins=bins, alpha=alpha, label='SPT cycle 11')
ax.hist(SPT_cycle12_bkg_values, bins=bins, alpha=alpha, label='SPT cycle 12')

ax.set(xlabel=r'Background [MJy sr$^{-1}$]', ylabel='Observation Targets', xlim=[0.14, 0.32])
ax.legend()
# fig.savefig('Data/Background_Noises/Plots/SSDF_SPT_cycle6-8_10-12_merged_hists.pdf', format='pdf')
plt.show()

# print('cycle 6: {}'.format(len(SPT_cycle6_bkg_values)))
# print('cycle 7: {}'.format(len(SPT_cycle7_bkg_values)))
# print('cycle 8: {}'.format(len(SPT_cycle8_bkg_values)))
