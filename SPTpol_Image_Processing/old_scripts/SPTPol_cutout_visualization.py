"""
SPTPol_cutout_visualization.py
Author: Benjamin Floyd

Automates opening the SPTPol 100d cluster cutouts in DS9 for visual inspections.
"""

import glob
import re
from itertools import groupby
from subprocess import Popen

from astropy.table import Table


def cluster_keys(name):
    return cluster_id_pattern.search(name).group(0)


def flag_input(prompt):
    res = input(prompt) or '0'
    try:
        res = eval(res)
        return res
    except AttributeError:
        return res


ds9_exec = '/Applications/miniconda/envs/astro3/bin/ds9'

# Pattern for cluster ids
cluster_id_pattern = re.compile(r'SPT-CLJ\d+-\d+')

# Get a list of all the cutout files
cutout_files = glob.glob('Data/SPTPol/images/cluster_cutouts_trimmed/*.fits')

# Sort the list using the cluster ids
cutout_files = sorted(cutout_files, key=cluster_keys)

# Group the files by cluster
cutout_dict = {k: list(g) for k, g in groupby(cutout_files, key=cluster_keys)}

cluster_names, flag = [], []
for i, (cluster_id, images) in enumerate(cutout_dict.items()):
    print(f'Displaying images for {cluster_id} [{i + 1}/{len(cutout_dict)}]')

    # Open DS9 and display the images
    ds9_proc = Popen([ds9_exec, '-invert', '-zscale',
                      images[0], '-frame', 'new', images[1],
                      '-frame', 'new', images[2], '-frame', 'new', images[3],
                      '-frame', 'lock', 'wcs',
                      '-frame', 'first',
                      '-zoom', 'to', 'fit'])

    cluster_names.append(cluster_id)
    flag.append(flag_input('Flag: [0] '))

    ds9_proc.terminate()

cluster_flags = Table([cluster_names, flag], names=['SPT_ID', 'Flag'])
cluster_flags.write('Data/SPTPol/spt100d_mask_notes.cat', format='ascii')
