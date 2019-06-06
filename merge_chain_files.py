#!/usr/bin/env python
"""
merge_chain_files.py
Author: Benjamin Floyd

Combines chain files of multiple runs for a particular test suite into a single file.
"""

import argparse
import glob
import h5py
from subprocess import Popen, PIPE


def visitor_func(name, node):
    if isinstance(node, h5py.Dataset):
        print('{} is a dataset'.format(name))
    else:
        print('{} is a group'.format(name))


parser = argparse.ArgumentParser(description='Combines multiple emcee chain files in hdf5 format into a single file.')
parser.add_argument('-i', '--input', dest='input_files', help='File names of chain files to be combined', nargs='*')
parser.add_argument('-o', '--output', dest='output_file', help='Name of output file to be created')
parser.add_argument('--show-structure', help='Prints the structure of the output file after merging.', action='store_true')
args = parser.parse_args()

if not args.input_files:
    print('Input files not provided. Please list input files here.')
    args.input_files = input()
    if ',' in args.input_files:
        args.input_files = args.input_files.split(',')

if not args.output_file:
    print('Output file not specified. Please give a name for the output file.')
    args.output_file = input()


# Get the filenames of the original chain files
original_names = glob.glob(args.input_files) if isinstance(args.input_files, str) else args.input_files

# Read in the group names from each file
chain_names = {}
for filename in original_names:
    with h5py.File(filename, 'r') as f:
        chain_names[filename] = list(f.keys())

# Combine the files together
for filename in original_names:
    for chain_name in chain_names[filename]:
        copycmd = Popen(['h5copy', '-i', filename, '-o', args.output_file, '-s', chain_name, '-d', chain_name],
                        stdout=PIPE, stderr=PIPE)
        out, err = copycmd.communicate()
        if out or err:
            print(out)
            print(err)

# Validate the new file has the correct structure
if args.show_structure:
    print('Structure of output file {}:'.format(args.output_file))
    with h5py.File(args.output_file, 'r') as f:
        f.visititems(visitor_func)
