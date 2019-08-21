"""
Catalog_fixer.py
Author: Benjamin Floyd
Some of the SExtractor catalogs have a tab character which interferes with astropy reading the tables in correctly.
"""
import glob
from itertools import takewhile, dropwhile

from astropy.table import Table


def get_index(x, y):
    for xx in x:
        if xx.startswith(y):
            return x.index(xx)


# orig_cats = glob.glob('/Users/btfkwd/Documents/SPT_AGN/Data/Catalogs.old/*.cat')
orig_cats = glob.glob('/Users/btfkwd/Documents/SPT_AGN/Data/SPTPol/catalogs/SSDF2.20130918.v9.private_old.cat')
for files in orig_cats:
    print("Opening: " + files)
    with open(files, 'r') as sexcat:
        # Read all the lines in.
        lines = sexcat.readlines()

        header = [ln for ln in takewhile(lambda x: x.startswith('#'), lines)]
        data = [ln.replace('\t', ' ') for ln in dropwhile(lambda x: x.startswith('#'), lines)]

        # Find the starting points of the 3.6 um photometery columns
        ch1_cols = get_index(header, '#  22')
        ch2_cols = get_index(header, '#  34')
        # Modify the column names for the 3.6 um magnitudes and fluxes
        for i, ch1_lines in enumerate(header[ch1_cols:ch1_cols+12]):
            ch1_line = ch1_lines.split(maxsplit=3)

            # Prepend the column names with 'I1'
            ch1_line[2] = 'I1_' + ch1_line[2]

            # For the aperture measurements append the aperture size (in arcseconds) to the column name
            if ch1_line[3].startswith('4'):
                ch1_line[2] += '4'
            elif ch1_line[3].startswith('6'):
                ch1_line[2] += '6'

            # Join the modified line together and replace the existing line with our new version
            new_ch1_line = '{commchar}  {colnum} {colname}\t{comment}'.format(commchar=ch1_line[0],
                                                                              colnum=ch1_line[1],
                                                                              colname=ch1_line[2],
                                                                              comment=ch1_line[3])
            header[ch1_cols + i] = new_ch1_line

        # Modify the column names for the 4.5 um magnitudes and fluxes
        for i, ch2_lines in enumerate(header[ch2_cols:ch2_cols+12]):
            ch2_line = ch2_lines.split(maxsplit=3)

            # Prepend the column names with 'I2'
            ch2_line[2] = 'I2_' + ch2_line[2]

            # For the aperture measurements append the aperture size (in arcseconds) to the column name
            if ch2_line[3].startswith('4'):
                ch2_line[2] += '4'
            elif ch2_line[3].startswith('6'):
                ch2_line[2] += '6'

            # Join the modified line together and replace the existing line with our new version
            new_ch2_line = '{commchar}  {colnum} {colname}\t{comment}'.format(commchar=ch2_line[0],
                                                                              colnum=ch2_line[1],
                                                                              colname=ch2_line[2],
                                                                              comment=ch2_line[3])
            header[ch2_cols + i] = new_ch2_line

    # Write the corrected file out.
    new_file = files.replace('old', 'new')
    print("Writing: " + new_file)
    with open(new_file, 'w') as outcat:
        outcat.writelines(header)
        outcat.writelines(data)

# for catalog in glob.glob('/Users/btfkwd/Documents/SPT_AGN/Data/Catalogs.new/*.cat'):
for catalog in glob.glob('/Users/btfkwd/Documents/SPT_AGN/Data/SPTPol/catalogs/SSDF2.20130918.v9.private_new.cat'):
    try:
        tempcat = Table.read(catalog, format='ascii.sextractor')
    except:
        print('Error in {}'.format(catalog))