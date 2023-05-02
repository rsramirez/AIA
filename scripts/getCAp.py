#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "R. Sanchez-Ramirez"
__created__ = "2023-05-02 17:34:18"
__updated__ = "2023-05-02 17:34:32"

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from astropy.table import Table
from glob import glob

parser = argparse.ArgumentParser(
    description='Procesa los resultados de apphot (IRAF).')

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('-f', '--filename',
                    help='Nombre del archivo .cog de getRAp.py a procesar.')

group.add_argument('-d', '--dirname',
                    help='Nombre del directorio con los .cog a promediar.')

parser.add_argument('-r', '--rap',
                    help='Radio de apertura seleccionado.',
                    required=True)

parser.add_argument('--o', help=
                    'Sobreescribir la tabla generada anteriormente.',
                    action='store_true')

args = parser.parse_args()
args = vars(args)

linestyles = [
    ('solid', 'solid'),  # Same as (0, ()) or '-'
    ('dotted', 'dotted'),  # Same as (0, (1, 1)) or ':'
    ('dashed', 'dashed'),  # Same as '--'
    ('dashdot', 'dashdot'),  # Same as '-.'
    ('loosely dotted', (0, (1, 10))),
    ('dotted', (0, (1, 1))),
    ('densely dotted', (0, (1, 1))),
    ('long dash with offset', (5, (10, 3))),
    ('loosely dashed', (0, (5, 10))),
    ('dashed', (0, (5, 5))),
    ('densely dashed', (0, (5, 1))),

    ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('densely dashdotted', (0, (3, 1, 1, 1))),

    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def openFile(filename):

    try:
        table = Table.read(filename, format='ascii.ecsv')
    except Exception:
        print('ERROR: El archivo no es un .cog generado por getRAp.py')
        sys.exit()

    return table


def getCAp(table):

    table = table[table['RAp'] <= float(args['rap'])]
    row = table[table['RAp'] == float(args['rap'])]
    # print(row)

    mi = row['iMag'][0]
    emi = row['eiMag'][0]

    # print('mi = %.4f +- %.4f' % (mi, emi))

    table.meta['iMag'] = mi
    table.meta['eiMag'] = emi

    table['DMag'] = mi - table['iMag']
    table['eDMag'] = np.sqrt(emi ** 2 + table['eiMag'] ** 2)

    bad = np.where(table['DMag'] >= 0.)
    table['DMag'][bad] = 0.0
    table['eDMag'][bad] = 0.0
    table = table['RAp', 'DMag', 'eDMag']
    filt = table.meta['filter']
    table.meta = {}
    table.meta['RAp'] = args['rap']
    table.meta['filter'] = filt

    # table.pprint()

    return table


def main(name, i=0):
    print('Procesando el fichero %s...' % name)

    table = openFile(name)
    table = getCAp(table)

    ax.errorbar(table['RAp'].data, table['DMag'].data, yerr=table['eDMag'].data,
                label=os.path.splitext(name)[0],
                ls=linestyles[i][1], ms=5, lw=1, c='Gray')

    name = os.path.splitext(name)[0] + '.cap'
    table.write(name, format='ascii.ecsv', overwrite=args['o'])

    print('\t Fichero de CAp creado (%s).' % name)

    return table


fig, ax = plt.subplots()

if args['filename']:

    main(args['filename'])

    ax.set_xlabel('$R_{Ap}$ [pix]')
    ax.set_ylabel('$\Delta$m')

    ax.legend(loc=4)

    name = '%s.cap.png' % os.path.splitext(args['filename'])[0]
    fig.savefig(name)

else:
    print('Procesando el directorio %s...' % args['dirname'])

    files = glob(os.path.join(args['dirname'], '*.cog'))
    DMags = []
    eDMags = []
    for i, cog in enumerate(files):
        table = main(cog, i)

        DMags.append(table['DMag'].data)
        eDMags.append(table['eDMag'].data)

    print('Promediando ficheros de CAp...')

    DMag = np.mean(DMags, axis=0)
    eDMag = np.sqrt(np.sum(np.array(eDMags) ** 2, axis=0))

    table['DMag'] = DMag
    table['eDMag'] = eDMag

    ax.errorbar(table['RAp'].data, table['DMag'].data, yerr=table['eDMag'].data,
                label='Mean',
                ls=linestyles[0][1], ms=5, lw=2, c='Red')

    name = '%s.cap' % table.meta['filter'].strip()
    table.write(name, format='ascii.ecsv', overwrite=args['o'])

    print('\t Fichero de CAp creado (%s).' % name)

    ax.set_xlabel('$R_{Ap}$ [pix]')
    ax.set_ylabel('$\Delta$m')

    ax.legend(loc=4)
    name = 'CAp-%s.png' % table.meta['filter'].strip()
    fig.savefig(name)
