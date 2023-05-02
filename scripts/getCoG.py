#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "R. Sanchez-Ramirez"
__created__ = "2023-05-02 17:32:19"
__updated__ = "2023-05-02 17:32:36"

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from astropy.table import Table
from glob import glob

parser = argparse.ArgumentParser(
    description='Procesa los resultados de apphot (IRAF).')

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('-m', '--magfile',
                    help='Nombre del archivo .mag de phot.')

group.add_argument('-d', '--dirname',
                    help='Nombre del directorio con los .mag a procesar.')

parser.add_argument('-g', '--gain', type=float,
                    help='Ganancia de la CCD.',
                    required=True)

parser.add_argument('--o', help=
                    'Sobreescribir la tabla generada anteriormente.',
                    action='store_true')

args = parser.parse_args()
args = vars(args)


def openMagFile(mfile):

    with open(mfile) as mf:
        raw = mf.readlines()
        header = ''.join(raw[:79])
        # print(header)

        pars = raw[75:79]
        # print(pars)

        dpars = {}
        dpars['airmass'] = float(pars[-1].split()[1])
        dpars['filter'] = pars[-1].split()[2].split('_')[0].upper()
        dpars['skydev'] = float(pars[-2].split()[1])
        dpars['nsky'] = int(pars[-2].split()[3])
        dpars['x'] = float(pars[-3].split()[0])
        dpars['y'] = float(pars[-3].split()[1])
        dpars['ex'] = float(pars[-3].split()[4])
        dpars['ey'] = float(pars[-3].split()[5])
        # print(dpars)

        data = ''.join(raw[79:])
        # print(data)

        table = Table.read(data, format='ascii.fixed_width_no_header',
                           col_starts=(0, 13, 26, 37, 51, 58, 64),
                           col_ends=(10, 22, 36, 47, 57, 63, 81),)
        table = getMagTable(table, dpars)

        return table


def getMagTable(table, dpars):

    area = table['col3'].data.astype(float)
    table.remove_columns(['col2', 'col7'])

    table.rename_column('col1', 'RAp')
    table.rename_column('col3', 'Area')
    table.rename_column('col4', 'Flux')
    table.rename_column('col5', 'iMag')
    table.rename_column('col6', 'eiMag')

    table['RAp'].dtype = np.float
    table['Area'].dtype = np.float
    table['Flux'].dtype = np.float
    table['iMag'].dtype = np.float
    table['eiMag'].dtype = np.float

    eflux = np.sqrt(table['Flux'].data / args['gain'] + area *
                    dpars['skydev'] ** 2 + (area * dpars['skydev']) ** 2 /
                    dpars['nsky'])
    snr = table['Flux'].data / eflux

    table['eiMag'] = 1.0857 * eflux / table['Flux'].data
    table.add_column(eflux, name='eFlux', index=3)
    table.add_column(snr, name='SNR', index=4)

    table.meta.update(dpars)

    return table


def getRAp(table):

    rap = table['RAp'].data
    area = table['Area'].data
    flux = table['Flux'].data
    snr = table['SNR'].data
    skydev = table.meta['skydev']
    nsky = table.meta['nsky']

    nrel = len(flux) - 1
    erel = []
    frel = False
    fdsnr = False
    for j in range(len(flux) - 1):
        err = (100. * np.sqrt((flux[j + 1] - flux[j]) / float(args['gain']) +
                              (area[j + 1] - area[j]) * skydev ** 2 +
                              ((area[j + 1] - area[j]) * skydev) ** 2 / nsky) /
                              (flux[j + 1] - flux[j]))
        erel.append(err)

        if err > 10. and not frel:
            nrel = j - 1
            frel = True

        disc = (snr[j + 1] / snr[j] - 1.) * 100.

        if disc < 0. and not fdsnr:
            ndsnr = j
            fdsnr = True

    erel.append(0.)

    table['eRel'] = erel
    table.meta['oRAp'] = rap[nrel]
    table.meta['sRAp'] = rap[ndsnr]

    return table


def plotCoG(table):

    fig, ax = plt.subplots()
    ax.plot(table['RAp'].data, table['Flux'].data, color='Red', marker='o')

    ax.set_xlabel('$R_{Ap}$ [pix]')
    ax.set_ylabel('Flux [ADU]')

    ax2 = ax.twinx()
    ax2.plot(table['RAp'].data, table['SNR'].data, color='Blue')
    ax2.set_ylabel('SNR')

    fig.axes[1].vlines(x=[table.meta['sRAp'], ], colors=['Black', ],
                       linestyle='--', ymin=0.25 * table['SNR'].data.max(),
                       ymax=table['SNR'].data.max(),
                       label='SNR máxima (%d pix)'
                       % (table.meta['sRAp']))

    fig.axes[1].vlines(x=[table.meta['oRAp'], ], colors=['Black', ],
                       ymin=0.25 * table['SNR'].data.max(),
                       ymax=table['SNR'].data.max(),
                       label='Error relativo $<$ 0.1 (%d pix)'
                       % (table.meta['oRAp']))

    fig.axes[1].legend(loc=4)

    return fig


def main(mag):

    print('Procesando el fichero %s...' % mag)

    table = openMagFile(mag)
    table = getRAp(table)

    name = os.path.splitext(mag)[0]
    num = os.path.splitext(os.path.splitext(mag)[1])[0]

    name += num + '.cog'
    table.write(name, format='ascii.ecsv', overwrite=args['o'])

    print('\t Fichero de CoG creado (%s).' % name)

    fig = plotCoG(table)
    fig.tight_layout()

    name = os.path.splitext(mag)[0] + num + '.png'
    fig.savefig(name)
    plt.close('all')

    print('\t Gráfico de CoG creado (%s).' % name)


if args['magfile']:
    main(args['magfile'])

else:
    print('Procesando el directorio %s...' % args['dirname'])

    files = glob(os.path.join(args['dirname'], '*.mag.????'))

    for mag in files:
        main(mag)
