#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "R. Sanchez-Ramirez"
__created__ = "2023-05-02 17:38:13"
__updated__ = "2023-05-02 18:08:34"

import argparse
import dynesty
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from astropy.table import Table, hstack, join, unique, vstack
from chainconsumer import ChainConsumer
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from glob import glob
from scipy.special import ndtri
from scipy.stats import norm

parser = argparse.ArgumentParser(
    description='Calcula las Ecs. de Transformación.\n\n' +
    'El script se puede ejecutar con 3 argumentos:\n' +
    '$ python3 getET.py [arg] [opciones]\n' +
    'fake: Simula una observación.\n' +
    'prep: Prepara los ficheros y evalúa los datos antes del ajuste.\n' +
    'fit: Ajusta los datos a la ec. Mcat = Mins - kX + a + bC \n' +
    'Para obtener ayuda de cada uno de los argumentos:\n' +
    '$ python3 getET.py [arg] help',
    formatter_class=argparse.RawTextHelpFormatter
    )

subparser = parser.add_subparsers(dest='command')
fake = subparser.add_parser('fake', description='Work in progress...')
prep = subparser.add_parser('prep', description='')
fit = subparser.add_parser('fit', description='')

# fake
fake.add_argument('-k',
                    help='Extincion atmosférica.',
                    type=float, required=True)
fake.add_argument('-a',
                    help='Punto cero.',
                    type=float, required=True)
fake.add_argument('-b',
                    help='Término de color.',
                    type=float, required=True)
fake.add_argument('-n',
                    help='Número de observaciones.',
                    type=int, required=True)

# prep
prep.add_argument('-d', '--dirname',
                    help='Nombre del directorio con los .cog a usar.',
                    required=True)
prep.add_argument('-r', '--rap',
                    help='Radio de apertura seleccionado.',
                    type=float, required=True)
prep.add_argument('-l2', '--lambda2',
                    help='Filtro para calcular el color.',)
prep.add_argument('-k',
                    help='Extincion atmosférica aproximada.',
                    type=float, required=True)
prep.add_argument('-b',
                    help='Término de color aproximado.',
                    type=float)
prep.add_argument('-e', '--exclude', nargs='+',
                  help='Lista de Id (separadas por espacios) a eliminar',
                  type=str)
prep.add_argument('--o', help='Sobreescribir la tabla generada anteriormente.',
                    action='store_true')

# fit
fit.add_argument('-l1', '--lambda1',
                    help='Nombre del fichero .et a usar (lambda 1).',
                    required=True)

fit.add_argument('-l2', '--lambda2',
                    help='Nombre del filtro secundario a usar (lambda 2). ' +
                    'Si se introduce un fichero .et se usará en el cálculo ' +
                    'del color (TBI).', required=True)

fit.add_argument('-k',
                    help='Extinción estimada/publicada.',
                    type=float, required=True)
fit.add_argument('-ek',
                    help='Error estimado/publicado de la extinción.',
                    type=float, required=True)
fit.add_argument('-a',
                    help='Punto cero estimado/publicado.',
                    type=float, required=True)
fit.add_argument('-ea',
                    help='Error estimado/publicado del punto cero.',
                    type=float, required=True)
fit.add_argument('-b',
                    help='Término de color estimado/publicado.',
                    type=float, required=True)
fit.add_argument('-eb',
                    help='Error estimado/publicado del término de color.',
                    type=float, required=True)
fit.add_argument('--o', help=
                    'Sobreescribir la tabla generada anteriormente.',
                    action='store_true')

args = parser.parse_args()
args = vars(args)


def openFile(filename):

    try:
        table = Table.read(filename, format='ascii.ecsv')
    except Exception:
        print('ERROR: Formato de archivo no válido')
        sys.exit()

    return table


def getList():

    print('Procesando el directorio %s...' % args['dirname'])

    files = glob(os.path.join(args['dirname'], '*.cog'))
    if not len(files):
        print('No hay ningún archivo .cog en %s' % args['dirname'])
        sys.exit()

    Id = []
    x = []
    y = []
    X = []
    Mi = []
    eMi = []
    Mcat = []
    eMcat = []
    for cog in files:
        table = openFile(cog)
        row = table[table['RAp'] == args['rap']]

        tstds = '%s.fit' % cog.split('_')[0].split('/')[1]
        stds = Table.read(tstds, format='fits')

        field = cog.split('-')[0].split('/')[1]
        star = os.path.splitext(os.path.splitext(cog)[0])[1][1:]
        band = table.meta['filter']

        id = '%s-%s' % (field, star)

        cat = stds[stds['Id'] == id]

        Id.append(id)
        x.append(table.meta['x'])
        y.append(table.meta['y'])
        X.append(table.meta['airmass'])
        Mi.append(row['iMag'][0])
        eMi.append(row['eiMag'][0])
        Mcat.append(cat['%s' % band][0])
        eMcat.append(cat['e%s' % band][0])

    table = Table()
    table['Id'] = Id
    table['x'] = x
    table['y'] = y
    table['X'] = X
    table['Mi'] = Mi
    table['eMi'] = eMi
    table['Mcat'] = Mcat
    table['eMcat'] = eMcat

    table.meta['filter'] = band
    table.meta['RAp'] = args['rap']

    table.sort('X')

    name = '%s.et' % band

    testData(table)

    if args['exclude']:
        mask = np.array([i for i, el in enumerate(table['Id'])
                         if el not in args['exclude']])
        table = table[mask]

    table.write(name, format='ascii.ecsv', overwrite=True)

    print('\n Tabla final:')
    print(table)
    print('\nArchivo %s.et generado.' % band)


def testData(table1):
    print('Analizando los datos...')

    band1 = table1.meta['filter']

    table2 = Table.read('campos.fit', format='fits')

    band2 = args['lambda2']

    table = join(table1, table2, join_type='inner', keys='Id',
                 metadata_conflicts='silent')

    table.sort('X')

    if args['b']:
        inter = table['Id', 'X', 'Mi', band1, band2]
        inter['Mo'] = np.round(inter['Mi'] - args['k'] * inter['X'], 3)
        inter['C'] = np.round(inter[band1] - inter[band2], 3)
        inter['a'] = np.round(inter[band1] - inter['Mo'] -
                              args['b'] * inter['C'], 1)
    else:
        inter = table['Id', 'X', 'Mi', band1]
        inter['Mo'] = np.round(inter['Mi'] - args['k'] * inter['X'], 3)
        inter['a'] = np.round(inter[band1] - inter['Mo'], 1)

    if args['exclude']:
        mask = np.array([i for i, el in enumerate(inter['Id'])
                         if el not in args['exclude']])
        excl = np.array([i for i, el in enumerate(inter['Id'])
                         if el in args['exclude']])
        excl = inter[excl]
        inter = inter[mask]

        print('\nFilas excluidas:')
        print(excl)

    print('\nFilas incluidas:')
    print(inter)


def getTransformation(My, eMy, X, C, eC, band1, band2, Id):

    def loglike(theta):

        k, a, b = theta  # unpack the parameters

        variance = (eC * b) ** 2 + eMy ** 2

        model = a + b * C - k * X
        residual = My - model

        return -0.5 * np.sum(residual ** 2 / variance + np.log(variance))

    def prior_transform(utheta):

        uk, ua, ub = utheta

        k = mk + sk * ndtri(uk)  # convert back to k
        a = ma + sa * ndtri(ua)  # convert back to a
        b = mb + sb * ndtri(ub)  # convert back to b

        return k, a, b

    mk = args['k']
    sk = args.get('ek', 0.5)
    ma = args['a']
    sa = args.get('ea', 1.0)
    mb = args['b']
    sb = args.get('eb', 0.5)

    dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=3)
    dsampler.run_nested()
    dres = dsampler.results

    # Extract sampling results.
    weights = np.exp(dres.logwt - dres.logz[-1])  # normalized weights
    samples = dyfunc.resample_equal(dres.samples, weights)

    print('Número de iteraciones: %d' % samples.shape[0])

    # output marginal likelihood
    print('La evidencia es %.4f ± %.4f' % (dres.logz[-1], dres.logzerr[-1]))

    # Compute 10%-90% quantiles.
    quantiles = [dyfunc.quantile(samps, [0.025, 0.975], weights=weights)
                 for samps in dres.samples.T]

    # print(quantiles)

    # Compute weighted mean and covariance.
    mean, cov = dyfunc.mean_and_cov(dres.samples, weights)

    labels = [r'$\kappa_{%s}$' % band1,
              r'$\alpha_{%s%s}$' % (band1, band2),
              r'$\beta_{%s%s}$' % (band1, band2)]

    # Plot a summary of the run.
    fig, axes = dyplot.runplot(dres, label_kwargs={'fontsize': 16.})
    fig.tight_layout()
    fig.savefig('et_run_%s_%s.png' % (band1, band2))

    fig, axes = dyplot.traceplot(dres, labels=labels, show_titles=True,
                                 title_fmt='.3f',
                                 label_kwargs={'fontsize': 16.}, truths=mean,)

    fig.tight_layout()
    fig.savefig('et_pars_%s_%s.png' % (band1, band2))

    k = mean[0]
    ek = np.mean(abs(quantiles[0] - k))

    a = mean[1]
    ea = np.mean(abs(quantiles[1] - a))

    b = mean[2]
    eb = np.mean(abs(quantiles[2] - b))

    I = np.linspace(X.min(), X.max(), 100)
    J = np.linspace(C.min(), C.max(), 100)

    Y = k * I
    Z = a + b * J - Y

    # Calculate range our uncertainty gives using 2D matrix multplication
    Ib = (samples[:, 0][:, None] * I)
    Yb = np.percentile(Ib, 100 * norm.cdf([-2, -1, 1, 2]), axis=0)

    Jb = (-samples[:, 0][:, None] * I +
          samples[:, 1][:, None] +
          samples[:, 2][:, None] * J)
    Zb = np.percentile(Jb, 100 * norm.cdf([-2, -1, 1, 2]), axis=0)

    fig, ax = plt.subplots(dpi=300.)
    ax.errorbar(X, My, yerr=eMy, fmt='.', label="Observaciones",
                ms=5, lw=1, c='Black')
    for i, txt in enumerate(Id):
        ax.annotate(txt, (X[i], My[i]))

    ax.plot(I, Z, label="Mejor ajuste", c='Red')
    ax.fill_between(I, Zb[0,:], Zb[-1,:],
                     label='2$\sigma$ (I.C.)', fc='Black', alpha=0.1)
    ax.fill_between(I, Zb[1,:], Zb[-2,:],
                     label='1$\sigma$ (I.C.)', fc='Orange', alpha=0.4)
    ax.legend(loc='best')

    ax.set_xlabel('Masa de aire')
    ax.set_ylabel('%s$_{cat}$ - %s$_{ins}$' % (band1, band1))

    fig.tight_layout()
    fig.savefig('et_fit_%s.png' % (band1,))

    fig, ax = plt.subplots(dpi=300.)
    ax.errorbar(C, My, xerr=eC, yerr=eMy, fmt='.', label="Observaciones",
                ms=5, lw=1, c='Black')
    for i, txt in enumerate(Id):
        ax.annotate(txt, (C[i], My[i]))

    ax.plot(J, Z, label="Mejor ajuste", c='Red')
    ax.fill_between(J, Zb[0,:], Zb[-1,:],
                     label='2$\sigma$ (I.C.)', fc='Black', alpha=0.1)
    ax.fill_between(J, Zb[1,:], Zb[-2,:],
                     label='1$\sigma$ (I.C.)', fc='Orange', alpha=0.4)
    ax.legend(loc='best')

    ax.set_xlabel('(%s - %s)' % (band1, band2))
    ax.set_ylabel('%s$_{cat}$ - %s$_{ins}$' % (band1, band1))

    fig.tight_layout()
    fig.savefig('et_fit_%s_%s.png' % (band1, band2))

    plt.close('all')

    return mean, quantiles


def getFakeData():

    print('Creando datos ficticios...')

    Mc1 = np.random.uniform(19., 21., args['n'])
    eMc1 = np.random.uniform(0.002, 0.005, args['n'])

    Mc2 = np.random.uniform(19., 21., args['n'])
    eMc2 = np.random.uniform(0.002, 0.005, args['n'])

    eMins1 = np.random.uniform(0.005, 0.12, args['n'])

    X = np.random.uniform(1., 1.8, args['n'])

    C = Mc1 - Mc2
    eC = np.sqrt(eMc1 ** 2 + eMc2 ** 2)

    Mi1 = Mc1 + args['k'] * X - args['a'] - args['b'] * C
    eMi1 = np.sqrt(eMins1 ** 2 + eMc1 ** 2 + (args['b'] * eC) ** 2)

    My = Mc1 - Mi1
    eMy = np.sqrt(eMc1 ** 2 + eMi1 ** 2)

    band1 = 'B'
    band2 = 'V'

    getPlots(X, Mi1, eMi1, C, eC, Mc1, eMc1, band1, band2)

    mean, quantiles = getTransformation(My, eMy, X, C, eC, band1, band2)

    k = mean[0]
    ek = np.mean(abs(quantiles[0] - k))

    a = mean[1]
    ea = np.mean(abs(quantiles[1] - a))

    b = mean[2]
    eb = np.mean(abs(quantiles[2] - b))

    print('k(%s) = %.4f ± %.4f' % (band1, k, ek,))

    print('a(%s, %s) = %.4f ± %.4f' % (band1, band2, a, ea,))

    print('b(%s, %s) = %.4f ± %.4f' % (band1, band2, b, eb,))


if args['command'] == 'fake':
    getFakeData()

    sys.exit()

if args['command'] == 'prep':
    if args['b'] and not args['lambda2']:
        prep.error('the following arguments are required: lambda2')
    if args['lambda2'] and not args['b']:
        prep.error('the following arguments are required: b')

    getList()

    sys.exit()

if args['command'] == 'fit':

    print('Procesando el ficheros %s...' % (args['lambda1'],))

    table1 = openFile(args['lambda1'])
    band1 = table1.meta['filter']
    # table2 = openFile(args['filel2'])
    table2 = Table.read('campos.fit', format='fits')
    # band2 = table2.meta['filter']
    band2 = args['lambda2']

    table = join(table1, table2, join_type='inner', keys='Id',
                 metadata_conflicts='silent')

    table.sort('X')

    print('Calculando las ecs. de transformación para el filtro %s' % band1)
    print('con el color (%s - %s)...' % (band1, band2))

    Id = table['Id'].data
    X = table['X'].data

    Mi1 = table['Mi'].data
    eMi1 = table['eMi'].data

    Mc1 = table[band1].data
    eMc1 = table['e%s' % band2].data

    Mc2 = table[band2].data
    eMc2 = table['e%s' % band2].data

    My = Mc1 - Mi1
    eMy = np.sqrt(eMc1 ** 2 + eMi1 ** 2)

    C = Mc1 - Mc2
    eC = np.sqrt(eMc1 ** 2 + eMc2 ** 2)

    mean, quantiles = getTransformation(My, eMy, X, C, eC, band1, band2, Id)

    k = mean[0]
    ek = np.mean(abs(quantiles[0] - k))

    a = mean[1]
    ea = np.mean(abs(quantiles[1] - a))

    b = mean[2]
    eb = np.mean(abs(quantiles[2] - b))

    print('k(%s) = %.4f ± %.4f' % (band1, k, ek,))

    print('a(%s, %s) = %.4f ± %.4f' % (band1, band2, a, ea,))

    print('b(%s, %s) = %.4f ± %.4f' % (band1, band2, b, eb,))
