#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import collections
import colorcet as cc
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import re
import seaborn as sns
import scipy.stats as st
import sys

from sklearn import decomposition
from sklearn import preprocessing

import global_config as gcf
import IPython
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

np.set_printoptions(precision=4, suppress=True)

rawfile = os.path.join(gcf.OUTPUT_DIR, 'lib234.raw.tsv')
rawdata = pd.read_csv(rawfile, sep='\t', header=0, index_col=0)

PREFIX = os.path.splitext(os.path.basename(__file__))[0]
def partnerfile(ext):
  return os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, ext]))

logging.info('Reading OD data from: {gcf.OD_FRAME}'.format(**vars()))
od_data = pd.read_csv(gcf.OD_FRAME, sep='\t')

def dose_mapper(name):
  if name[0] == 'a' or name[2:] == 'd1':
    return 'sober'
  if name[2:] == 'd2':
    return 'low'
  if name[2:] == 'd3':
    return 'high'
  logging.error('encountered unhandled sample {name}'.format(**vars()))
  sys.exit(2)
rawdata['dose'] = rawdata['sample'].apply(dose_mapper)

logging.info('Fitting g values...'.format(**vars()))
def g_fit(od_group):
  g, _ = np.polyfit(od_group.time, np.log2(od_group.od), 1)
  return g
filtered = od_data.drop('sample', axis='columns').dropna()
sample_od_groups = filtered.groupby(['day', 'tube'])
g_map = [[day, tube, g_fit(value)] for (day, tube), value in sample_od_groups]
g_map = pd.DataFrame(g_map, columns=['day', 'tube', 'g_fit'])
rawdata = pd.merge(rawdata, g_map, how='left', on=['day', 'tube'])

logging.info('Constructing fitness exponent grid...'.format(**vars()))
def normalize(counts):
  return counts * (float(gcf.NORMAL_SIZE) / counts.sum())
rawdata['norm'] = rawdata.groupby('sample').raw.transform(normalize)
rawdata['log'] = np.log2(rawdata.norm.clip(1))
rawdata.set_index(['variant', 'sample'], inplace=True)
def diff_samples(group):
  wide = group.log.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(axis=1)
  wide.iloc[:, 0] = 0.0
  return wide.stack()
grouper = rawdata.groupby(['day', 'tube'], group_keys=False)
rawdata['dgaskew'] = grouper.apply(diff_samples)
diffcenters = rawdata.loc[rawdata.control].dgaskew.unstack().median()
dg = rawdata.dgaskew.unstack().subtract(diffcenters, axis='columns')
rawdata['deltagamma'] = dg.stack()
X = rawdata.loc[rawdata.dose == 'sober'].deltagamma.unstack()

logging.info('DCA-smoothing batch effects...'.format(**vars()))
taggers = list()
taggers.append(('early', lambda s: (s[1] == '1')))
taggers.append(('mid', lambda s: (s[1] == '2')))
taggers.append(('late', lambda s: (s[1] == '3')))
taggers.append(('none', lambda s: (s[0] == 'a' or s[2:] == 'd1')))
taggers.append(('low', lambda s: (s[0] != 'a' and s[2:] == 'd2')))
taggers.append(('high', lambda s: (s[0] != 'a' and s[2:] == 'd3')))
taggers.append(('a', lambda s: (s[0] == 'a')))
taggers.append(('b', lambda s: (s[0] == 'b')))
taggers.append(('c', lambda s: (s[0] == 'c')))
taggers.append(('d1', lambda s: (s[2:] == 'd1')))
taggers.append(('d2', lambda s: (s[2:] == 'd2')))
taggers.append(('d3', lambda s: (s[2:] == 'd3')))
D = np.asarray([[t(s) for (_, t) in taggers] for s in X])
def rebase(A, D):
  U_, s_, Vt_ = np.linalg.svd(D, full_matrices=True)
  rank_ = (~np.isclose(s_, 0)).sum()
  basis_ = U_[:, :rank_]
  return np.dot(A, np.dot(basis_, basis_.T))
XDDt = pd.DataFrame(rebase(X, D), index=X.index, columns=X.columns)

def diff_time(group):
  wide = group.time.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(axis=1)
  wide.iloc[:, 0] = 0.0
  return wide.stack()
grouper = rawdata.groupby(['day', 'tube'], group_keys=False)
rawdata['delta_t'] = grouper.apply(diff_time)

logging.info('Dividing by measured gt...'.format(**vars()))
flatdf = XDDt[['a1d1', 'a2d1', 'a3d1',
               'a1d2', 'a2d2', 'a3d2',
               'a1d3', 'a2d3', 'a3d3',
               'b1d1', 'b2d1', 'b3d1',
               'c1d1', 'c2d1', 'c3d1']]
flatdf = (flatdf.stack() / (rawdata.g_fit * rawdata.delta_t)).dropna()
flatdf = flatdf.reset_index()
flatdf['id'] = flatdf['sample'].map(lambda x: x[2:] + x[:1])
flatdf['tp'] = flatdf['sample'].map(lambda x: int(x[1]))
flatdf.drop('sample', axis=1, inplace=True)
flatdf = flatdf.set_index(['variant', 'id', 'tp']).unstack()
flatdf.columns = flatdf.columns.droplevel(0)

backdf = flatdf.drop(1, axis=1)
grouper = backdf.reset_index().groupby('variant')
plotdf = grouper.agg(np.median)

ttest_colors = grouper.apply(lambda x: st.ttest_rel(x[2], x[3])[1])

xlim = (-2, 0.3)
ylim = (-2, 0.3)
plotdf.plot.scatter(2, 3,
                    s=2, c=ttest_colors,
                    xlim=xlim, ylim=ylim)
plt.savefig(partnerfile('png'))
plt.clf()

IPython.embed()
