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

PREFIX = os.path.splitext(os.path.basename(__file__))[0]
def partnerfile(ext):
  return os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, ext]))

logging.info('Reading OD data from: {gcf.OD_FRAME}'.format(**vars()))
od_data = pd.read_csv(gcf.OD_FRAME, sep='\t')

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

# BEFORE batch effect smoothing

logging.info('Solving USV* = SVD(X)...'.format(**vars()))
U, s, Vt = np.linalg.svd(X, full_matrices=False)
if U[:, 0].mean() > 0:
  U, s, Vt = -U, s, -Vt
V = Vt.T
stackable = pd.DataFrame(U[:, 0],
                         index=rawdata.unstack().index,
                         columns=['a3d1'])
stackable.columns = stackable.columns.set_names(['sample'])
control_gammas = stackable.stack().loc[rawdata.control].unstack()
nullbound = control_gammas.median() - 2 * control_gammas.std()
outer = stackable.loc[(stackable < nullbound).a3d1]

old_U, old_s, old_V = U, s, V
old_control_gammas = control_gammas
old_nullbound = nullbound
old_outer = outer

# AFTER batch effect smoothing

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

logging.info('Solving USV* = SVD(XDDt)...'.format(**vars()))
U, s, Vt = np.linalg.svd(XDDt, full_matrices=False)
if U[:, 0].mean() > 0:
  U, s, Vt = -U, s, -Vt
V = Vt.T
stackable = pd.DataFrame(U[:, 0],
                         index=rawdata.unstack().index,
                         columns=['a3d1'])
stackable.columns = stackable.columns.set_names(['sample'])
control_gammas = stackable.stack().loc[rawdata.control].unstack()
nullbound = control_gammas.median() - 2 * control_gammas.std()
outer = stackable.loc[(stackable < nullbound).a3d1]

def diff_time(group):
  wide = group.time.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(axis=1)
  wide.iloc[:, 0] = 0.0
  return wide.stack()
grouper = rawdata.groupby(['day', 'tube'], group_keys=False)
rawdata['delta_t'] = grouper.apply(diff_time)

omap_file = os.path.join(gcf.DATA_DIR, 'orig_map.tsv')
origmap = pd.read_csv(omap_file, sep='\t')

# TODO(jsh): Compute the new family edge relationship
within_one = collections.defaultdict(list)
within_two = collections.defaultdict(list)
for v in origmap.variant:
  letters = list(v)
  for i, j in itertools.combinations_with_replacement(range(len(v)), 2):
    x = list(v)
    x[i] = 'N'
    x[j] = 'N'
    x = ''.join(x)
    if i == j:
      within_one[x].append(v)
    within_two[x].append(v)

edges = dict()
for group in within_two.values():
  for x, y in itertools.combinations(group, 2):
    if x < y:
      x, y = y, x
    edges[(x, y)] = 2
for group in within_one.values():
  for x, y in itertools.combinations(group, 2):
    if x < y:
      x, y = y, x
    edges[(x, y)] = 1
for x in origmap.variant:
  edges[(x, x)] = 2

omap = dict(zip(origmap.variant, origmap.original))

children = defaultdict(set)
for x in omap:
  children[x].add(omap[x])
for (x, y), weight in edges.iteritems():
  if x == y:
    continue
  if x in children[y] or y in children[x]:
    continue
  ax, ay = omap[x], omap[y]
  if ax != ay:
    logging.fatal('redheaded stepchild: {x}, {y}, {weight}.'.format(**vars()))
    sys.exit(2)
  gx = (x, ax) in edges and edges[(x, ax)] or edges[(ax, x)]
  gy = (y, ay) in edges and edges[(y, ay)] or edges[(ay, y)]
  if gx == 2 and gy == 1:
    children[gy].add(gx)
  if gy > gx:
    children[gx].add(gy)

IPython.embed()

# TODO(jsh): Filter families by parent gamma stability (?)
