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
rawdata['sid'] = rawdata['sample'].map(lambda x: x[2:] + x[:1])
rawdata['tp'] = rawdata['sample'].map(lambda x: int(x[1]))
rawdata.drop(['day', 'tube', 'sample'], axis='columns', inplace=True)

PREFIX = os.path.splitext(os.path.basename(__file__))[0]
def partnerfile(ext):
  return os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, ext]))

logging.info('Reading OD data from: {gcf.OD_FRAME}'.format(**vars()))
od_data = pd.read_csv(gcf.OD_FRAME, sep='\t')
od_data['sid'] = od_data['sample'].map(lambda x: x[2:] + x[:1])
od_data['tp'] = od_data['sample'].map(lambda x: int(x[1]))
od_data.drop(['day', 'tube', 'sample'], axis='columns', inplace=True)

def dose_mapper(name):
  if name[2] == 'a' or name[:2] == 'd1':
    return 'sober'
  if name[:2] == 'd2':
    return 'low'
  if name[:2] == 'd3':
    return 'high'
  logging.error('encountered unhandled sample {name}'.format(**vars()))
  sys.exit(2)
rawdata['dose'] = rawdata['sid'].apply(dose_mapper)

logging.info('Fitting g values...'.format(**vars()))
def g_fit(od_group):
  g, _ = np.polyfit(od_group.time, np.log2(od_group.od), 1)
  return g
g_map = [[sid, g_fit(group)] for sid, group in od_data.groupby('sid')]
g_map = pd.DataFrame(g_map, columns=['sid', 'g_fit'])

def namespan_func(k):
  def namespan(xxx_todo_changeme):
    (sid, tp) = xxx_todo_changeme
    front, back = tp-k, tp
    return '{sid}{front}{back}'.format(**vars())
  return namespan

logging.info('Constructing fitness exponent grid...'.format(**vars()))
def normalize(counts):
  return counts * (float(gcf.NORMAL_SIZE) / counts.sum())
rawdata['norm'] = rawdata.groupby(['sid', 'tp']).raw.transform(normalize)
rawdata['log'] = np.log2(rawdata.norm.clip(1))
rawdata.set_index(['variant', 'sid', 'tp'], inplace=True)
def diff_samples(group, k=1):
  wide = group.log.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(k, axis=1)
  return wide.stack()
grouper = rawdata.groupby(['sid'], group_keys=False)
relevant = list()
for i in range(1, 4):
  diff = grouper.apply(diff_samples, i)
  diffcenters = diff.loc[rawdata.control].unstack(level=[1,2]).median()
  dg = diff.unstack(level=[1,2]).subtract(diffcenters, axis='columns')
  mask = (rawdata.dose == 'sober')
  chosen = dg.stack(level=[0,1]).loc[mask].unstack(level=[1,2])
  namespan = namespan_func(i)
  chosen.columns = chosen.columns.map(namespan)
  relevant.append(chosen)
X = pd.concat(relevant, axis=1)

logging.info('DCA-smoothing batch effects...'.format(**vars()))
taggers = list()
taggers.append(('span01', lambda s: (s[-2:] == '01')))
taggers.append(('span02', lambda s: (s[-2:] == '02')))
taggers.append(('span03', lambda s: (s[-2:] == '03')))
taggers.append(('span12', lambda s: (s[-2:] == '12')))
taggers.append(('span13', lambda s: (s[-2:] == '13')))
taggers.append(('span23', lambda s: (s[-2:] == '23')))
taggers.append(('none', lambda s: (s[2] == 'a' or s[:2] == 'd1')))
taggers.append(('low', lambda s: (s[2] != 'a' and s[:2] == 'd2')))
taggers.append(('high', lambda s: (s[2] != 'a' and s[:2] == 'd3')))
taggers.append(('a', lambda s: (s[2] == 'a')))
taggers.append(('b', lambda s: (s[2] == 'b')))
taggers.append(('c', lambda s: (s[2] == 'c')))
taggers.append(('d1', lambda s: (s[:2] == 'd1')))
taggers.append(('d2', lambda s: (s[:2] == 'd2')))
taggers.append(('d3', lambda s: (s[:2] == 'd3')))
D = np.asarray([[tagger(col) for (_, tagger) in taggers] for col in X])
def rebase(A, D):
  U_, s_, Vt_ = np.linalg.svd(D, full_matrices=True)
  rank_ = (~np.isclose(s_, 0)).sum()
  basis_ = U_[:, :rank_]
  return np.dot(A, np.dot(basis_, basis_.T))
XDDt = pd.DataFrame(rebase(X, D), index=X.index, columns=X.columns)
XDDt.columns.set_names('spid', inplace=True)

def diff_time(group, k=1):
  wide = group.time.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(k, axis=1)
  return wide.stack()
grouper = rawdata.groupby(['sid'], group_keys=False)
relevant = list()
for i in range(1, 4):
  namespan = namespan_func(i)
  diff = grouper.apply(diff_time, i)
  chosen = diff.loc[rawdata.dose == 'sober'].unstack(level=[1,2])
  chosen.columns = chosen.columns.map(namespan)
  relevant.append(chosen)
t_map = pd.concat(relevant, axis=1).iloc[0]
t_map.name = 'delta_t'
gt_map = pd.DataFrame(t_map)
gt_map.index.name = 'spid'
gt_map['sid'] = gt_map.index.map(lambda x: x[:3])
gt_map = pd.merge(gt_map.reset_index(), g_map, on='sid', how='left')
gt_map.drop(['sid'], axis=1, inplace=True)
gt_map.set_index('spid', inplace=True)

logging.info('Dividing by measured gt...'.format(**vars()))
flatdf = XDDt / (gt_map.g_fit * gt_map.delta_t)
parts = [(x[:3], x[3:]) for x in flatdf.columns]
flatdf.columns = pd.MultiIndex.from_tuples(parts, names=['sid', 'span'])
flatdf.sort_index(axis=1, inplace=True)

control_mask = rawdata.control.unstack(level=[1,2]).iloc[:, 0]
control_mask.name = 'control'

flatspans = flatdf.stack(level=0)
flatspans = flatspans.reset_index().drop('sid', axis=1)
contspans = flatdf.loc[control_mask].stack(level=0)
contspans = contspans.reset_index().drop('sid', axis=1)

logging.info('Computing Mann-Whitney U statistics...'.format(**vars()))
def mwu_wrapper(cspans):
  def compute_mwu(group):
    group.drop('variant', axis=1, inplace=True)
    test = (lambda x: st.mannwhitneyu(x, contspans[x.name],
                                      alternative='two-sided')[1])
    return group.apply(test)
  return compute_mwu
group_mwu = mwu_wrapper(contspans)

pvalues = flatspans.groupby('variant', group_keys=False).apply(group_mwu)

cps = pvalues.loc[control_mask]
rps = pvalues.loc[~control_mask]

print((rps < 0.001).sum())
print((cps < 0.001).sum())

IPython.embed()
