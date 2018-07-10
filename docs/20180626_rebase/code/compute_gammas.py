#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import collections
import itertools
import logging
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

import global_config as gcf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

CODEFILE = pathlib.Path(__file__).name
GAMMA_FILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.tsv')
RAW_FILE = gcf.OUTPUT_DIR / 'lib234.raw.tsv'

# --------------------

def dose_mapper(name):
  if name[2] == 'a' or name[:2] == 'd1':
    return 'sober'
  if name[:2] == 'd2':
    return 'low'
  if name[:2] == 'd3':
    return 'high'
  logging.error('encountered unhandled sample {name}'.format(**vars()))
  sys.exit(2)

def read_preprocessed_data(rawfile):
  logging.info('Getting preprocessed data from {rawfile}.'.format(**vars()))
  rawdata = pd.read_csv(rawfile, sep='\t', header=0, index_col=0)
  rawdata['sid'] = rawdata['sample'].map(lambda x: x[2:] + x[:1])
  rawdata['tp'] = rawdata['sample'].map(lambda x: int(x[1]))
  rawdata.drop(['day', 'tube', 'sample'], axis='columns', inplace=True)
  rawdata['dose'] = rawdata['sid'].apply(dose_mapper)
  return rawdata

def diff_samples(group, *, k=1):
  tall = group.log
  tall = tall.loc[~tall.index.duplicated()]
  wide = tall.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(k, axis=1)
  return wide.stack()

def normalize(counts):
  return counts * (float(gcf.NORMAL_SIZE) / counts.sum())

def compute_rough_gammas(rawdata):
  logging.info('Computing rough gammas.')
  rawdata['norm'] = rawdata.groupby(['sid', 'tp']).raw.transform(normalize)
  rawdata['log'] = np.log2(rawdata.norm.clip(1))
  rawdata.set_index(['variant', 'sid', 'tp'], inplace=True)
  rawdata = rawdata.loc[~rawdata.index.duplicated()]
  grouper = rawdata.groupby(['sid'], group_keys=False)
  relevant = list()
  for i in range(1, 4):
    diff = grouper.apply(diff_samples, k=i)
    diffcenters = diff.loc[rawdata.control].unstack(level=[-2,-1]).median()
    dg = diff.unstack(level=[-2,-1]).subtract(diffcenters, axis='columns')
    mask = (rawdata.dose == 'sober')
    chosen = dg.stack(level=[0,1]).loc[mask].unstack(level=[-2,-1])
    namespan = namespan_func(i)
    chosen.columns = chosen.columns.map(namespan)
    relevant.append(chosen)
  X = pd.concat(relevant, axis=1)
  return X

def rebase(A, D):
  U_, s_, Vt_ = np.linalg.svd(D, full_matrices=True)
  rank_ = (~np.isclose(s_, 0)).sum()
  basis_ = U_[:, :rank_]
  return np.dot(A, np.dot(basis_, basis_.T))

def dca_smooth_gammas(unsmoothed_gammas):
  logging.info('Applying DCA smoothing.')
  X = unsmoothed_gammas
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
  XDDt = pd.DataFrame(rebase(X, D), index=X.index, columns=X.columns)
  XDDt.columns.set_names('spid', inplace=True)
  return XDDt

def g_fit(od_group):
  g, _ = np.polyfit(od_group.time, np.log2(od_group.od), 1)
  return g

def diff_time(group, k=1):
  tall = group.time
  tall = tall.loc[~tall.index.duplicated()]
  wide = tall.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(k, axis=1)
  return wide.stack()

def namespan_func(k):
  def namespan(id_pair):
    (sid, tp) = id_pair
    front, back = tp-k, tp
    return '{sid}{front}{back}'.format(**vars())
  return namespan

def get_od_data(oddatafile):
  logging.info('Reading OD data from: {oddatafile}'.format(**vars()))
  od_data = pd.read_csv(oddatafile, sep='\t')
  od_data['sid'] = od_data['sample'].map(lambda x: x[2:] + x[:1])
  od_data['tp'] = od_data['sample'].map(lambda x: int(x[1]))
  od_data.drop(['day', 'tube', 'sample'], axis='columns', inplace=True)
  return od_data

def normalize_gammas(rawdata, XDDt, od_data):
  logging.info('Normalizing gammas to 1/gt.')
  g_map = [[sid, g_fit(group)] for sid, group in od_data.groupby('sid')]
  g_map = pd.DataFrame(g_map, columns=['sid', 'g_fit'])
  rawdata = rawdata.loc[~rawdata.index.duplicated()]
  grouper = rawdata.groupby(['sid'], group_keys=False)
  relevant = list()
  for i in range(1, 4):
    namespan = namespan_func(i)
    diff = grouper.apply(diff_time, i)
    chosen = diff.loc[rawdata.dose == 'sober'].unstack(level=[-2,-1])
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
  flatdf = XDDt / (gt_map.g_fit * gt_map.delta_t)
  parts = [(x[:3], x[3:]) for x in flatdf.columns]
  flatdf.columns = pd.MultiIndex.from_tuples(parts, names=['sid', 'span'])
  flatdf.sort_index(axis=1, inplace=True)
  return flatdf

def main():
  rawdata = read_preprocessed_data(RAW_FILE)
  rough_gammas = compute_rough_gammas(rawdata)
  smooth_gammas = dca_smooth_gammas(rough_gammas)
  oddatafile = gcf.OD_FRAME
  od_data = get_od_data(oddatafile)
  solo_gammas = normalize_gammas(rawdata, smooth_gammas, od_data)
  solo_gammas.to_csv(GAMMA_FILE, sep='\t')


if __name__ == '__main__':
  main()
