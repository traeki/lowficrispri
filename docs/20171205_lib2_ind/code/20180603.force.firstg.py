#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import collections
import itertools
import logging
import os.path
import re
import sys

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
import tensorflow as tf

import global_config as gcf

import IPython

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)
PREFIX = os.path.splitext(os.path.basename(__file__))[0]
def partnerfile(ext):
  return os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, ext]))

ALLSPANS = ['01', '12', '23', '02', '13', '03']

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


def first_base_off(row):
  variant, original = row['variant'], row['original']
  for i in range(len(variant)):
    if variant[0] == 'G' and original[0] != 'G':
      fixone = original[0] + variant[1:]
      # that's the one mismatch IFF fixing it works, so we're done either way
      return fixone == original
  return False

def build_one_edit_pairs(omap_file):
  logging.info('Building one edit pairs from {omap_file}.'.format(**vars()))
  omap_df = pd.read_csv(omap_file, sep='\t')
  omap = dict(list(zip(omap_df.variant, omap_df.original)))
  first_to_g = omap_df.loc[omap_df.apply(first_base_off, axis=1)]
  first_to_g = first_to_g.reset_index()[['variant', 'original']]
  return first_to_g


def compute_rough_gammas(rawdata):
  logging.info('Computing rough gammas.')
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
  wide = group.time.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(k, axis=1)
  return wide.stack()

def namespan_func(k):
  def namespan(xxx_todo_changeme):
    (sid, tp) = xxx_todo_changeme
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
  flatdf = XDDt / (gt_map.g_fit * gt_map.delta_t)
  parts = [(x[:3], x[3:]) for x in flatdf.columns]
  flatdf.columns = pd.MultiIndex.from_tuples(parts, names=['sid', 'span'])
  flatdf.sort_index(axis=1, inplace=True)
  return flatdf


def compute_relative_gammas(rawdata, oneoffs, individual_gammas):
  """Compare gammas between "parent" and "child" strains.

  Args:
    oneoffs: two-column list of mismatch pairs
    individual_gammas: 
  Returns:
    relative_gammas dataframe
  """
  logging.info('Computing relative gammas.'.format(**vars()))
  control_mask = rawdata.control.unstack(level=[1,2]).iloc[:, 0]
  control_mask.name = 'control'
  flatspans = individual_gammas.stack(level=0)
  flatspans = flatspans.reset_index().drop('sid', axis=1)
  contspans = individual_gammas.loc[control_mask].stack(level=0)
  contspans = contspans.reset_index().drop('sid', axis=1)
  parent_gammas = pd.merge(oneoffs, flatspans,
                           left_on='original', right_on='variant',
                           how='left', suffixes=('', '_orig'))
  parent_gammas.drop('variant_orig', axis=1, inplace=True)
  parent_gammas.set_index(['variant', 'original'], inplace=True)
  child_gammas = pd.merge(oneoffs, flatspans,
                          left_on='variant', right_on='variant',
                          how='left')
  child_gammas.set_index(['variant', 'original'], inplace=True)
  geodelt_gammas = (child_gammas / parent_gammas) - 1
  filtered = geodelt_gammas.where(parent_gammas.abs() > (contspans.std()*15))
  return filtered

def plot_relgamma_distribution(relgammas, plotfile):
  logging.info('Plotting relgamma dist to {plotfile}...'.format(**vars()))
  plt.figure(figsize=(10,6))
  ax = sns.distplot(relgammas)
  main_title_str = 'Distribution of Relative Gammas'
  plt.title(main_title_str)
  plt.tight_layout()
  plt.savefig(plotfile)
  plt.clf()

if __name__ == '__main__':
  omap_file = os.path.join(gcf.DATA_DIR, 'orig_map.tsv')
  oneoffs = build_one_edit_pairs(omap_file)
  rawfile = os.path.join(gcf.OUTPUT_DIR, 'lib234.raw.tsv')
  oddatafile = gcf.OD_FRAME
  rawdata = read_preprocessed_data(rawfile)
  rough_gammas = compute_rough_gammas(rawdata)
  smooth_gammas = dca_smooth_gammas(rough_gammas)
  od_data = get_od_data(oddatafile)
  solo_gammas = normalize_gammas(rawdata, smooth_gammas, od_data)
  relative_gammas = compute_relative_gammas(rawdata, oneoffs, solo_gammas)
  trainlist = set([x.strip() for x in open(gcf.BROAD_OLIGO_FILE)])
  dfralist = set([x.strip() for x in open(gcf.DFRA_OLIGO_FILE)])
  muraalist = set([x.strip() for x in open(gcf.MURAA_OLIGO_FILE)])
  spans = ['03']
  for span in spans:
    logging.info('Working on **SPAN {span}**...'.format(**vars()))
    span_gammas = relative_gammas[[span]]
    usable_data = span_gammas.loc[~span_gammas[span].isnull()]
    broad_check = lambda x: x in trainlist
    broad_mask = usable_data.reset_index().variant.apply(broad_check)
    broad_mask.index = usable_data.index
    usable_data = usable_data.loc[broad_mask]
    rgdist_plotfile = partnerfile('png')
    plot_relgamma_distribution(usable_data, rgdist_plotfile)
