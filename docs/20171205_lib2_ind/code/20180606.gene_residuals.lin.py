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


def get_subvariants(variant, original):
  subvars = list()
  for i in range(len(variant)):
    if variant[i] != original[i]:
      fixone = variant[:i] + original[i] + variant[i+1:]
      if fixone != original:
        subvars.append(fixone)
  return subvars

def one_base_off(row):
  variant, original = row['variant'], row['original']
  for i in range(len(variant)):
    if variant[i] != original[i]:
      fixone = variant[:i] + original[i] + variant[i+1:]
      # that's the one mismatch IFF fixing it works, so we're done either way
      return fixone == original
  return False

def build_one_edit_pairs(omap_file):
  logging.info('Building one edit pairs from {omap_file}.'.format(**vars()))
  omap_df = pd.read_csv(omap_file, sep='\t')
  omap = dict(list(zip(omap_df.variant, omap_df.original)))
  synthetic_singles = list()
  for variant, original in omap.items():
    for sv in get_subvariants(variant, original):
      if sv in omap:
        synthetic_singles.append((variant, sv))
  synthetic_singles = pd.DataFrame(synthetic_singles,
                                   columns=['variant', 'original'])
  orig_singles = omap_df.loc[omap_df.apply(one_base_off, axis=1)]
  oneoffs = pd.concat([synthetic_singles, orig_singles], axis=0)
  oneoffs = oneoffs.reset_index()[['variant', 'original']]
  return oneoffs


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

# Functions to compute features, used by build_feature_frame() below.
def get_gc_cont(row):
  original = row['original']
  return original.count('G') + original.count('C')
def get_firstbase(row):
  original = row['original']
  return original[0]
def get_mm_brackets(row):
  variant = 'N' + row['variant'] + 'N'
  original = 'N' + row['original'] + 'N'
  for i in range(len(variant)):
    if variant[i] != original[i]:
      return ''.join([original[i-1], original[i+1]])
def get_mm_leading(row):
  variant = 'NN' + row['variant']
  original = 'NN' + row['original']
  for i in range(len(variant)):
    if variant[i] != original[i]:
      return original[i-2:i]
def get_mm_trailing(row):
  variant = row['variant'] + 'NG'
  original = row['original'] + 'NG'
  for i in range(len(variant)):
    if variant[i] != original[i]:
      return original[i+1:i+3]
def get_mm_idx(row):
  variant, original = row['variant'], row['original']
  for i in range(len(variant)):
    if variant[i] != original[i]:
      return (20-i)
def get_mm_trans(row):
  variant, original = row['variant'], row['original']
  for i in range(len(variant)):
    if variant[i] != original[i]:
      return ''.join([original[i], variant[i]])

def build_feature_frame(mismatch_pairs):
  """Computes all features for the provided mismatch pairs.
  Args:
    mismatch_pairs: dataframe with variant/original fields
  Returns:
    feature_frame:
      multiindex is just mismatch_pairs
      columns are the feature names
  """
  featset = list()
  firstbase = mismatch_pairs.apply(get_firstbase, axis=1)
  firstbase.name = 'firstbase'
  featset.append(firstbase)
  gc_cont = mismatch_pairs.apply(get_gc_cont, axis=1)
  gc_cont.name = 'gc_cont'
  featset.append(gc_cont)
  mm_brackets = mismatch_pairs.apply(get_mm_brackets, axis=1)
  mm_brackets.name = 'mm_brackets'
  featset.append(mm_brackets)
  mm_leading = mismatch_pairs.apply(get_mm_leading, axis=1)
  mm_leading.name = 'mm_leading'
  featset.append(mm_leading)
  mm_trailing = mismatch_pairs.apply(get_mm_trailing, axis=1)
  mm_trailing.name = 'mm_trailing'
  featset.append(mm_trailing)
  mm_idx = mismatch_pairs.apply(get_mm_idx, axis=1)
  mm_idx.name = 'mm_idx'
  featset.append(mm_idx)
  mm_trans = mismatch_pairs.apply(get_mm_trans, axis=1)
  mm_trans.name = 'mm_trans'
  featset.append(mm_trans)
  # Try the joint feature in case it allows better outcomes.
  mm_zip = list(zip(mm_idx, mm_trans))
  mm_both = ['{0:02}/{1}'.format(*x) for x in mm_zip]
  mm_both = pd.Series(mm_both, index=mm_trans.index)
  mm_both.name = 'mm_both'
  featset.append(mm_both)
  feature_frame = pd.concat([mismatch_pairs] + featset, axis=1)
  feature_frame.set_index(['variant', 'original'], inplace=True)
  return feature_frame

def build_feature_columns(feature_frame):
  feat_cols = list()
  fcol_firstbase = tf.feature_column.categorical_column_with_vocabulary_list(
      'firstbase', sorted(feature_frame.firstbase.unique()))
  feat_cols.append(fcol_firstbase)
  fcol_gc = tf.feature_column.numeric_column('gc_cont')
  feat_cols.append(fcol_gc)
  fcol_brackets = tf.feature_column.categorical_column_with_vocabulary_list(
      'mm_brackets', sorted(feature_frame.mm_brackets.unique()))
  feat_cols.append(fcol_brackets)
  fcol_leading = tf.feature_column.categorical_column_with_vocabulary_list(
      'mm_leading', sorted(feature_frame.mm_leading.unique()))
  feat_cols.append(fcol_leading)
  fcol_trailing = tf.feature_column.categorical_column_with_vocabulary_list(
      'mm_trailing', sorted(feature_frame.mm_trailing.unique()))
  feat_cols.append(fcol_trailing)
  fcol_both = tf.feature_column.categorical_column_with_vocabulary_list(
      'mm_both', sorted(feature_frame.mm_both.unique()))
  feat_cols.append(fcol_both)
  return feat_cols

def split_data(X_all, y_all, grouplabels):
  gss = GroupShuffleSplit(test_size=0.3, random_state=42)
  splititer = gss.split(X_all, y_all, grouplabels)
  train_rows, test_rows = next(splititer)
  train_rows = shuffle(train_rows)
  test_rows = shuffle(test_rows)
  X_train = X_all.loc[train_rows, :]
  y_train = y_all.loc[train_rows]
  X_test = X_all.loc[test_rows, :]
  y_test = y_all.loc[test_rows]
  return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, grouplabels, model_dir):
  model = tf.estimator.LinearRegressor(feature_columns=feat_cols,
                                       model_dir=model_dir)
  train_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_train, y=y_train,
      batch_size=10, num_epochs=None,
      shuffle=True)
  model.train(input_fn=train_input_func, steps=4000)
  return model

def evaluate_model(model, X_eval, y_eval):
  eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_eval, y=y_eval,
      batch_size=10, num_epochs=1,
      shuffle=False)
  eval_out = model.evaluate(input_fn=eval_input_func)
  return eval_out

def apply_model(model, X_test):
  test_pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test, shuffle=False)
  preds = [x['predictions'][0] for x in model.predict(test_pred_input_func)]
  return preds

def train_eval_visualize(X_all, y_all, grouplabels, feat_cols):
  X_train, y_train = X_all, y_all
  logging.info('WARNING *** THIS IS SOME BULLSHIT *** WARNING')
  logging.info('Training model ON ALL DATA...'.format(**vars()))
  logging.info('WARNING *** THIS IS SOME BULLSHIT *** WARNING')
  model_dir = partnerfile(y_all.name + '.model.bad.no')
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  model = train_model(X_all, y_all, feat_cols, model_dir)
  logging.info('Evaluating model...'.format(**vars()))
  train_eval = evaluate_model(model, X_train, y_train)
  train_eval_str = 'BULLSHIT EVAL:\n{train_eval}'.format(**vars())
  logging.info(train_eval_str)
  logging.info('Applying model...'.format(**vars()))
  X_y_all = pd.concat([X_all, y_all], axis=1)
  predictions = apply_model(model, X_all)
  X_y_all['y_pred'] = predictions
  X_y_all['errerr'] = (X_y_all.y_pred - X_y_all[y_all.name]) ** 2
  keys = list()
  # TODO(jsh): FAMILY!
  keys.append('gene')
  keys.append('gc_cont')
  keys.append('firstbase')
  keys.append('mm_brackets')
  keys.append('mm_leading')
  keys.append('mm_trailing')
  keys.append('mm_idx')
  keys.append('mm_trans')
  for key in keys:
    genegrouper = X_y_all.groupby(key)
    groupedfile = partnerfile('.'.join([key, 'stats', 'tsv']))
    keptcols = ['03', 'y_pred', 'errerr']
    outframe = genegrouper.mean().sort_values('errerr')[keptcols]
    outframe.to_csv(groupedfile, sep='\t', float_format='%05.3f')
  IPython.embed()


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
  feature_frame = build_feature_frame(oneoffs)
  gene_map = rawdata.reset_index()[['variant', 'gene_name']].drop_duplicates()
  gene_map.set_index('variant', inplace=True)
  gene = feature_frame.reset_index().variant.map(gene_map.gene_name)
  gene.index = feature_frame.index
  feature_frame['gene'] = gene
  feat_cols = build_feature_columns(feature_frame)
  trainlist = set([x.strip() for x in open(gcf.BROAD_OLIGO_FILE)])
  dfralist = set([x.strip() for x in open(gcf.DFRA_OLIGO_FILE)])
  muraalist = set([x.strip() for x in open(gcf.MURAA_OLIGO_FILE)])
  spans = ['03']
  for span in spans:
    logging.info('Working on **SPAN {span}**...'.format(**vars()))
    span_gammas = relative_gammas[[span]]
    X_y = pd.merge(feature_frame, span_gammas,
                   left_index=True, right_index=True)
    usable_data = X_y.loc[~X_y[span].isnull()]
    broad_check = lambda x: x in trainlist
    broad_mask = usable_data.reset_index().variant.apply(broad_check)
    broad_mask.index = usable_data.index
    usable_data = usable_data.loc[broad_mask]
    X_all = usable_data[feature_frame.columns].reset_index(drop=True)
    y_all = usable_data[span].reset_index(drop=True)
    grouplabels = usable_data.reset_index().variant
    train_eval_visualize(X_all, y_all, grouplabels, feat_cols)
