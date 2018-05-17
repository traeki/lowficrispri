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

rawfile = os.path.join(gcf.OUTPUT_DIR, 'lib234.raw.tsv')
rawdata = pd.read_csv(rawfile, sep='\t', header=0, index_col=0)

omap_file = os.path.join(gcf.DATA_DIR, 'orig_map.tsv')
omap_df = pd.read_csv(omap_file, sep='\t')
omap = dict(zip(omap_df.variant, omap_df.original))

logging.info('Building "subparent" relation.'.format(**vars()))
synthetic_singles = list()
def get_subvariants(variant, original):
  subvars = list()
  for i in range(len(variant)):
    if variant[i] != original[i]:
      fixone = variant[:i] + original[i] + variant[i+1:]
      if fixone != original:
        subvars.append(fixone)
  return subvars
for variant, original in omap.iteritems():
  for sv in get_subvariants(variant, original):
    if sv in omap:
      synthetic_singles.append((variant, sv))
synthetic_singles = pd.DataFrame(synthetic_singles,
                                 columns=['variant', 'original'])

def one_base_off(row):
  variant, original = row['variant'], row['original']
  if variant == original: return False
  for i in range(len(variant)):
    if variant[i] != original[i]:
      fixone = variant[:i] + original[i] + variant[i+1:]
      # that's the one mismatch IFF fixing it works, so we're done either way
      return fixone == original
orig_singles = omap_df.loc[omap_df.apply(one_base_off, axis=1)]

oneoffs = pd.concat([synthetic_singles, orig_singles], axis=0)
oneoffs = oneoffs.reset_index()[['variant', 'original']]

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
  def namespan((sid, tp)):
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
parts = map(lambda x: (x[:3], x[3:]), flatdf.columns)
flatdf.columns = pd.MultiIndex.from_tuples(parts, names=['sid', 'span'])
flatdf.sort_index(axis=1, inplace=True)

control_mask = rawdata.control.unstack(level=[1,2]).iloc[:, 0]
control_mask.name = 'control'

flatspans = flatdf.stack(level=0)
flatspans = flatspans.reset_index().drop('sid', axis=1)
contspans = flatdf.loc[control_mask].stack(level=0)
contspans = contspans.reset_index().drop('sid', axis=1)

logging.info('Creating mismatch features.'.format(**vars()))
def get_mismatch_idx(row):
  variant, original = row['variant'], row['original']
  for i in range(len(variant)):
    if variant[i] != original[i]:
      return i
mismatch_idx = oneoffs.apply(get_mismatch_idx, axis=1)
mismatch_idx.name = 'mismatch_idx'

def get_mismatch_trans(row):
  variant, original = row['variant'], row['original']
  for i in range(len(variant)):
    if variant[i] != original[i]:
      return ''.join([original[i], variant[i]])
mismatch_trans = oneoffs.apply(get_mismatch_trans, axis=1)
mismatch_trans.name = 'mismatch_trans'

featset = [mismatch_idx, mismatch_trans]
featnames = ['mismatch_idx', 'mismatch_trans']
oneoff_features = pd.concat([oneoffs] + featset, axis=1)
oneoff_features.set_index(['variant', 'original'], inplace=True)

logging.info('Computing geodelt-gammas.'.format(**vars()))
# spans = ['01', '12', '23', '02', '13', '03']
spans = ['13', '03']
parent_gammas = pd.merge(oneoffs, flatspans,
                         left_on='original', right_on='variant',
                         how='left', suffixes=('', '_orig'))
parent_gammas = parent_gammas[['variant', 'original'] + spans]
parent_gammas.set_index(['variant', 'original'], inplace=True)
child_gammas = pd.merge(oneoffs, flatspans,
                        left_on='variant', right_on='variant',
                        how='left')
child_gammas = child_gammas[['variant', 'original'] + spans]
child_gammas.set_index(['variant', 'original'], inplace=True)
geodelt_gammas = child_gammas / parent_gammas
filtered_geodelt_gammas = geodelt_gammas.where(
    parent_gammas.abs() > (contspans.std()*10)
)

logging.info('Formalizing feature columns.'.format(**vars()))
oneoff_scored = pd.merge(filtered_geodelt_gammas, oneoff_features,
                         left_index=True, right_index=True)
fcol_idx = tf.feature_column.categorical_column_with_vocabulary_list(
    'mismatch_idx', sorted(oneoff_features.mismatch_idx.unique()))
fcol_trans = tf.feature_column.categorical_column_with_vocabulary_list(
    'mismatch_trans', sorted(oneoff_features.mismatch_trans.unique()))
feat_cols = [fcol_idx, fcol_trans]

train_evals = dict()
test_evals = dict()
for chosen_span in spans:
  logging.info('MODEL FOR SPAN: {chosen_span}'.format(**vars()))
  usable_data = oneoff_scored.loc[~oneoff_scored[chosen_span].isnull()]
  X_all =usable_data[featnames].reset_index(drop=True)
  y_all =usable_data[chosen_span].reset_index(drop=True)
  gss = GroupShuffleSplit(test_size=0.3, random_state=42)
  splititer = gss.split(X_all, y_all, usable_data.reset_index().variant)
  train_rows, test_rows = splititer.next()
  train_rows = shuffle(train_rows)
  test_rows = shuffle(test_rows)
  X_train = X_all.loc[train_rows, :]
  y_train = y_all.loc[train_rows]
  X_test = X_all.loc[test_rows, :]
  y_test = y_all.loc[test_rows]
  logging.info('Creating input functions'.format(**vars()))
  train_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_train, y=y_train,
      batch_size=10, num_epochs=None,
      shuffle=True)
  train_eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_train, y=y_train,
      batch_size=10, num_epochs=1,
      shuffle=False)
  test_eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test, y=y_test,
      batch_size=10, num_epochs=1,
      shuffle=False)

  logging.info('Creating the model.'.format(**vars()))
  model = tf.estimator.LinearRegressor(feature_columns=feat_cols)

  logging.info('Training the model.'.format(**vars()))
  model.train(input_fn=train_input_func, steps=20000)
  train_evals[chosen_span] = model.evaluate(input_fn=train_eval_input_func)
  test_evals[chosen_span] = model.evaluate(input_fn=test_eval_input_func)

  test_pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test, shuffle=False)
  preds = [x['predictions'][0] for x in model.predict(test_pred_input_func)]

  plt.figure(figsize=(6,6))
  plt.xlim(-.1, 2)
  plt.ylim(-.1, 2)
  sns.regplot(np.array(y_test, dtype='float64'),
              np.array(preds, dtype='float64'),
              scatter_kws={
                's': 2,
                'alpha': 0.2,
              })
  plt.xlabel('Measured')
  plt.ylabel('Predicted')
  plt.title('Ratios of Phenotype (Child / Parent)')
  plt.savefig(partnerfile(chosen_span + '.png'))
  plt.clf()

for span in test_evals:
  train_evaluation = train_evals[span]
  test_evaluation = test_evals[span]
  logging.info('TRAIN EVAL [{span}]:\n{train_evaluation}'.format(**vars()))
  logging.info('TEST EVAL [{span}]:\n{test_evaluation}'.format(**vars()))
