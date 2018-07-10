#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]
import collections
import itertools
import logging
import pathlib
import re
import shutil
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
import tensorflow as tf

import global_config as gcf
from build_feature_frame import FEATURE_FILE
from compute_relative_gammas import RELATIVE_FILE
from compute_relative_gammas import PARENT_FILE
from compute_relative_gammas import CHILD_FILE

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

CODEFILE = pathlib.Path(__file__).name
TRAIN_FILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.train.tsv')
TEST_FILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.test.tsv')
GUIDESETS = dict()
GUIDESETS['broad'] = set([x.strip() for x in open(gcf.BROAD_OLIGO_FILE)])
GUIDESETS['muraa'] = set([x.strip() for x in open(gcf.MURAA_OLIGO_FILE)])
GUIDESETS['dfra'] = set([x.strip() for x in open(gcf.DFRA_OLIGO_FILE)])
GUIDESETS['all'] = set([x.strip() for x in open(gcf.OLIGO_FILE)])
MODEL_DIRS = dict()
for subset in GUIDESETS:
  base = gcf.OUTPUT_DIR / 'models' / CODEFILE
  MODEL_DIRS[subset] = base.with_suffix('.' + subset + '.model')


# --------------------

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

def split_data(all_data, grouplabels):
  gss = GroupShuffleSplit(test_size=0.3)
  splititer = gss.split(all_data, groups=grouplabels)
  train_rows, test_rows = next(splititer)
  train_rows = shuffle(train_rows)
  test_rows = shuffle(test_rows)
  train = all_data.iloc[train_rows, :]
  test = all_data.iloc[test_rows, :]
  return train, test

def train_model(X_train, y_train, feat_cols, model_dir):
  model = tf.estimator.LinearRegressor(feature_columns=feat_cols,
                                       model_dir=model_dir)
  train_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_train, y=y_train,
      batch_size=10, num_epochs=None,
      shuffle=True)
  model.train(input_fn=train_input_func, steps=3000)
  return model


def main():
  span = '03'
  feature_frame = pd.read_csv(FEATURE_FILE, sep='\t', index_col=[0,1])
  feature_frame.fillna('NA', inplace=True)
  output_columns = set(['y_meas'])
  data_columns = set(feature_frame.columns) - output_columns
  relative_gammas = pd.read_csv(RELATIVE_FILE, sep='\t', index_col=[0,1])
  relative_gammas = relative_gammas[[span]]
  relative_gammas.columns = ['y_meas']
  parent_gammas = pd.read_csv(PARENT_FILE, sep='\t', index_col=[0,1])
  parent_gammas = parent_gammas[[span]]
  parent_gammas.columns = ['parent']
  child_gammas = pd.read_csv(CHILD_FILE, sep='\t', index_col=[0,1])
  child_gammas = child_gammas[[span]]
  child_gammas.columns = ['child']
  supplement = pd.concat([relative_gammas, parent_gammas], axis=1)
  feat_cols = build_feature_columns(feature_frame)
  feature_frame = pd.merge(feature_frame, supplement,
                           left_index=True, right_index=True)
  feature_frame.reset_index(inplace=True)
  feature_frame['family'] = feature_frame.original
  feature_frame.set_index(['variant', 'original'], inplace=True)
  usable_data = feature_frame.loc[~feature_frame['y_meas'].isnull()]
  grouplabels = usable_data.family
  # TODO(jsh) why are train/test coming back empty?
  train, test = split_data(usable_data, grouplabels)
  train.to_csv(TRAIN_FILE, sep='\t')
  test.to_csv(TEST_FILE, sep='\t')
  for subsetkey, guideset in GUIDESETS.items():
    logging.info('TRAINING: {subsetkey}'.format(**vars()))
    subset_check = lambda x: x in guideset
    subset_mask = train.reset_index().variant.apply(subset_check)
    subset_mask.index = train.index
    subset_data = train.loc[subset_mask]
    X_subset = subset_data[list(data_columns)].reset_index(drop=True)
    y_subset = subset_data[list(output_columns)].reset_index(drop=True)
    model_dir = MODEL_DIRS[subsetkey]
    # TODO(jsh): build the "next" subdirectory and use it (for e.g. cross-val)
    shutil.rmtree(model_dir, ignore_errors=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model = train_model(X_subset, y_subset, feat_cols, model_dir)

if __name__ == '__main__':
  main()
