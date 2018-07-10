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
import tensorflow as tf

import global_config as gcf
from build_oneoffs import ONEOFF_FILE

import IPython

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

CODEFILE = pathlib.Path(__file__).name
FEATURE_FILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.tsv')

# --------------------

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


def main():
  oneoffs = pd.read_csv(ONEOFF_FILE, sep='\t')
  feature_frame = build_feature_frame(oneoffs)
  feature_frame.to_csv(FEATURE_FILE, sep='\t')

if __name__ == '__main__':
  main()
