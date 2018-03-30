#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import itertools
import logging
import colorcet as cc
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

logging.info('Solving USV* = SVD(X)...'.format(**vars()))
U, s, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T
stackable = pd.DataFrame(U[:, 0],
                         index=rawdata.unstack().index,
                         columns=['a3d1'])
stackable.columns = stackable.columns.set_names(['sample'])
control_gammas = stackable.stack().loc[rawdata.control].unstack()
# TODO(jsh): Figure out a way to fix dir/sign of nullbound
nullbound = 2 * control_gammas.std() + control_gammas.median()
outer = stackable.loc[(stackable > nullbound).a3d1]

IPython.embed()
