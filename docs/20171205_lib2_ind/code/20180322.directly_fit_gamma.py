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

def back_half(samplename):
  return (int(samplename[1]) > 1)
rawdata['late'] = rawdata['sample'].apply(back_half)

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
wide = rawdata.log.unstack()
sync = wide.copy()
for column in wide.columns:
  startsample = column[:1] + '0' + column[2:]
  sync[column] -= wide[startsample]
rawdata['synced'] = sync.stack()
synccons = sync.loc[rawdata.control.unstack().iloc[:, 0]]
recentered = sync.subtract(synccons.median(), axis='columns')
X = recentered.stack().loc[(rawdata.dose == 'sober') & rawdata.late].unstack()

logging.info('Solving USV* = SVD(X)...'.format(**vars()))
U, s, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T
stackable = pd.DataFrame(U[:, 0],
                         index=rawdata.unstack().index,
                         columns=['a3d1'])
stackable.columns = stackable.columns.set_names(['sample'])
control_gammas = stackable.stack().loc[rawdata.control].unstack()
nullbound = -(2 * control_gammas.std()) + control_gammas.median()
outer = stackable.loc[(stackable < nullbound).a3d1]

# TODO(jsh): Feels like early span is effectively penalized three times, etc.
# TODO(jsh): change to incremental shift-difference
# TODO(jsh): Fix the time sub-selection

IPython.embed()

