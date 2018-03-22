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

datafile = os.path.join(gcf.OUTPUT_DIR, 'lib234.rawlogs.tsv')
data = pd.read_csv(datafile, sep='\t', header=0, index_col=0)

def dose_mapper(name):
  if name[0] == 'a' or name[2:] == 'd1':
    return 'none'
  if name[2:] == 'd2':
    return 'low'
  if name[2:] == 'd3':
    return 'high'
  logging.error('encountered unhandled sample {name}'.format(**vars()))
  sys.exit(2)
data['dose'] = data.sample.apply(dose_mapper)

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

# TODO(jsh): Gather sample-threads by dose
variant_grouper = data.groupby(['variant'])


# TODO(jsh): Figure out how to fit a single thread
#            * Specify the parameterized equation model
#            * What is the term for relative vs. absolute drift
#              * Maybe this is a backfill from control fits?
#            * Do we fit once for all points, or fit N times plus
#              per-trace noise term?
# TODO(jsh): How do we compute residuals?
# TODO(jsh): Given control residuals, can we:
#            * Ascribe probability to other fit-explanations?
#            * Describe the null hypothesis?
# TODO(jsh): Figure out modifications for rho fitting
