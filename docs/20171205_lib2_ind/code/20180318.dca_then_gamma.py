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

PREFIX = os.path.splitext(os.path.basename(__file__))[0]
def partnerfile(ext):
  return os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, ext]))

# widen log by sample
data = data.set_index(['variant', 'sample'])
logwide = data.log.unstack()

# build D
taggers = list()
taggers.append(lambda x: True) # global
def timepoint_tagger(i):
  return lambda x: i == int(x[1])
for i in range(4):
  taggers.append(timepoint_tagger(i)) # timepoints
taggers.append(lambda x: (x[0] == 'a' or x[2:] == 'd1')) # sober
taggers.append(lambda x: (x[0] != 'a' and x[2:] == 'd2')) # low
taggers.append(lambda x: (x[0] != 'a' and x[2:] == 'd3')) # high
samples = logwide.columns
tags = [[t(x) for t in taggers] for x in samples]
D = np.asarray(tags)

# compute SVD of D
U, s, Vt = np.linalg.svd(D, full_matrices=True)
rank = (~np.isclose(s, 0)).sum()

# apply constraints to A
A = logwide
basis = U[:, :rank]
projection = np.dot(A, np.dot(basis, basis.T))
kernel = A - projection

# TODO(jsh): normalize A
# TODO(jsh): take logs

data['log'] = np.log2(data.raw.clip(1))

# TODO(jsh): impute NaNs
# TODO(jsh): choose span
# TODO(jsh): compute gammas on all samples
  # TODO(jsh): threshold or attribute uncertainty
# TODO(jsh): graph

logging.error('THIS THREAD IS ABANDONED, JUST KEEP RECENTERING CONTROL MEDIAN')
