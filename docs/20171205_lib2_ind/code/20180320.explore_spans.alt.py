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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

np.set_printoptions(precision=4, suppress=True)

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

#############

diffdatafile = os.path.join(gcf.OUTPUT_DIR, 'lib234.diffdata.tsv')
diffdata = pd.read_csv(diffdatafile, sep='\t', header=0)
solodatafile = os.path.join(gcf.OUTPUT_DIR, 'lib234.data.tsv')
solodata = pd.read_csv(solodatafile, sep='\t', header=0)

#############

def widen_groups(data):
  indexed = data.set_index(['variant', 'sample_s', 'sample_e'])
  return indexed.gamma.unstack(level=[1,2])

def impute_over_nans(data):
  imp = preprocessing.Imputer(strategy='median', axis=1)
  imp.fit(data)
  filled = imp.transform(data)
  return pd.DataFrame(filled, columns=data.columns, index=data.index)

widediffdata = widen_groups(diffdata)
# TODO(jsh): OH MY GOD THIS IS TERRIBLE
# cleaned = impute_over_nans(widediffdata)
cleaned = widediffdata

taggers = dict()
taggers['glob'] = lambda s, e: True
taggers['early'] = lambda s, e: (s[1] == '0' and e[1] == '1')
taggers['mid'] = lambda s, e: (s[1] == '1' and e[1] == '2')
taggers['late'] = lambda s, e: (s[1] == '2' and e[1] == '3')

starts = cleaned.columns.get_level_values(0)
ends = cleaned.columns.get_level_values(1)
pairs = list(zip(starts, ends))

def D_from_taggers(names):
  tags = [[taggers[t](s, e) for t in names] for (s, e) in pairs]
  return np.asarray(tags)

D = dict()
D['glob'] = D_from_taggers(['glob'])
D['early'] = D_from_taggers(['early'])
D['mid'] = D_from_taggers(['mid'])
D['late'] = D_from_taggers(['late'])
A = cleaned

aligned = dict()

# compute SVD of glob
U, s, Vt = np.linalg.svd(D['glob'], full_matrices=True)
rank = (~np.isclose(s, 0)).sum()
# compute aligned
basis = U[:, :rank]
glob_projection = np.dot(A, np.dot(basis, basis.T))

for shared in ['early', 'mid', 'late']:
  # compute SVD of shared
  U, s, Vt = np.linalg.svd(D[shared], full_matrices=True)
  rank = (~np.isclose(s, 0)).sum()
  # apply constraints to A
  basis = U[:, :rank]
  shared_projection = np.dot(A, np.dot(basis, basis.T))
  aligned[shared] = shared_projection

outer = [-1.2, 0.4]

for name, residue in aligned.items():
  relevant = [taggers[name](s, e) for (s, e) in pairs]
  fig = plt.figure(figsize=(6,6))
  g = sns.jointplot(glob_projection[:, relevant][:, 0],
                    residue[:, relevant][:, 0],
                    s=2, linewidth=0.5, alpha=0.5)
  plt.suptitle(
      'global vs. aligned of {name} span'.format(**vars()),
      fontsize=16)
  g.ax_joint.plot(outer, outer)
  g.set_axis_labels('global', name + ' aligned')
  graphflat = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, name, 'png']))
  plt.tight_layout()
  logging.info('Writing flat graph to {graphflat}'.format(**vars()))
  plt.savefig(graphflat)
  plt.close()

for xname, yname in [('early', 'mid'), ('early', 'late'), ('mid', 'late')]:
  xrelevant = [taggers[xname](s, e) for (s, e) in pairs]
  yrelevant = [taggers[yname](s, e) for (s, e) in pairs]
  xrep = aligned[xname][:, xrelevant][:, 0]
  yrep = aligned[yname][:, yrelevant][:, 0]
  fig = plt.figure(figsize=(6,6))
  g = sns.jointplot(xrep, yrep, s=2, linewidth=0.5, alpha=0.5)
  plt.suptitle(
      '{xname} aligned vs. {yname} aligned'.format(**vars()),
      fontsize=16)
  g.ax_joint.plot(outer, outer)
  g.set_axis_labels(xname + ' aligned', yname + ' aligned')
  graphflat = os.path.join(gcf.OUTPUT_DIR,
                           '.'.join([PREFIX, xname, yname, 'png']))
  plt.tight_layout()
  logging.info('Writing flat graph to {graphflat}'.format(**vars()))
  plt.savefig(graphflat)
  plt.close()
