#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import itertools
import logging
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

import pdb
from IPython import embed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

np.set_printoptions(precision=4, suppress=True)

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
What does it look like if we paint beaks in apple-to-apples base set
'''

# <NO_EDIT>
statout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'stats']))
graphout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'svg']))
graphflat = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'png']))
notesout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'notes']))

diffdatafile = os.path.join(gcf.OUTPUT_DIR, 'lib234.diffdata.tsv')
diffdata = pd.read_csv(diffdatafile, sep='\t', header=0)
solodatafile = os.path.join(gcf.OUTPUT_DIR, 'lib234.data.tsv')
solodata = pd.read_csv(solodatafile, sep='\t', header=0)

# Output notes
with open(notesout, 'w') as f:
  f.write(QUESTION.format(**vars()))
# </NO_EDIT>

#############

def widen_groups(data):
  indexed = data.set_index(['variant', 'sample_s', 'sample_e'])
  return indexed.gamma.unstack(level=[1,2])

def impute_over_nans(data):
  imp = preprocessing.Imputer(strategy='median', axis=1)
  imp.fit(data)
  filled = imp.transform(data)
  return pd.DataFrame(filled, columns=data.columns, index=data.index)

def flip_signs(A, B):
    """
    utility function for resolving the sign ambiguity in SVD
    http://stats.stackexchange.com/q/34396/115202
    """
    signs = np.sign(A) * np.sign(B)
    return A, B * signs

widediffdata = widen_groups(diffdata)
cleaned = impute_over_nans(widediffdata)

starts = cleaned.columns.get_level_values(0)
ends= cleaned.columns.get_level_values(1)
pairs = zip(starts, ends)
minspans = [(s, e) for (s, e) in pairs if (int(e[1]) - int(s[1])) == 1]
subset = cleaned[minspans]

def relation(tagger):
    return [tagger(k) for k in minspans]

def dose_none(k):
  s, e = k
  if s[0] == 'a':
    return True
  if s[2:] == 'd1':
    return True
  return False
def dose_low(k):
  s, e = k
  if s[0] == 'a':
    return False
  if s[2:] == 'd2':
    return True
  return False
def dose_high(k):
  s, e = k
  if s[0] == 'a':
    return False
  if s[2:] == 'd3':
    return True
  return False

def span_early(k):
  s, e = k
  assert 1 == int(e[1]) - int(s[1])
  return int(s[1]) == 0
def span_mid(k):
  s, e = k
  assert 1 == int(e[1]) - int(s[1])
  return int(s[1]) == 1
def span_late(k):
  s, e = k
  assert 1 == int(e[1]) - int(s[1])
  return int(s[1]) == 2

dose_rels = list()
dose_rels.append(relation(dose_none))
dose_rels.append(relation(dose_low))
dose_rels.append(relation(dose_high))
dose_rels = np.asarray(dose_rels)

span_rels = list()
span_rels.append(relation(span_early))
span_rels.append(relation(span_mid))
span_rels.append(relation(span_late))
span_rels = np.asarray(span_rels)

cond_rels = list()
for i in dose_rels:
  for j in span_rels:
    cond_rels.append(i & j)
cond_rels = np.asarray(cond_rels)

day_rels = list()
for day in ('d1', 'd2', 'd3'):
  def day_match(k):
    s, e = k
    return s[2:] == day
  day_rels.append(relation(day_match))
day_rels = np.asarray(day_rels)

tube_rels = list()
for tube in ('a', 'b', 'c'):
  def tube_match(k):
    s, e = k
    return s[0] == tube
  tube_rels.append(relation(tube_match))
tube_rels = np.asarray(tube_rels)

rep_rels = list()
for i in day_rels:
  for j in tube_rels:
    rep_rels.append(i & j)
rep_rels = np.asarray(rep_rels)

global_rel = np.ones((1, len(minspans)), dtype=bool)

all_rels = np.concatenate([global_rel,
                           dose_rels, span_rels, cond_rels,
                           day_rels, tube_rels, rep_rels])
all_rels = all_rels.T

exp_rels = np.concatenate([global_rel,
                           dose_rels, span_rels, cond_rels])
exp_rels = exp_rels.T

D = exp_rels
U, s, Vt = np.linalg.svd(D, full_matrices=True)
S = np.diag(s)
Si = np.linalg.pinv(S)

good_mask = ~np.isclose(s, 0)
rank = sum(good_mask)

A = subset
cols = A.columns
rows = A.index

Uspan = U[:, :rank]
A_expected = A.dot(Uspan.dot(Uspan.T))
A_exp_scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
A_expected = A_exp_scaler.fit_transform(A_expected.T).T
A_expected = pd.DataFrame(A_expected, columns=cols, index=rows)
U_exp, s_exp, Vt_exp = np.linalg.svd(A_expected, full_matrices=False)
V_exp = Vt_exp.T

Unull = U[:, rank:]
A_unexpected = A.dot(Unull.dot(Unull.T))
A_unx_scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
A_unexpected = A_unx_scaler.fit_transform(A_unexpected.T).T
A_unexpected = pd.DataFrame(A_unexpected, columns=cols, index=rows)
U_unx, s_unx, Vt_unx = np.linalg.svd(A_unexpected, full_matrices=False)
V_unx = Vt_unx.T

U_scores = U_unx

PC_names = ['PC{0}'.format(i+1) for i in range(U_scores.shape[1])]
guide_scores = pd.DataFrame(U_scores, columns=PC_names, index=cleaned.index)

cutoff = 4*(guide_scores.std())
hits = guide_scores < -cutoff

masked = cleaned.copy()
masked['hits'] = hits.PC1

samples = masked.loc[masked.hits].index.tolist()
# with open(os.path.join(gcf.OUTPUT_DIR, 'beaksamples.txt'), 'w') as f:
#   f.write('\n'.join(samples))

plotsets = list()
plotsets.append(('a0d1', 'a2d1'))
plotsets.append(('b0d1', 'b2d1'))
plotsets.append(('c0d1', 'c2d1'))
plotsets.append(('a0d2', 'a2d2'))
plotsets.append(('a0d3', 'a2d3'))

plt.figure(figsize=(6,6))
sns.pairplot(masked,
             diag_kind='kde',
             vars=plotsets,
             hue='hits',
             plot_kws=dict(s=5, linewidth=0.5, alpha=0.2))
# plt.suptitle(
#     'gammas, checking for agreement'.format(**vars()),
#     fontsize=16)
plt.tight_layout()
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
plt.savefig(graphflat)
plt.close()
