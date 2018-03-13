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

def relation(tagger):
    return [tagger(k) for k in pairs]

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

dose_rels = list()
dose_rels.append(relation(dose_none))
dose_rels.append(relation(dose_low))
dose_rels.append(relation(dose_high))
dose_rels = np.asarray(dose_rels).T

span_rels = list()
N_TIMEPOINTS = 4
def build_span(start, end):
  def span_tagger(k):
    s, e = k
    return (int(s[1]) == start) and (int(e[1]) == end)
  return span_tagger
for i in range(N_TIMEPOINTS - 1):
  for j in range(i + 1, N_TIMEPOINTS):
    span_rels.append(relation(build_span(0, 1)))
span_rels = np.asarray(span_rels).T

cond_rels = list()
for i in dose_rels.T:
  for j in span_rels.T:
    cond_rels.append(i & j)
cond_rels = np.asarray(cond_rels).T

day_rels = list()
for day in ('d1', 'd2', 'd3'):
  def day_match(k):
    s, e = k
    return s[2:] == day
  day_rels.append(relation(day_match))
day_rels = np.asarray(day_rels).T

tube_rels = list()
for tube in ('a', 'b', 'c'):
  def tube_match(k):
    s, e = k
    return s[0] == tube
  tube_rels.append(relation(tube_match))
tube_rels = np.asarray(tube_rels).T

rep_rels = list()
for i in day_rels.T:
  for j in tube_rels.T:
    rep_rels.append(i & j)
rep_rels = np.asarray(rep_rels).T

global_rel = np.ones((1, len(pairs)), dtype=bool).T

all_rels = np.concatenate([global_rel,
                           dose_rels, span_rels, cond_rels,
                           day_rels, tube_rels, rep_rels], axis=1)

exp_rels = np.concatenate([global_rel,
                           dose_rels, span_rels, cond_rels], axis=1)

glob_dose_rels = np.concatenate([global_rel, dose_rels], axis=1)

def im_ker_partition(D):
  'Returns: im(), ker() for transformation D'
  U, s, Vt = np.linalg.svd(D, full_matrices=True)
  rank = (~np.isclose(s, 0)).sum()
  im_basis, ker_basis = U[:, :rank], U[:, rank:]
  def image(A):
    return np.dot(A, np.dot(im_basis, im_basis.T))
  def kernel(A):
    return np.dot(A, np.dot(ker_basis, ker_basis.T))
  return image, kernel


image_glob, kernel_glob = im_ker_partition(global_rel)
image_dose, kernel_dose = im_ker_partition(dose_rels)
image_both, kernel_both = im_ker_partition(glob_dose_rels)

subspace_filtered = image_both(cleaned)
glob_piece = image_glob(cleaned)
dose_orthog = image_dose(kernel_glob(cleaned))
subspace_filtered = pd.DataFrame(subspace_filtered,
                                 index=cleaned.index,
                                 columns=cleaned.columns)
U_glob, s_glob, Vt_glob = np.linalg.svd(glob_piece, full_matrices=False)
U_dose, s_dose, Vt_dose = np.linalg.svd(dose_orthog, full_matrices=False)

U_scores = U_dose

PC_names = ['PC{0}'.format(i+1) for i in range(U_scores.shape[1])]
guide_scores = pd.DataFrame(U_scores, columns=PC_names, index=cleaned.index)
# scaler = preprocessing.MaxAbsScaler()
scaler = preprocessing.MinMaxScaler()
colors = scaler.fit_transform(guide_scores)
# # Fold [-1, 1] -> [0, 1]
# colors = colors.abs()
# # Move [-1, 1] -> [0, 1]
# colors = (colors + 1)/2
colors = pd.DataFrame(colors,
                      index=guide_scores.index,
                      columns=guide_scores.columns)

plt.figure(figsize=(6,6))
plt.scatter(U_glob[:,0], U_dose[:,0],
            s=2, linewidth=0.5, alpha=0.5, c=cc.m_inferno_r(colors.PC2))

# plt.suptitle('INSERT OVERALL TITLE HERE'.format(**vars()), fontsize=16)
plt.tight_layout()
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
plt.savefig(graphflat)
plt.close()

scored = pd.merge(guide_scores.reset_index(),
                  diffdata[['variant', 'gene_name']],
                  on='variant', how='left')
genes_pc1 = scored.groupby('gene_name').PC1.mean().sort_values()
pc1_head = genes_pc1[:20]
pc1_tail = genes_pc1[-20:]
genes_pc2 = scored.groupby('gene_name').PC2.mean().sort_values()
pc2_head = genes_pc2[:20]
pc2_tail = genes_pc2[-20:]

import IPython
IPython.embed()
