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

# <NO_EDIT>
graphout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'svg']))
graphflat = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'png']))

diffdatafile = os.path.join(gcf.OUTPUT_DIR, 'lib234.diffdata.tsv')
diffdata = pd.read_csv(diffdatafile, sep='\t', header=0)
solodatafile = os.path.join(gcf.OUTPUT_DIR, 'lib234.data.tsv')
solodata = pd.read_csv(solodatafile, sep='\t', header=0)

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

U_scores = U_glob

gene_map = diffdata[['variant', 'gene_name']].drop_duplicates()

PC_names = ['PC{0}'.format(i+1) for i in range(U_scores.shape[1])]
guide_scores = pd.DataFrame(U_scores, columns=PC_names, index=cleaned.index)
scored = pd.merge(guide_scores.reset_index(),
                  gene_map,
                  on='variant', how='left')
scored = scored.drop_duplicates()
glob_piece = pd.DataFrame(glob_piece,
                          index=cleaned.index,
                          columns=cleaned.columns)

best_gamma = glob_piece.mean(axis=1).reset_index()
refined = pd.merge(best_gamma, gene_map, on='variant', how='left')
refined = refined.drop_duplicates()
refined = refined.set_index('variant')
refined.rename(columns={0:'gamma'}, inplace=True)
reficons = refined.groupby('gene_name').get_group('CONTROL').gamma
nullbound = -2 * reficons.std() + reficons.median()

def pick_two(group):
  focus = group.loc[(group.gamma < nullbound) & (group.gamma > 4*nullbound)]
  focus = focus.sort_values('gamma', ascending=False)
  inner = group.loc[group.gamma >= nullbound]
  inner = inner.sort_values('gamma')
  outer = group.loc[group.gamma <= 4*nullbound]
  outer = outer.sort_values('gamma', ascending=False)
  targeted = focus[1:].iloc[(focus[1:].gamma - (2*nullbound)).abs().argsort()]
  chosen = list()
  if len(focus) >= 2:
    chosen.append(focus[:1])
    chosen.append(targeted[:1])
  elif len(focus) == 1:
    chosen.append(focus)
    if len(inner) > 0:
      chosen.append(inner[:1])
    else:
      chosen.append(outer[:1])
  else:
    rest = max(0, 2-len(inner))
    chosen.append(inner[:2])
    chosen.append(outer[:rest])
  chosen = pd.concat(chosen)
  if not len(chosen) == 2:
    logging.error('Did not find two guides for group {0}'.format(group.name))
    sys.exit(2)
  return chosen

picks = refined.groupby('gene_name').apply(pick_two)
picks = picks.gamma
picks.to_csv
pickfile = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'tsv']))
logging.info('Writing picks to {pickfile}'.format(**vars()))
picks.to_csv(pickfile, sep='\t')
