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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import global_config as gcf
import sklearn_preproc as prep

import pdb
from IPython import embed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

np.set_printoptions(precision=4, suppress=True)

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
What does SVD actually mean?
Can we relate its components to subspace constraints?
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

# genemap = diffdata.set_index('variant').gene_name
# genemap = genemap.reset_index().drop_duplicates().set_index('variant')

singlespans = [(i, i+1) for i in range(3)]
tubes = ['a', 'b', 'c']
days = ['d1', 'd2', 'd3']
spanbrackets = list()
for front, back in singlespans:
  for tube in tubes:
    for day in days:
      fsample = '{tube}{front}{day}'.format(**vars())
      bsample = '{tube}{back}{day}'.format(**vars())
      spanbrackets.append((fsample, bsample))

prepped = prep.prep_for_sklearn(diffdata)
subset = prepped[spanbrackets]

scaler = StandardScaler(with_mean=True, with_std=True)
scaled = scaler.fit_transform(subset)

U, s, V = np.linalg.svd(scaled, full_matrices=False)
S = np.diag(s)
L = np.diag(s*s)
check_scaled = np.dot(U, np.dot(S, V))
C = np.dot(V.T, np.dot(L, V))
CC = np.dot(scaled.T, scaled)

sample_ids = subset.columns.get_level_values(0)
# def mask_covariance(sample_ids):
dim = len(sample_ids)
tubes = [x[0]+x[2:] for x in sample_ids]
spans = [x[1] for x in sample_ids]
def dose(sample):
  if sample[0] == 'a':
    return 0
  if sample[2:] == 'd1':
    return 0
  if sample[2:] == 'd2':
    return 7.5
  if sample[2:] == 'd3':
    return 30
  logging.fatal('dose(sample) choked on {sample}'.format(**vars()))
  sys.exit(2)
doses = [dose(x) for x in sample_ids]
def agreement(tags):
  return lambda i, j: tags[i] == tags[j]
sametube = np.fromfunction(np.vectorize(agreement(tubes)),
                           (dim, dim), dtype=int)
samespan = np.fromfunction(np.vectorize(agreement(spans)),
                           (dim, dim), dtype=int)
samedose = np.fromfunction(np.vectorize(agreement(doses)),
                           (dim, dim), dtype=int)

# mask = samedose & samespan
mask = np.ones_like(C)
C_m = (C * mask)
l_m, V_m = np.linalg.eig(C_m)
idx = l_m.argsort()[::-1]
l_m = l_m[idx]
V_m = V_m[:,idx]
# C_m_check = np.dot(V_m, np.dot(np.diag(L_m), V_m.T))
# np.allclose(C_m, C_m_check)
L_m = np.diag(l_m)
s_m = np.sqrt(l_m)
S_m = np.diag(s_m)
S_m_inv = np.linalg.inv(S_m)
U_m = np.dot(np.dot(scaled, V_m), S_m_inv)
# scaled_check = np.dot(U_m, np.dot(S_m, V_m.T))
# np.allclose(scaled_check, scaled)

colnames = ['WC{i}'.format(**vars()) for i in range(1, subset.shape[1]+1)]
weights = pd.DataFrame(U_m, index=subset.index, columns=colnames)
cutoff = weights.std()*4
hits = weights.abs() > cutoff
tags = weights.loc[hits.WC1].index.tolist()
masked = prepped.copy()
masked['hits'] = hits.WC3

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
plt.suptitle(
    'COVARIANCE-HACKED gammas, pairwise plots [No Drug]',
    fontsize=16)
plt.tight_layout()
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
plt.savefig(graphflat)
plt.close()
