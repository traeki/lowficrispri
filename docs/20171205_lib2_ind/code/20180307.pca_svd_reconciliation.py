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

def all_no_drug_spans():
  allspans = list()
  for i in range(0, 3):
    for j in range(i+1, 4):
      allspans.append((i,j))
  tubes = ['a', 'b', 'c']
  days = ['d1', 'd2', 'd3']
  spanbrackets = list()
  # for front, back in singlespans:
  for front, back in allspans:
    for tube in tubes:
      for day in days:
        if not ((day == 'd1') or (tube == 'a')):
          continue
        fsample = '{tube}{front}{day}'.format(**vars())
        bsample = '{tube}{back}{day}'.format(**vars())
        spanbrackets.append((fsample, bsample))
  return spanbrackets


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

# Generate same principal component with
#   * sklearn.decomposition.PCA
#   * numpy.linalg.svd

# widediffdata = widen_groups(diffdata)
# TODO(jsh): THIS IS SETTING UP TINY BULLSHIT NOT THE REAL FRAME
widediffdata = widen_groups(diffdata).iloc[:5, :3]
cleaned = impute_over_nans(widediffdata)
scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
# TODO(jsh): THIS IS SETTING UP TINY BULLSHIT NOT THE REAL FRAME
# subset = cleaned[all_no_drug_spans()]
subset = cleaned
X = scaler.fit_transform(subset)

# Get PC1 from PCA

pca = decomposition.PCA(n_components=3)
pca.fit(X)
# axisnames = ['PC{0}'.format(i) for i in range(1, len(pca.components_)+1)]
# rated_guides = pd.DataFrame(pca.components_.T,
#                             columns=axisnames, index=prepped.index)

# Eigendecompose Covariance matrix into VLVt and compute Y=XV
C = np.cov(X, rowvar=False)
l, principal_axes = np.linalg.eig(C)
idx = l.argsort()[::-1]
l, principal_axes = l[idx], principal_axes[:, idx]
principal_components = X.dot(principal_axes)

# SVDecompose X directly into USVt
U, s, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T
S = np.diag(s)

# Verify that mixed decomposition reconstitutes C
assert np.allclose(np.dot(V, np.dot(np.diag(l), Vt)), C)
# Verify that eig->VLVt decomposition reconstitutes C
assert np.allclose(principal_axes.dot(np.diag(l).dot(principal_axes.T)), C)
# Verify that svd->V matches eig->V
assert np.allclose(*flip_signs(V, principal_axes))
# Verify that svd->US matches eig->XV
assert np.allclose(*flip_signs(U.dot(S), principal_components))
# Verify that svd->singular_values match eig->eigenvalues
n = X.shape[0] # chosen because it works, not sure why this is 'n', though.
assert np.allclose((s ** 2) / (n - 1), l)

# S should be diagonal
assert np.allclose(S, S.T)
# V S.T U.T == X.T
assert np.allclose(V.dot(S.T.dot(U.T)), X.T)
# U S V.T == X
assert np.allclose(U.dot(S.dot(V.T)), X)

Y = scaler.fit_transform(subset.T)
# Y = X.T
C_alt = np.cov(Y, rowvar=False)
l_alt, principal_axes_alt = np.linalg.eig(C_alt)
l_alt, principal_axes_alt = l_alt.real, principal_axes_alt.real
idx_alt = l_alt.argsort()[::-1]
l_alt, principal_axes_alt = l_alt[idx_alt], principal_axes_alt[:, idx_alt]
principal_components_alt = Y.dot(principal_axes_alt)

U_alt, s_alt, Vt_alt = np.linalg.svd(Y, full_matrices=False)
V_alt = Vt_alt.T
S_alt = np.diag(s_alt)

altpca = decomposition.PCA(n_components=3)
altpca.fit(Y)

embed()

# svd:U S V (X) == alt:svd:V S U
# TODO(jsh): assert np.allclose(S, S_alt)
assert np.allclose(*flip_signs(V, U_alt))
assert np.allclose(*flip_signs(U, V_alt))

assert np.allclose(S_alt, S_alt.T)
# alt:V S.T U.T == Y.T
assert np.allclose(V_alt.dot(S_alt.T.dot(U_alt.T)), Y.T)
# alt:U S V.T == Y
assert np.allclose(U_alt.dot(S_alt.dot(V_alt.T)), Y)

# Verify that alt:eig -> V L Vt = cov(Y)
assert np.allclose(np.dot(principal_axes_alt,
                          np.dot(np.diag(l_alt),
                                 principal_axes_alt.T)),
                   C_alt)
# Verify that svd->V matches eig->V
assert np.allclose(*flip_signs(V_alt, principal_axes_alt))
# Verify that svd->US matches eig->YV
assert np.allclose(*flip_signs(U_alt.dot(S_alt), principal_components_alt))
# Verify that svd->singular_values match eig->eigenvalues
n_alt = Y.shape[0]
assert np.allclose((s_alt ** 2) / (n_alt - 1), l_alt[:len(s_alt)])
# Verify that mixed decomposition reconstitutes C
assert np.allclose(np.dot(V_alt,
                          np.dot(np.diag(l_alt[:len(s_alt)]),
                                 Vt_alt)),
                   C_alt)

# Compare

# U, s, V = np.linalg.svd(scaled, full_matrices=False)
# S = np.diag(s)
# L = np.diag(s*s)
# check_scaled = np.dot(U, np.dot(S, V))
# C = np.dot(V.T, np.dot(L, V))
# CC = np.dot(scaled.T, scaled)
#
# masked = prepped.copy()
# masked['hits'] = hits.PC3
#
# samples = masked.loc[masked.hits].index.tolist()
# with open(os.path.join(gcf.OUTPUT_DIR, 'beaksamples.txt'), 'w') as f:
#   f.write('\n'.join(samples))
#
# plotsets = list()
# plotsets.append(('a0d1', 'a2d1'))
# plotsets.append(('b0d1', 'b2d1'))
# plotsets.append(('c0d1', 'c2d1'))
# plotsets.append(('a0d2', 'a2d2'))
# plotsets.append(('a0d3', 'a2d3'))
#
# plt.figure(figsize=(6,6))
# sns.pairplot(masked,
#              diag_kind='kde',
#              vars=plotsets,
#              hue='hits',
#              plot_kws=dict(s=5, linewidth=0.5, alpha=0.2))
# plt.suptitle(
#     'gammas, pairwise plots [No Drug], in-condition "differentiators"'.format(**vars()),
#     fontsize=16)
# plt.tight_layout()
# logging.info('Writing flat graph to {graphflat}'.format(**vars()))
# plt.savefig(graphflat)
# plt.close()
