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

# PCA for variants can tell us which variants best explain the differences
# *among no drug controls*.  Best explainers should be real assholes.

# genemap = diffdata.set_index('variant').gene_name
# genemap = genemap.reset_index().drop_duplicates().set_index('variant')

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

prepped = prep.prep_for_sklearn(diffdata)
subset = prepped[spanbrackets]

scaler = StandardScaler(with_mean=True, with_std=True)
scaled = scaler.fit_transform(subset)

pca = PCA(n_components=3, svd_solver='full')
pca.fit(scaled.T)
axisnames = ['PC{0}'.format(i) for i in range(1, len(pca.components_)+1)]
rated_guides = pd.DataFrame(pca.components_.T,
                            columns=axisnames, index=prepped.index)
cutoff = 4*(rated_guides.std())
hits = rated_guides > cutoff

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

##################################

def flip_signs(A, B):
    """
    utility function for resolving the sign ambiguity in SVD
    http://stats.stackexchange.com/q/34396/115202
    """
    signs = np.sign(A) * np.sign(B)
    return A, B * signs


X = scaled

# the p x p covariance matrix
C = np.cov(X, rowvar=False)
print "C = \n", C
# C is a symmetric matrix and so it can be diagonalized:
l, principal_axes = np.linalg.eig(C)
# sort results wrt. eigenvalues
idx = l.argsort()[::-1]
l, principal_axes = l[idx], principal_axes[:, idx]
# the eigenvalues in decreasing order
print "l = \n", l
# a matrix of eigenvectors (each column is an eigenvector)
print "V = \n", principal_axes
# projections of X on the principal axes are called principal components
principal_components = X.dot(principal_axes)
print "Y = \n", principal_components

# we now perform singular value decomposition of X
# "economy size" (or "thin") SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T
S = np.diag(s)

# 1) then columns of V are principal directions/axes.
assert np.allclose(*flip_signs(V, principal_axes))

# 2) columns of US are principal components
assert np.allclose(*flip_signs(U.dot(S), principal_components))

# 3) singular values are related to the eigenvalues of covariance matrix
assert np.allclose((s ** 2) / (n - 1), l)

# 8) dimensionality reduction
k = 3
PC_k = principal_components[:, 0:k]
US_k = U[:, 0:k].dot(S[0:k, 0:k])
assert np.allclose(*flip_signs(PC_k, US_k))

embed()
