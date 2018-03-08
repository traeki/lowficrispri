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

widediffdata = widen_groups(diffdata)
cleaned = impute_over_nans(widediffdata)
scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
subset = cleaned[all_no_drug_spans()]
# transposing to scale along the strains, instead of along the measurments
X = scaler.fit_transform(subset.T)

# SVDecompose X directly into USVt
U, s, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T
S = np.diag(s)

# S should be diagonal
assert np.allclose(S, S.T)
# V S.T U.T == X.T
assert np.allclose(V.dot(S.T.dot(U.T)), X.T)
# U S V.T == X
assert np.allclose(U.dot(S.dot(V.T)), X)

PC_names = ['PC{0}'.format(i+1) for i in range(len(Vt))]
rated_guides = pd.DataFrame(V, columns=PC_names, index=cleaned.index)
cutoff = 4*(rated_guides.std())
hits = rated_guides > cutoff
masked = cleaned.copy()
masked['hits'] = hits.PC3

samples = masked.loc[masked.hits].index.tolist()
with open(os.path.join(gcf.OUTPUT_DIR, 'beaksamples.txt'), 'w') as f:
  f.write('\n'.join(samples))

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
    'gammas, painted by PC3 on strains, via SVD'.format(**vars()),
    fontsize=16)
plt.tight_layout()
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
plt.savefig(graphflat)
plt.close()
