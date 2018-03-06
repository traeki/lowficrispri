#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pdb
import pandas as pd
import seaborn as sns
import scipy.stats as st
import sys

from sklearn.decomposition import PCA

import global_config as gcf
import sklearn_preproc as prep

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
Where is this beak coming from?
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

# Compute expected SD = f = splinefit(mean(variant), SD(variant))
# Divide SD(variant) by f(variant)
# Result: how many standard deviations away from its own average the variant is.
# Problem: "mean" isn't scaled according to fitness relationships between axes.

# Two PCA-related ideas
# 1) Plot points in old-style graphs based on each PCA score for PC[1-10]
# 2) Plot points in old-style graphs after mapping into and out of first k
#    PCs.  When does the beak manifest?
# 2 sounds easier...

prepped = prep.prep_for_sklearn(diffdata)

genemap = diffdata.set_index('variant').gene_name
genemap = genemap.reset_index().drop_duplicates().set_index('variant')

plotsets = list()
plotsets.append(('a0d1', 'a2d1'))
plotsets.append(('b0d1', 'b2d1'))
plotsets.append(('c0d1', 'c2d1'))
plotsets.append(('a0d2', 'a2d2'))
plotsets.append(('b0d2', 'b2d2'))
plotsets.append(('c0d2', 'c2d2'))
plotsets.append(('a0d3', 'a2d3'))
plotsets.append(('b0d3', 'b2d3'))
plotsets.append(('c0d3', 'c2d3'))

for depth in [10,]:
  logging.info('Doing PCA round-trip for depth {depth}.'.format(**vars()))
  pca = PCA(n_components=depth)
  pca.fit(prepped)
  pdb.set_trace()
  axisnames = ['PC{0}'.format(i) for i in range(1, depth+1)]
  pca_warped = pca.transform(prepped)
  pca_warped = pd.DataFrame(pca_warped,
                            columns=axisnames, index=prepped.index)
  pca_smoothed = pca.inverse_transform(pca_warped)
  pca_smoothed = pd.DataFrame(pca_smoothed,
                              columns=prepped.columns, index=prepped.index)
  logging.info('Building figure for depth {depth}.'.format(**vars()))
  plt.figure(figsize=(9,9))
  sns.pairplot(pca_smoothed[plotsets],
               diag_kind='kde',
               plot_kws=dict(s=5, edgecolor='blue', linewidth=0.5, alpha=0.2))
  plt.suptitle(
      'gammas, pairwise plots [No Drug], PC1-{depth}'.format(**vars()),
      fontsize=16)
  depthflat = os.path.join(
      gcf.OUTPUT_DIR, '.'.join([PREFIX, 'pcsmooth', str(depth), 'png']))
  logging.info('Writing flat graph to {depthflat}'.format(**vars()))
  plt.tight_layout()
  plt.savefig(depthflat)
  plt.close()
