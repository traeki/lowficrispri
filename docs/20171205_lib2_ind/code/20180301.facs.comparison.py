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

import global_config as gcf
import sklearn_preproc as prep

import ipdb
from IPython import embed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
Where do the flow-cytometry measurements fall relative to smoothed gammas?
'''

# <NO_EDIT>
statout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'stats']))
graphout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'svg']))
graphflattemplate = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, '{PCS}.png']))
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

# subset = prepped.filter(regex='a|d1')

prepped = prep.prep_for_sklearn(diffdata)

# Compute gammas for 5/10 doublings for pooled
PCS = 1
pca = PCA(n_components=PCS)
pca.fit(prepped)
smoothed = pca.inverse_transform(pca.transform(prepped))
smoothed = pd.DataFrame(smoothed, index=prepped.index, columns=prepped.columns)


# Compute gammas for 5/10 doublings for facs

facsgammasfile = os.path.join(gcf.DATA_DIR, '20171205.facs_gammas.tsv')
facsgammas = pd.read_csv(facsgammasfile, sep='\t', header=0)

variant_idfile = os.path.join(gcf.DATA_DIR, '20171205.dfrA.variant.ids.tsv')
variant_id = pd.read_csv(variant_idfile, sep='\t', header=0)

filtered = facsgammas.loc[~facsgammas.strain.isin(['WT'])].copy()
filtered.strain = filtered.strain.astype('int64')
labeled = pd.merge(filtered, variant_id,
                   left_on='strain', right_on='id',
                   how='inner')
labeled.drop(['t0_t3', 't0_t2', 't1_t3', 't2_t3',
              'id', 'strain', 'row', 'col', 'well'],
             axis='columns', inplace=True)
facsfinal = labeled.groupby(['variant', 'dose']).mean().unstack(level=1)

early_no_drug =   [('a0d1', 'a1d1'),
                   ('b0d1', 'b1d1'),
                   ('c0d1', 'c1d1'),
                   ('a0d2', 'a1d2'),
                   ('a0d3', 'a1d3')]
early_low_drug =  [('b0d2', 'b1d2'),
                   ('c0d2', 'c1d2')]
early_high_drug = [('b0d3', 'b1d3'),
                   ('c0d3', 'c1d3')]

late_no_drug =    [('a1d1', 'a3d1'),
                   ('b1d1', 'b3d1'),
                   ('c1d1', 'c3d1'),
                   ('a1d2', 'a3d2'),
                   ('a1d3', 'a3d3')]
late_low_drug =   [('b1d2', 'b3d2'),
                   ('c1d2', 'c3d2')]
late_high_drug =  [('b1d3', 'b3d3'),
                   ('c1d3', 'c3d3')]

cols = dict()
cols[('early', 'none', 'pool')] = smoothed[early_no_drug].mean(axis='columns')
cols[('early', 'low', 'pool')] = smoothed[early_low_drug].mean(axis='columns')
cols[('early', 'high', 'pool')] = smoothed[early_high_drug].mean(axis='columns')
cols[('late', 'none', 'pool')] = smoothed[late_no_drug].mean(axis='columns')
cols[('late', 'low', 'pool')] = smoothed[late_low_drug].mean(axis='columns')
cols[('late', 'high', 'pool')] = smoothed[late_high_drug].mean(axis='columns')
cols[('early', 'none', 'facs')] = facsfinal[('t0_t1', 0.0)]
cols[('early', 'low', 'facs')] =  facsfinal[('t0_t1', 7.5)]
cols[('early', 'high', 'facs')] = facsfinal[('t0_t1', 30.0)]
cols[('late', 'none', 'facs')] =  facsfinal[('t1_t2', 0.0)]
cols[('late', 'low', 'facs')] =   facsfinal[('t1_t2', 7.5)]
cols[('late', 'high', 'facs')] =  facsfinal[('t1_t2', 30.0)]
final = pd.DataFrame(cols).dropna()


# Graph comparison
doses = ['none', 'low', 'high']
rows = len(doses)
spans = ['early', 'late']
cols = len(spans)
gsize = 4
fig, axes = plt.subplots(rows, cols, figsize=(gsize*cols,gsize*rows),
                         sharex='col', sharey='row')
(start, stop) = (-1.5, .1)
for j, span in enumerate(spans):
  for i, dose in enumerate(doses):
    ax = axes[i, j]
    ax.plot((start, stop), (start, stop), 'r--', linewidth=0.5)
    final[(span, dose)].plot('pool', 'facs', kind='scatter', ax=ax)
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ax.set_title('{span}/{dose}'.format(**vars()))
    ax.set(xlabel='gamma [pool]')
    ax.set(ylabel='gamma [facs]')
    ax.label_outer()
fig.suptitle(
    'gamma comparisons by timespan/dose'.format(**vars()), fontsize=16)
graphflat = graphflattemplate.format(**vars())
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
plt.savefig(graphflat)
plt.close()
