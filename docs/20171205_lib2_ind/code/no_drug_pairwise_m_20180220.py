#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
#import pdb
import pandas as pd
import seaborn as sns
import scipy.stats as st
import sys

from scipy.interpolate import UnivariateSpline

import pooled.data_to_dataframe as dtd

import global_config as gcf

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
How correctly consistent do the no drug cases look?  First let's look at them
pairwise.
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

# Construct/output graph

grouper = diffdata.groupby(['sample_s', 'sample_e'])
plotsets = list()
plotsets.append(('a1d1', 'a2d1'))
plotsets.append(('b1d1', 'b2d1'))
plotsets.append(('c1d1', 'c2d1'))
plotsets.append(('a1d2', 'a2d2'))
plotsets.append(('a1d3', 'a2d3'))

rows = len(plotsets)
cols = len(plotsets)
gsize = 4
fig, axes = plt.subplots(rows, cols, figsize=(gsize*cols,gsize*rows),
                         sharex='col', sharey='row')
sizes=5
(start, stop) = (-1.6, 0.3)
for i, yitem in enumerate(plotsets):
  for j, xitem in enumerate(plotsets):
    ax = axes[i, j]
    ax.plot([start, stop], [start, stop], 'b--', linewidth=.5)
    leftname = '_'.join(xitem)
    rightname = '_'.join(yitem)
    logging.info(
        'Plotting {leftname} vs {rightname} on {i}, {j}'.format(**vars()))
    left = grouper.get_group(xitem)
    right = grouper.get_group(yitem)
    merged = pd.merge(left, right, on='variant', suffixes=['_l', '_r'])
    ax = sns.regplot('gamma_l', 'gamma_r', data=merged, fit_reg=False, ax=ax,
                     scatter_kws=dict(s=sizes, alpha=0.2, color='magenta'))
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ax.set(xlabel='gamma [' + leftname + ']')
    ax.set(ylabel='gamma [' + rightname + ']')
    ax.label_outer()

fig.suptitle('gammas, pairwise plots [No Drug]')
fig.subplots_adjust(hspace=0, wspace=0)
logging.info('Writing rich graph to {graphout}'.format(**vars()))
plt.savefig(graphout)
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
plt.savefig(graphflat)
plt.close(fig)
