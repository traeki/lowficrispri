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
Suspicion is that the g-fit is causing beaks, so how do those fit lines actually look?
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

days = sorted(solodata.day.unique())
tubes = sorted(solodata.tube.unique())
grouper = solodata.groupby(['day', 'tube'])

rows = len(days)
cols = len(tubes)
gsize = 4
fig, axes = plt.subplots(rows, cols, figsize=(gsize*cols,gsize*rows),
                         sharex='col', sharey='row')
sizes=5
for i, day in enumerate(days):
  for j, tube in enumerate(tubes):
    ax = axes[i, j]
    group = grouper.get_group((day, tube))
    ax.scatter(group.time, np.log2(group.od), s=sizes, color='brown')
    g, intercept = np.polyfit(group.time, np.log2(group.od), 1)
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + (g * x_vals)
#     logging.info('day: {day}'.format(**vars()))
#     logging.info('tube: {tube}'.format(**vars()))
#     logging.info('i: {i}'.format(**vars()))
#     logging.info('j: {j}'.format(**vars()))
#     logging.info('x_vals: {x_vals}'.format(**vars()))
#     logging.info('y_vals: {y_vals}'.format(**vars()))
    ax.plot(x_vals, y_vals, 'g--', linewidth=.5)
    ax.set(xlabel=tube)
    ax.set(ylabel=day)
    ax.label_outer()

fig.suptitle('growth fit lines by day/tube')
fig.subplots_adjust(hspace=0, wspace=0)
# logging.info('Writing rich graph to {graphout}'.format(**vars()))
# plt.savefig(graphout)
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
plt.savefig(graphflat)
plt.close(fig)
