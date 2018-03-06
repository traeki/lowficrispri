#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
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
Can I even graph a thing?
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

fig = plt.figure(figsize=(6,6))
sizes=10
grouper = diffdata.groupby(['sample_s', 'sample_e'])
x = ('a0d3', 'a3d3')
y = ('a0d1', 'a3d1')
left = grouper.get_group(x)
right = grouper.get_group(y)
merged = pd.merge(left, right, on='variant', suffixes=['_l', '_r'])
ax = sns.regplot('gamma_l', 'gamma_r', data=merged, fit_reg=False,
                 scatter_kws=dict(s=sizes, alpha=0.2, color='magenta'))
ax.plot([-1.6, .1], [-1.6, .1], 'b--')

(start, stop) = (-1.6, 0.1)
ax.set_xlim(start, stop)
ax.set_ylim(start, stop)
ax.set(xlabel='gamma [Day 3, No Drug]')
ax.set(ylabel='gamma [Day 1, No Drug]')
plt.title('Growth over 10 doublings; Day 1 vs. Day 3')
plt.tight_layout()
plt.savefig(graphout)
plt.savefig(graphflat)
plt.close(fig)
