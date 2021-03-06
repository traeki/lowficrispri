#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import seaborn as sns
import sys

import pooled.data_to_dataframe as dtd

import global_config as gcf

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
How much falloff do we see for light drug in just the center span (T1 -> T2)?
'''

# <NO_EDIT>
statout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'stats']))
graphout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'svg']))
notesout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'notes']))

gammasfile = os.path.join(gcf.OUTPUT_DIR, 'lib234.gammas.tsv')
gammas = pd.read_csv(gammasfile, sep='\t', header=0, index_col=[0,1])
normalfile = os.path.join(gcf.OUTPUT_DIR, 'lib234.normal.tsv')
normal = pd.read_csv(normalfile, sep='\t', header=0, index_col=[0,1])
annosfile = os.path.join(gcf.OUTPUT_DIR, 'lib234.annos.tsv')
annos = pd.read_csv(annosfile, sep='\t', header=0, index_col=[0,1])
# </NO_EDIT>


# FILTER: Keep all for now
def selection_filter(inframe):
  # mask = annos.essential & annos.highfi
  return inframe # .loc[mask]

filteredgammas = selection_filter(gammas)


# GROUPING: NONE
# groups = filteredgammas.groupby(by=annos.gene_name, group_keys=True)
groups = None


# <NO_EDIT>
# Output Stastical summary of filtered/grouped data
with open(statout, 'w') as f:
  f.write('Post-filter statistics:' + '\n')
  f.write(str(filteredgammas.describe()) + '\n')
  f.write('\n\n')
  f.write('Head:' + '\n')
  f.write(str(filteredgammas.head()) + '\n')
  if groups is not None:
    f.write('\n\n')
    f.write('Groups:' + '\n')
    f.write(str(groups.describe()) + '\n')

# Output notes
with open(notesout, 'w') as f:
  f.write(QUESTION.format(**vars()))
# </NO_EDIT>


# Construct/output graph
fig = plt.figure(figsize=(6,6))
sizes=10
ax = sns.regplot('a1d3_a2d3', 'b1d3_b2d3', data=filteredgammas, fit_reg=False,
                 scatter_kws=dict(s=sizes, alpha=0.2, color='magenta'))

(start, stop) = (-1.6, 0.1)
ax.set_xlim(start, stop)
ax.set_ylim(start, stop)
ax.set(xlabel='gamma [No Drug]')
ax.set(ylabel='gamma [30 ng/mL Trimethoprim]')
plt.title('Growth from 5 - 7.5 doublings; No Drug vs. Heavy Drug')
plt.tight_layout()
plt.savefig(graphout)
plt.close(fig)
