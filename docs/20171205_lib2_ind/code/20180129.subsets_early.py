#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]

import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import seaborn as sns
import sys

import pooled.data_to_dataframe as dtd

import global_config as gcf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
At a high level, how did phenotypes differ between drugged and undrugged
samples over the early run (T0 -> T1)?
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


# FILTERS
def gene_filter(inframe, gene):
  mask = (annos.gene_name == gene)
  return inframe.loc[mask]

def control_filter(inframe):
  mask = annos.control
  return inframe.loc[mask]

def other_filter(inframe, genes):
  unmask = annos.control
  for gene in genes:
    unmask |= (annos.gene_name == gene)
  mask = ~unmask
  return inframe.loc[mask]

subsets = dict()
subsets['all'] = gammas
subsets['dfrA'] = gene_filter(gammas, 'dfrA')
subsets['murAA'] = gene_filter(gammas, 'murAA')
subsets['control'] = control_filter(gammas)
subsets['other'] = other_filter(gammas, ['dfrA', 'murAA'])

# GROUPING: NONE
# groups = filteredgammas.groupby(by=annos.gene_name, group_keys=True)
groups = None


# <NO_EDIT>
# Output Stastical summary of filtered/grouped data
with open(statout, 'w') as f:
  for key in subsets:
    f.write('Filter statistics [' + key + ']:' + '\n')
    filtered = subsets[key]
    f.write(str(filtered.describe()) + '\n')
    f.write('\n\n')
    f.write('Head [' + key + ']:' + '\n')
    f.write(str(filtered.head()) + '\n')
  if groups is not None:
    f.write('\n\n')
    f.write('Groups:' + '\n')
    f.write(str(groups.describe()) + '\n')

# Output notes
with open(notesout, 'w') as f:
  f.write(QUESTION.format(**vars()))
# </NO_EDIT>


# Construct/output graph

replicate_pairs = [('t0d3_a1d3', 't0d3_b1d3'),
                   ('t0d3_a1d3', 't0d3_c1d3'),
                   ('t0d3_b1d3', 't0d3_c1d3')]
labels = dict()
labels['t0d3_a1d3'] = 'No Drug - A'
labels['t0d3_b1d3'] = '30 ng/mL - B'
labels['t0d3_c1d3'] = '30 ng/mL - C'

rows = len(subsets)
cols = len(replicate_pairs)
gsize = 4
fig, axes = plt.subplots(rows, cols, figsize=(gsize*cols,gsize*rows),
                         sharex='col', sharey='row')
sizes=5

for i, key in enumerate(subsets):
  for j, (rep_x, rep_y) in enumerate(replicate_pairs):
    (start, stop) = (-1.6, 0.1)
    subset = subsets[key]
    ax = axes[i, j]
    ax.plot([start, stop], [start, stop], 'b--', linewidth=.5)
    sns.regplot(rep_x, rep_y, data=subset, fit_reg=False, ax=ax, label=key,
                scatter_kws=dict(s=sizes, alpha=0.2, color='magenta'))
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ax.set(xlabel='gamma [' + labels[rep_x] + ']')
    ax.set(ylabel='gamma [' + labels[rep_y] + ']')
    ax.legend()
    ax.label_outer()

fig.suptitle('Growth over 5 doublings')
fig.subplots_adjust(hspace=0, wspace=0)

# plt.tight_layout()
plt.savefig(graphout)
plt.close(fig)
