#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com] import itertools
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
And now, what are the genes most impacted?
'''

# <NO_EDIT>
statout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'stats']))
graphout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'svg']))
graphflat = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'png']))
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

A = 't0d3_a3d3'
B = 't0d3_b3d3'
C = 't0d3_c3d3'

mean_drug = gammas[[B, C]].mean(axis='columns')
std_drug = gammas[[B, C]].std(axis='columns')
orth_dist = gammas[A].subtract(mean_drug).abs() / np.sqrt(2)

nanmask = mean_drug.isnull() | std_drug.isnull()
spl = UnivariateSpline(mean_drug.loc[~nanmask], std_drug.loc[~nanmask])

nodrug_sigma = spl(gammas[A].add(mean_drug)/2)

z = orth_dist.divide(nodrug_sigma)

p_cutoff = .05 / len(z)
zb = st.norm.ppf(1 - (p_cutoff/2))

print annos.loc[z > zb].gene_name.value_counts().head(20)
