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
How many guides had gammas between {gcf.SICKLINE} and the nullset cutoff for
each gene?  I.e. what is the distribution/coverage of "useful" guides?
'''

statout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'stats']))
graphout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'svg']))
notesout = os.path.join(gcf.OUTPUT_DIR, '.'.join([PREFIX, 'notes']))

gammasfile = os.path.join(gcf.OUTPUT_DIR, 'lib2.gammas.tsv')
gammas = pd.read_csv(gammasfile, sep='\t', header=0, index_col=[0,1])
normalfile = os.path.join(gcf.OUTPUT_DIR, 'lib2.normal.tsv')
normal = pd.read_csv(normalfile, sep='\t', header=0, index_col=[0,1])
annosfile = os.path.join(gcf.OUTPUT_DIR, 'lib2.annos.tsv')
annos = pd.read_csv(annosfile, sep='\t', header=0, index_col=[0,1])

def selection_filter(inframe):
  nullstd = gammas[annos.control].std()
  zeroline = -nullstd*2
  tally = (gammas >= gcf.SICKLINE) & (gammas < zeroline)
  mask = annos.essential
  return tally.loc[mask, 'B0_B2']

# Filter to subset of interest
filteredgammas = selection_filter(gammas)

# Group data into units to be graphed
groups = filteredgammas.groupby(by=annos.gene_name, group_keys=True)

# Output Stastical summary of filtered/grouped data
with open(statout, 'w') as f:
  f.write('Post-filter statistics:' + '\n')
  f.write(str(filteredgammas.describe()) + '\n')
  f.write('\n\n')
  f.write('Head:' + '\n')
  f.write(str(filteredgammas.head()) + '\n')
  f.write('\n\n')
  f.write('Groups:' + '\n')
  f.write(str(groups.describe()) + '\n')


# Output notes
with open(notesout, 'w') as f:
  f.write(QUESTION.format(**vars()))

# Construct/output graph
genecounts = groups.aggregate(np.sum)
sns.distplot(genecounts, kde=False, bins=np.array(range(55))-0.5)
plt.xlabel('# of "useful" guides\n'
           '(gamma outside null set, '
               'inside {gcf.SICKLINE} cutoff)'.format(**vars()))
plt.ylabel('# of genes in bin')
plt.title('Distribution of "useful" guide counts per essential gene')
plt.tight_layout()
plt.savefig(graphout)
