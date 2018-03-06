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

from sklearn.decomposition import FastICA

import global_config as gcf
import sklearn_preproc as prep

import pdb
from IPython import embed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
What does it look like if we paint beaks in apple-to-apples base set (i.e. same
condition, same timespan)
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

prepped = prep.prep_for_sklearn(diffdata)

plotsets = list()
plotsets.append(('a0d1', 'a2d1'))
plotsets.append(('b0d1', 'b2d1'))
plotsets.append(('c0d1', 'c2d1'))
plotsets.append(('a0d2', 'a2d2'))
# plotsets.append(('a0d3', 'a2d3'))

subset = prepped[plotsets]

ica = FastICA(n_components=3, random_state=0)
ica.fit(subset.T)
axisnames = ['IC{0}'.format(i) for i in range(1, len(ica.mixing_.T)+1)]
rated_guides = pd.DataFrame(ica.mixing_,
                            columns=axisnames, index=prepped.index)
cutoff = 4*(rated_guides.std())
hits = rated_guides > cutoff

masked = prepped.copy()
masked['hits'] = hits.IC1

samples = masked.loc[masked.hits].index.tolist()
with open(os.path.join(gcf.OUTPUT_DIR, 'beaksamples.txt'), 'w') as f:
  f.write('\n'.join(samples))

plt.figure(figsize=(6,6))
sns.pairplot(masked,
             diag_kind='kde',
             vars=plotsets,
             hue='hits',
             plot_kws=dict(s=5, linewidth=0.5, alpha=0.2))
plt.suptitle(
    'gammas, pairwise plots [No Drug], in-condition "differentiators"'.format(**vars()),
    fontsize=16)
plt.tight_layout()
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
plt.savefig(graphflat)
plt.close()
