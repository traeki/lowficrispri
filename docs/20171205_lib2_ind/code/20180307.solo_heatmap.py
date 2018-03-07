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
from sklearn.preprocessing import StandardScaler

import global_config as gcf
import sklearn_preproc as prep

import pdb
from IPython import embed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

np.set_printoptions(precision=4, suppress=True)

PREFIX = os.path.splitext(os.path.basename(__file__))[0]

QUESTION = '''
What does it look like if we paint beaks in apple-to-apples base set
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

# PCA for variants can tell us which variants best explain the differences
# *among no drug controls*.  Best explainers should be real assholes.

# genemap = diffdata.set_index('variant').gene_name
# genemap = genemap.reset_index().drop_duplicates().set_index('variant')

indexed = solodata.set_index(['variant', 'sample'])
widened = indexed.log.unstack()
prepped = prep.clean_data_(widened)
scaler = StandardScaler(with_mean=True, with_std=False)
scaled = scaler.fit_transform(prepped)
scaledframe = pd.DataFrame(scaled,
                           index=prepped.index,
                           columns=prepped.columns)

cm = sns.clustermap(scaledframe.T, figsize=(12,12), yticklabels=1)
logging.info('Writing flat graph to {graphflat}'.format(**vars()))
cm.savefig(graphflat)
