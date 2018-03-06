#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import numpy as np
import pandas as pd
# import pdb
import matplotlib.pyplot as plt
# SUPPRESS BOGUS LAPACK WARNING
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
# SUPPRESS BOGUS LAPACK WARNING

from sklearn import preprocessing

def remove_overlapping(data):
  template = '{tube}{timepoint}d{day}'
  adjacent = [(i, i+1) for i in range(3)]
  goodnames = list()
  for day in (1, 2, 3):
    for tube in ('a', 'b', 'c'):
      for pair in adjacent:
        namepair = list()
        for timepoint in pair:
          namepair.append(template.format(**vars()))
        goodnames.append(namepair)
  goodnames = [(x,y) for x, y in goodnames]
  return data[goodnames]

def widen_groups_(data):
  indexed = data.set_index(['variant', 'sample_s', 'sample_e'])
  return indexed.gamma.unstack(level=[1,2])

def clean_data_(data):
  imp = preprocessing.Imputer(strategy='median', axis=1)
  imp.fit(data)
  filled = imp.transform(data)
  return pd.DataFrame(filled, columns=data.columns, index=data.index)

def prep_for_sklearn(data):
  return clean_data_(widen_groups_(data))
