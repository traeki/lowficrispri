#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import collections
import itertools
import logging
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

import global_config as gcf
from compute_gammas import read_preprocessed_data
from compute_gammas import GAMMA_FILE
from compute_gammas import RAW_FILE
from build_oneoffs import ONEOFF_FILE

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

CODEFILE = pathlib.Path(__file__).name
RELATIVE_FILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.tsv')
PARENT_FILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.parent.tsv')
CHILD_FILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.child.tsv')

# --------------------

def get_parent_child_gammas(oneoffs, solo_gammas):
  flatspans = solo_gammas.stack(level=0)
  flatspans = flatspans.reset_index().drop('sid', axis=1)
  parent_gammas = pd.merge(oneoffs, flatspans,
                           left_on='original', right_on='variant',
                           how='left', suffixes=('', '_orig'))
  parent_gammas.drop('variant_orig', axis=1, inplace=True)
  parent_gammas.set_index(['variant', 'original'], inplace=True)
  child_gammas = pd.merge(oneoffs, flatspans,
                          left_on='variant', right_on='variant',
                          how='left')
  child_gammas.set_index(['variant', 'original'], inplace=True)
  return parent_gammas, child_gammas

def compute_relative_gammas(rawdata, solo_gammas, parent_gammas, child_gammas):
  logging.info('Computing relative gammas.'.format(**vars()))
  rawdata = rawdata.set_index(['variant', 'sid', 'tp'])
  rawdata = rawdata.loc[~rawdata.index.duplicated()]
  control_mask = rawdata.control.unstack(level=[-2,-1]).iloc[:, 0]
  control_mask.name = 'control'
  contspans = solo_gammas.loc[control_mask].stack(level=0)
  contspans = contspans.reset_index().drop('sid', axis=1)
  geodelt_gammas = (child_gammas / parent_gammas) - 1
  filtered = geodelt_gammas.where(parent_gammas.abs() > (contspans.std()*10))
  return filtered


def main():
  rawdata = read_preprocessed_data(RAW_FILE)
  solo_gammas = pd.read_csv(GAMMA_FILE, sep='\t', header=[0,1], index_col=0)
  oneoffs = pd.read_csv(ONEOFF_FILE, sep='\t')
  parent_gammas, child_gammas = get_parent_child_gammas(oneoffs, solo_gammas)
  relative_gammas = compute_relative_gammas(rawdata, solo_gammas,
                                            parent_gammas, child_gammas)
  relative_gammas.to_csv(RELATIVE_FILE, sep='\t')
  parent_gammas.to_csv(PARENT_FILE, sep='\t')
  child_gammas.to_csv(CHILD_FILE, sep='\t')

if __name__ == '__main__':
  main()
