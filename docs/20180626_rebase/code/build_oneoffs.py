#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

import global_config as gcf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

CODEFILE = pathlib.Path(__file__).name
ONEOFF_FILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.tsv')

# --------------------

def get_subvariants(variant, original):
  subvars = list()
  for i in range(len(variant)):
    if variant[i] != original[i]:
      fixone = variant[:i] + original[i] + variant[i+1:]
      if fixone != original:
        subvars.append(fixone)
  return subvars

def one_base_off(row):
  variant, original = row['variant'], row['original']
  for i in range(len(variant)):
    if variant[i] != original[i]:
      fixone = variant[:i] + original[i] + variant[i+1:]
      # that's the one mismatch IFF fixing it works, so we're done either way
      return fixone == original
  return False

def build_one_edit_pairs(omap_file):
  logging.info('Building one edit pairs from {omap_file}.'.format(**vars()))
  omap_df = pd.read_csv(omap_file, sep='\t')
  omap = dict(list(zip(omap_df.variant, omap_df.original)))
  synthetic_singles = list()
  for variant, original in omap.items():
    for sv in get_subvariants(variant, original):
      if sv in omap:
        synthetic_singles.append((variant, sv))
  synthetic_singles = pd.DataFrame(synthetic_singles,
                                   columns=['variant', 'original'])
  orig_singles = omap_df.loc[omap_df.apply(one_base_off, axis=1)]
  # oneoffs = pd.concat([synthetic_singles, orig_singles], axis=0)
  oneoffs = orig_singles
  oneoffs = oneoffs.reset_index()[['variant', 'original']]
  return oneoffs


def main():
  omap_file = gcf.DATA_DIR / 'orig_map.tsv'
  oneoffs = build_one_edit_pairs(omap_file)
  oneoffs.to_csv(ONEOFF_FILE, sep='\t')

if __name__ == '__main__':
  main()
