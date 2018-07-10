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
import tensorflow as tf

import global_config as gcf

from train_models import TRAIN_FILE
from train_models import TEST_FILE
from train_models import MODEL_DIRS
from train_models import GUIDESETS
from train_models import build_feature_columns

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

CODEFILE = pathlib.Path(__file__).name

EVALFILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.tsv')

# --------------------

def main():
  # read in TEST and TRAIN
  data = dict()
  data['train'] = pd.read_csv(TRAIN_FILE, sep='\t')
  data['train'].fillna('NA', inplace=True)
  data['test'] = pd.read_csv(TEST_FILE, sep='\t')
  data['test'].fillna('NA', inplace=True)
  data['all'] = pd.concat([data['train'], data['test']], axis=0)
  feat_cols = build_feature_columns(data['train'])
  keys = itertools.product(GUIDESETS, GUIDESETS, data)
  results = list()
  # loop over models
  for modelkey in GUIDESETS:
    # load model from MODEL_DIR
    model_dir = MODEL_DIRS[modelkey]
    model = tf.estimator.LinearRegressor(feature_columns=feat_cols,
                                         model_dir=model_dir)
    for evalkey, guideset in GUIDESETS.items():
      for poolname, pool in data.items():
        template = 'EVALUATING {modelkey} MODEL on {evalkey}/{poolname}...'
        logging.info(template.format(**vars()))
        eval_check = lambda x: x in guideset
        eval_mask = pool.reset_index().variant.apply(eval_check)
        eval_mask.index = pool.index
        eval_data = pool.loc[eval_mask]
        data_columns = set(pool.columns) - set('y_meas')
        output_columns = set(['y_meas'])
        X_eval = eval_data[list(data_columns)].reset_index(drop=True)
        y_eval = eval_data[list(output_columns)].reset_index(drop=True)
        eval_input_func = tf.estimator.inputs.pandas_input_fn(
            x=X_eval, y=y_eval,
            batch_size=10, num_epochs=1,
            shuffle=False)
        eval_out = model.evaluate(input_fn=eval_input_func)
        result = dict()
        result['model'] = modelkey
        result['target'] = evalkey
        result['subset'] = poolname
        result.update(eval_out)
        results.append(pd.Series(result))
        logging.info('{eval_out}'.format(**vars()))
  results = pd.DataFrame(results)
  results.set_index(['model', 'target', 'subset'], inplace=True)
  results.to_csv(EVALFILE, sep='\t')

if __name__ == '__main__':
  main()
