#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]
import collections
import itertools
import logging
import pathlib
import re
import shutil
import sys

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import tensorflow as tf

import global_config as gcf
from compute_gammas import read_preprocessed_data
from compute_gammas import RAW_FILE
from train_models import TRAIN_FILE
from train_models import TEST_FILE
from train_models import MODEL_DIRS
from train_models import GUIDESETS
from train_models import NN_LAYERS
from train_models import build_feature_columns

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

CODEFILE = pathlib.Path(__file__).name
STATFILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.stats.tsv')

# --------------------

def apply_model(model, X_test):
  test_pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test, shuffle=False)
  preds = [x['predictions'][0] for x in model.predict(test_pred_input_func)]
  return preds

def plot_family(name, group, rho, pv, measured, predicted, plotfile):
  gene = group.iloc[0].gene
  family = name
  plt.figure(figsize=(6,6))
  template = 'Predictions vs. Measurements\n{gene}: {family}'
  main_title_str = template.format(**vars())
  plt.title(main_title_str)
  g = plt.scatter(measured, predicted, s=3)
  plt.xlim(-1.2, 0.2)
  plt.ylim(-1.2, 0.2)
  plt.xlabel('Measured')
  plt.ylabel('Predicted')
  template = 'Spearman: {rho:.2f}, P-value: {pv:.2f}'
  plt.text(-1.1, 0.1, template.format(**vars()))
  plt.tight_layout()
  plt.savefig(plotfile)
  plt.close()

def main():
  # read in TEST and TRAIN
  data = dict()
  data['train'] = pd.read_csv(TRAIN_FILE, sep='\t')
  data['train'].fillna('NA', inplace=True)
  data['test'] = pd.read_csv(TEST_FILE, sep='\t')
  data['test'].fillna('NA', inplace=True)
  data['all'] = pd.concat([data['train'], data['test']], axis=0)
  # TODO(jsh): Break out genemap into separate computation
  rawdata = read_preprocessed_data(RAW_FILE)
  gene_map = rawdata.reset_index()[['variant', 'gene_name']].drop_duplicates()
  gene_map.set_index('variant', inplace=True)
  feat_cols = build_feature_columns(data['train'])
  keys = itertools.product(GUIDESETS, GUIDESETS, data)
  results = list()
  # loop over models
  for modelkey in GUIDESETS:
    # load model from MODEL_DIR
    model_dir = MODEL_DIRS[modelkey]
    model = tf.estimator.DNNRegressor(feature_columns=feat_cols,
                                      hidden_units=NN_LAYERS,
                                      model_dir=model_dir)
    for evalkey, guideset in GUIDESETS.items():
      for poolname, pool in data.items():
        template = 'PLOTTING {modelkey} MODEL of {evalkey}/{poolname}...'
        logging.info(template.format(**vars()))
        eval_check = lambda x: x in guideset
        eval_mask = pool.reset_index().variant.apply(eval_check)
        eval_mask.index = pool.index
        eval_data = pool.loc[eval_mask]
        data_columns = set(pool.columns) - set('y_meas')
        output_columns = set(['y_meas'])
        X_eval = eval_data[list(data_columns)].reset_index(drop=True)
        gene = X_eval.reset_index().variant.map(gene_map.gene_name)
        gene.index = X_eval.index
        X_eval['gene'] = gene
        y_eval = eval_data[list(output_columns)].reset_index(drop=True)
        preds = apply_model(model, X_eval)
        X_eval['y_meas'] = y_eval.y_meas
        X_eval['y_pred'] = preds
        grouper = X_eval.groupby('family')
        threadlabel = '.'.join([modelkey, 'on', evalkey, poolname])
        plotdir_suffix = '.' + threadlabel + '.plots'
        plotdir = (gcf.OUTPUT_DIR / CODEFILE).with_suffix(plotdir_suffix)
        shutil.rmtree(plotdir, ignore_errors=True)
        plotdir.mkdir(parents=True, exist_ok=True)
        logging.info('Plotting Families to {plotdir}...'.format(**vars()))
        for family, group in grouper:
          exemplar = group.iloc[0]
          gene = exemplar.gene
          ext = '.' + gene + '.' + family + '.png'
          plotfile = (plotdir / CODEFILE).with_suffix(ext)
          measured = (group.y_meas + 1) * group.parent
          predicted = (group.y_pred + 1) * group.parent
          rho, pv = st.spearmanr(measured, predicted)
          plot_family(family, group, rho, pv, measured, predicted, plotfile)
          result = dict()
          result['model'] = modelkey
          result['target'] = evalkey
          result['subset'] = poolname
          result['family'] = family
          result['gene'] = gene
          result['rho'] = rho
          result['p_value'] = pv
          results.append(pd.Series(result))
  results = pd.DataFrame(results)
  results.set_index(['model', 'target', 'subset', 'family'], inplace=True)
  logging.info('Saving statistics to {STATFILE}...'.format(**globals()))
  results.to_csv(STATFILE, sep='\t')


if __name__ == '__main__':
  main()
