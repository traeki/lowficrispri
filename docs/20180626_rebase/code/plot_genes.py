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
from train_models import build_feature_columns 
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

CODEFILE = pathlib.Path(__file__).name
STATFILE = (gcf.OUTPUT_DIR / CODEFILE).with_suffix('.stats.tsv')

_USE_RHOBIN = False

# --------------------

def apply_model(model, X_test):
  test_pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test, shuffle=False)
  preds = [x['predictions'][0] for x in model.predict(test_pred_input_func)]
  return preds

def plot_gene(group, sprrho, prsrho, measured, predicted, plotfile):
  if _USE_RHOBIN:
    rhobin = 'low'
    if sprrho > 0.3:
      rhobin = 'mid'
    if sprrho > 0.6:
      rhobin = 'high'
    plotfile = plotfile.with_suffix('.' + rhobin + '.png')
  else:
    plotfile = plotfile.with_suffix('.png')
  gene = group.iloc[0].gene
  plt.figure(figsize=(6,6))
  template = 'Predictions vs. Measurements\n{gene}'
  main_title_str = template.format(**vars())
  plt.title(main_title_str)
  g = plt.scatter(predicted, measured, s=3, alpha=0.5)
  plt.xlim(-1.2, 0.2)
  plt.ylim(-1.2, 0.2)
  plt.xlabel('Predicted γ')
  plt.ylabel('Measured γ')
  template = 'Spearman: {sprrho:.2f}, Pearson: {prsrho:.2f}'
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
    model = tf.estimator.LinearRegressor(feature_columns=feat_cols,
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
        grouper = X_eval.groupby('gene')
        threadlabel = '.'.join([modelkey, 'on', evalkey, poolname])
        plotdir_suffix = '.' + threadlabel + '.plots'
        plotdir = (gcf.OUTPUT_DIR / CODEFILE).with_suffix(plotdir_suffix)
        shutil.rmtree(plotdir, ignore_errors=True)
        plotdir.mkdir(parents=True, exist_ok=True)
        logging.info('Plotting Genes to {plotdir}...'.format(**vars()))
        for gene, group in grouper:
          exemplar = group.iloc[0]
          gene = exemplar.gene
          ext = '.' + gene + '.png'
          plotfile = (plotdir / CODEFILE).with_suffix(ext)
          predicted = (group.y_pred + 1) * group.parent
          measured = (group.y_meas + 1) * group.parent
          sprrho, sprpv = st.spearmanr(predicted, measured)
          prsrho, prspv = st.pearsonr(predicted, measured)
          plot_gene(group, sprrho, prsrho, measured, predicted, plotfile)
          result = dict()
          result['model'] = modelkey
          result['target'] = evalkey
          result['subset'] = poolname
          result['gene'] = gene
          result['sprpv'] = sprpv
          result['prspv'] = prspv
          result['sprrho'] = sprrho
          result['prsrho'] = prsrho
          results.append(pd.Series(result))
  results = pd.DataFrame(results)
  results.set_index(['model', 'target', 'subset'], inplace=True)
  logging.info('Saving statistics to {STATFILE}...'.format(**globals()))
  results.to_csv(STATFILE, sep='\t')


if __name__ == '__main__':
  main()
