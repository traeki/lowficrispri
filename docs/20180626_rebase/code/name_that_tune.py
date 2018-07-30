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

_RAWDATA = read_preprocessed_data(RAW_FILE)
# TODO(jsh): Break out genemap into separate computation
GENE_MAP = _RAWDATA.reset_index()[['variant', 'gene_name']]
GENE_MAP = GENE_MAP.drop_duplicates()
GENE_MAP.set_index('variant', inplace=True)
OFFSETS = _RAWDATA.reset_index()[['variant', 'gene_len', 'offset']]
OFFSETS = OFFSETS.drop_duplicates()
OFFSETS.set_index('variant', inplace=True)

MAX_SAMPLE = 50
SAMPLE_STEP = 5
TRIALS = 10


# --------------------

def apply_model(model, X_test):
  test_pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test, shuffle=False)
  preds = [x['predictions'][0] for x in model.predict(test_pred_input_func)]
  return preds

def downsample_gene(group):
  gene = group.iloc[0].gene
  statblocks = list()
  statblocks.append(downsample_stats(group, gene, gene))
  subgrouper = group.groupby('original')
  for name, subgroup in subgrouper:
    statblocks.append(downsample_stats(subgroup, gene, name))
  return pd.concat(statblocks, axis=0)

def downsample_stats(group, gene, subset):
  results = list()
  max_group_sample = min(MAX_SAMPLE, group.shape[0])
  for k in range(SAMPLE_STEP, max_group_sample+1, SAMPLE_STEP):
    for i in range(TRIALS):
      trialset = group.sample(n=k)
      sprrho, sprpv = st.spearmanr(trialset.pred_gam, trialset.meas_gam)
      prsrho, prspv = st.pearsonr(trialset.pred_gam, trialset.meas_gam)
      result = dict()
      result['subset'] = subset
      result['sample_n'] = k
      result['gene'] = gene
      result['sprrho'] = sprrho
      result['prsrho'] = prsrho
      result['sprpv'] = sprpv
      result['prspv'] = prspv
      results.append(pd.Series(result))
  results = pd.DataFrame(results)
  return results

def plot_gene(group, plotfile):
  gene = group.iloc[0].gene
  plt.figure(figsize=(6,6))
  template = 'Stats by Sample Size'
  main_title_str = template.format(**vars())
  plt.title(main_title_str)
  plt.xlim(0, 50)
  plt.ylim(-1, 1)
  plt.xlabel('Sample Size')
  plt.ylabel('Spearman')
  sns.pointplot(data=group, x='sample_n', y='sprrho', hue='subset')
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
  feat_cols = build_feature_columns(data['train'])
  keys = itertools.product(GUIDESETS, GUIDESETS, data)
  results = list()
  # loop over models
  for modelkey in GUIDESETS:
    # TODO(jsh): horrible hack to speed testing loop
    if modelkey is not 'all':
      continue
    # load model from MODEL_DIR
    model_dir = MODEL_DIRS[modelkey]
    model = tf.estimator.LinearRegressor(feature_columns=feat_cols,
                                         model_dir=model_dir)
    for evalkey, guideset in GUIDESETS.items():
      for poolname, pool in data.items():
        # TODO(jsh): horrible hack to speed testing loop
        if poolname is not 'all' or evalkey is not 'all':
          continue
        template = 'PLOTTING {modelkey} MODEL of {evalkey}/{poolname}...'
        logging.info(template.format(**vars()))
        eval_check = lambda x: x in guideset
        eval_mask = pool.reset_index().variant.apply(eval_check)
        eval_mask.index = pool.index
        eval_data = pool.loc[eval_mask]
        output_columns = set(['y_meas'])
        data_columns = set(pool.columns) - output_columns
        X_eval = eval_data[list(data_columns)].reset_index(drop=True)
        gene = X_eval.reset_index().variant.map(GENE_MAP.gene_name)
        gene.index = X_eval.index
        X_eval['gene'] = gene
        y_eval = eval_data[list(output_columns)].reset_index(drop=True)
        preds = apply_model(model, X_eval)
        allcols = pd.concat([X_eval, y_eval], axis=1)
        allcols['y_pred'] = preds
        allcols['pred_gam'] = (allcols.y_pred + 1) * allcols.parent
        allcols['meas_gam'] = (allcols.y_meas + 1) * allcols.parent
        threadlabel = '.'.join([modelkey, 'on', evalkey, poolname])
        plotdir_suffix = '.' + threadlabel + '.plots'
        plotdir = (gcf.OUTPUT_DIR / CODEFILE).with_suffix(plotdir_suffix)
        shutil.rmtree(plotdir, ignore_errors=True)
        plotdir.mkdir(parents=True, exist_ok=True)
        logging.info('Computing downsample stats...'.format(**vars()))
        grouper = allcols.groupby('gene', group_keys=False)
        allstats = grouper.apply(downsample_gene)
        stat_grouper = allstats.groupby('gene', group_keys=False)
        logging.info('Plotting Genes to {plotdir}...'.format(**vars()))
        for gene, group in stat_grouper:
          exemplar = group.iloc[0]
          gene = exemplar.gene
          ext = '.' + gene + '.png'
          plotfile = (plotdir / CODEFILE).with_suffix(ext)
          plot_gene(group, plotfile)
        import IPython
        IPython.embed()


if __name__ == '__main__':
  main()
