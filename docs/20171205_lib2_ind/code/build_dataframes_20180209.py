#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import seaborn as sns
import sys

import global_config as gcf


# Generate raw counts DataFrame
# TODO(jsh): read in dataframes mapping
# TODO(jsh): -> day / tube / time / od / sample
od_data = pd.DataFrame.from_csv(gcf.OD_FRAME, sep='\t')
# TODO(jsh): -> sample / guide / raw
# TODO(jsh): -> guide / parent
# TODO(jsh): -> guide / anno1 / ... / annoN
annos = dtd.get_annotations(counts, gcf.GENOME, gcf.TARGETFILE, gcf.JMPBMK_ANNOS)
# TODO(jsh): join dataframes (drop sample name)
counts = dtd.pull_in_data(gcf.COUNT_GLOB, gcf.OLIGO_FILE)
# TODO(jsh): add 'normal' column
normal = dtd.normalize_counts(counts, gcf.NORMAL_SIZE, pseudocount=gcf.PSEUDO)
# TODO(jsh): add 'log_norm' column
logs = dtd.log_counts(normal)

pairs = dtd.pairlist(gcf.SAMPLES)

# TODO(jsh): for each pair
# TODO(jsh): -> get start/end
# TODO(jsh): -> compute duration
# TODO(jsh): -> day / tube / time(start) / time(end) / t / guide / diff
# TODO(jsh): -> threshold directly on pseudo-raw counts?
# TODO(jsh): Q: Can we merge PSEUDO into THRESHOLD?
diffs = dtd.compute_differences(counts, logs, pairs, gcf.THRESHOLD, gcf.PSEUDO)

# TODO(jsh): groupby (day/tube/samples)
# TODO(jsh): -> compute median of controls
nullmedian = diffs.loc[annos.control].median()
# TODO(jsh): -> recentered <- translate median to zero
# TODO(jsh): -> polyfit day/tube (time, od)
# TODO(jsh): -> rescaled <- recentered * g**-1 * t**-1

# TODO(jsh): save (2) dataframes to disk

def g_fit(start, end):
  startmask = (start[:1] + '_' + start[2:])
  endmask = (end[:1] + '_' + end[2:])
  assert startmask == endmask
  g, _ = np.polyfit(gcf.OD_FRAME[startmask].dropna().index,
                    gcf.OD_FRAME[startmask].dropna(), 1)
  return g

def compute_gammas(pairdiff, nullmedian):
  start, end = pairdiff.name().split('_')
  g_fit(pairdiff.name())
  duration = gcf.SAMPLES(
  diff_t = diffs.columns.map(name_to_time).values
  recentered = diffs - nullmedian
  rescaled = recentered.divide(g, axis='columns').divide(diff_t, axis='columns')
  return rescaled


g = pairs.map(g_fit, axis='columns')
t1 = gcf.TIMEPOINTS_D1
g2 = dtd.compute_g(gcf.OD_VS_TIME_D2)
t2 = gcf.TIMEPOINTS_D2
g3 = dtd.compute_g(gcf.OD_VS_TIME_D3)
t3 = gcf.TIMEPOINTS_D3

gammas = dtd.compute_gammas(diffs, g1, t1, nullmedian)


# Record checkpoint files for [normalized] counts, annotations, and gammas
gammas.to_csv(os.path.join(gcf.OUTPUT_DIR, 'lib234.gammas.tsv'), sep='\t')
annos.to_csv(os.path.join(gcf.OUTPUT_DIR, 'lib234.annos.tsv'), sep='\t')
normal.to_csv(os.path.join(gcf.OUTPUT_DIR, 'lib234.normal.tsv'), sep='\t')
