#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]

import colorcet
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import seaborn as sns
import sys

import pooled.data_to_dataframe as dtd
import global_config as gcf


# Generate raw counts DataFrame
counts = dtd.pull_in_data(gcf.COUNT_GLOB, gcf.OLIGO_FILE)
normal = dtd.normalize_counts(counts, gcf.NORMAL_SIZE, pseudocount=gcf.PSEUDO)

# Build parallel annotations DataFrame based on targets file and genome
annos = dtd.get_annotations(counts, gcf.GENOME, gcf.TARGETFILE, gcf.JMPBMK_ANNOS)

# Compute gammas
logs = dtd.log_counts(normal)
pairs = dtd.pairlist(gcf.REPLICATES)
diffs = dtd.compute_differences(counts, logs, pairs, gcf.THRESHOLD, gcf.PSEUDO)
g = dtd.compute_g(gcf.OD_VS_TIME)
t = gcf.TIMEPOINTS
nullmedian = diffs.loc[annos.control].median()
gammas = dtd.compute_gammas(diffs, g, t, nullmedian)

# Record checkpoint files for [normalized] counts, annotations, and gammas
gammas.to_csv(os.path.join(gcf.OUTPUT_DIR, 'lib2.gammas.tsv'), sep='\t')
annos.to_csv(os.path.join(gcf.OUTPUT_DIR, 'lib2.annos.tsv'), sep='\t')
normal.to_csv(os.path.join(gcf.OUTPUT_DIR, 'lib2.normal.tsv'), sep='\t')
