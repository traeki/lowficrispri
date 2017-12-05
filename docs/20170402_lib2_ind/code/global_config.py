#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]

import os.path
import pandas as pd

# Global Parameters
NORMAL_SIZE = 50 * 1000 * 1000
PSEUDO = 1
THRESHOLD = 50
SICKLINE = -0.4

# Data Paths
DIR_PREFIX = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(DIR_PREFIX, 'data')
SCRIPTS_DIR = os.path.join(DIR_PREFIX, 'code')
OUTPUT_DIR = os.path.join(DIR_PREFIX, 'output')

# Experimental Data
# Measured ODs for fitting the growth rate parameter "g"
OD_VS_TIME = os.path.join(DATA_DIR, '20170125.od.v.time.tsv')
# Identification of times (in minutes) associated with the samples.
TIMEPOINTS = pd.Series({0:0, 1:100, 2:195})
COUNT_GLOB = os.path.join(DATA_DIR, '??_*L003_R1_001.fastq.counts')

# Background Data
OLIGO_FILE = os.path.join(DATA_DIR, 'hawk12.oligos')
GENOME = os.path.join(DATA_DIR, 'bsu.NC_000964.gb')
TARGETFILE = os.path.join(DATA_DIR, 'lib2.targets.tsv')
JMPBMK_ANNOS = os.path.join(DATA_DIR,
    './B._subtilis_essential_and_reduced-fitness_genes_20160121.xlsx')
REPLICATES = [['A0', 'A1', 'A2'],
              ['B0', 'B1', 'B2']]
