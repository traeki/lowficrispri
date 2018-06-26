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
OD_FRAME = os.path.join(DATA_DIR, '20171205.od.v.time.tsv')
COUNT_GLOB = os.path.join(DATA_DIR, '??d?_*L00*.fastq.counts')

# Background Data
ORIG_MAP = os.path.join(DATA_DIR, 'orig_map.tsv')
OLIGO_FILE = os.path.join(DATA_DIR, 'hawk1234.oligos')
BROAD_OLIGO_FILE = os.path.join(DATA_DIR, 'hawk12.oligos')
MURAA_OLIGO_FILE = os.path.join(DATA_DIR, 'hawk3.oligos')
DFRA_OLIGO_FILE = os.path.join(DATA_DIR, 'hawk4.oligos')
GENOME = os.path.join(DATA_DIR, 'bsu.NC_000964.gb')
TARGETFILE = os.path.join(DATA_DIR, 'lib234.targets.joined.tsv')
JMPBMK_ANNOS = os.path.join(DATA_DIR,
    './B._subtilis_essential_and_reduced-fitness_genes_20160121.xlsx')
REPLICATES = [['a0d1', 'a1d1', 'a2d1', 'a3d1'],
              ['b0d1', 'b1d1', 'b2d1', 'b3d1'],
              ['c0d1', 'c1d1', 'c2d1', 'c3d1'],
              ['a0d2', 'a1d2', 'a2d2', 'a3d2'],
              ['b0d2', 'b1d2', 'b2d2', 'b3d2'],
              ['c0d2', 'c1d2', 'c2d2', 'c3d2'],
              ['a0d3', 'a1d3', 'a2d3', 'a3d3'],
              ['b0d3', 'b1d3', 'b2d3', 'b3d3'],
              ['c0d3', 'c1d3', 'c2d3', 'c3d3']]
