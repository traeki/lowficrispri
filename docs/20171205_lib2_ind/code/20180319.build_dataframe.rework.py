#!/usr/bin/env python

# Author: John Hawkins (jsh) [really@gmail.com]

import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import pprint
import seaborn as sns
import sys

import global_config as gcf

from Bio import SeqIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class Error(Exception):
  pass

class SampleError(Error):
  pass

def get_locus_lengths(genome):
  mapping = dict()
  g = SeqIO.parse(genome, 'genbank')
  for chrom in g:
    for feature in chrom.features:
      if feature.type == 'gene':
        locus = feature.qualifiers['locus_tag'][0]
        mapping[locus] = abs(feature.location.end - feature.location.start)
  return mapping


def import_subtilis_annotation_excel(jmpbmk_annos):
  essential_data = pd.read_excel(jmpbmk_annos)
  essential_data.rename(columns={
    essential_data.columns[9]: 'lowfit',
    essential_data.columns[11]: 'essential'}, inplace=True)
  essential_data = essential_data[[
    'locus_tag', 'gene_name', 'lowfit',
    'essential', 'small_transformation']]
  essential_data.lowfit = (essential_data.lowfit == 1)
  essential_data['small'] = (essential_data.small_transformation == 'yes')
  essential_data.drop('small_transformation', 1, inplace=True)
  essential_data['sick'] = essential_data.small | essential_data.lowfit
  essential_data.essential = (essential_data.essential == 'essential')
  return essential_data


def import_guide_locus_relation(tsv_file):
  target_data = pd.read_csv(tsv_file, sep='\t',
      names=['locus_tag', 'offset', 'variant', 'pam',
             'chrom', 'start', 'end', 'abs_dir', 'rel_dir',
             'weakness', 'specificity'])
  target_data = target_data[['locus_tag', 'variant']]
  target_data.drop_duplicates(inplace=True)
  return target_data


def get_guide_offsets(tsv_file):
  target_data = pd.read_csv(tsv_file, sep='\t',
      names=['locus_tag', 'offset', 'variant', 'pam',
             'chrom', 'start', 'end', 'abs_dir', 'rel_dir',
             'weakness', 'specificity'])
  target_data = target_data[['variant', 'offset']]
  target_data.offset = target_data.offset.astype(int)
  target_data.drop_duplicates(inplace=True)
  target_data.reset_index(inplace=True, drop=True)
  return target_data


def get_annotations(data, genome, tsv_file, jmpbmk_annos):
  essential_data = import_subtilis_annotation_excel(jmpbmk_annos)
  locus_len_map = get_locus_lengths(genome)
  guide_offset_map = get_guide_offsets(tsv_file)
  guide_locus_rel = import_guide_locus_relation(tsv_file)
  markup = pd.merge(guide_locus_rel, essential_data,
                    on='locus_tag', how='inner')
  markup.reset_index(drop=True, inplace=True)
  # Outer join to pick up control variants
  variants = data[['variant']].drop_duplicates()
  annotations = pd.merge(variants, markup,
                         how='outer', on='variant')

  annotations.locus_tag.fillna('CONTROL', inplace=True)
  annotations.gene_name.fillna('CONTROL', inplace=True)
  annotations['gene_len'] = annotations.locus_tag.map(locus_len_map)
  annotations = pd.merge(annotations, guide_offset_map,
                         on='variant', how='left')
  annotations.lowfit.fillna(False, inplace=True)
  annotations.small.fillna(False, inplace=True)
  annotations.sick.fillna(False, inplace=True)
  annotations.essential.fillna(False, inplace=True)
  annotations['control'] = annotations.locus_tag == 'CONTROL'
  duplicated = data.duplicated(subset='original', keep=False)
  invariant = data.original == data.variant
  annotations['offstrand'] = ~duplicated & ~annotations.control
  annotations['highfi'] = duplicated & invariant
  annotations['lowfi'] = duplicated & ~invariant
  annotations['dead'] = annotations.essential & annotations.highfi
  return annotations


def read_countfiles(sample_pattern):
  """Read the count files into columns."""
  def get_sample(countfile):
    base = os.path.basename(countfile).split('.')[0]
    frame = pd.read_csv(countfile, sep='\t', names=['variant', 'raw'])
    frame.raw = frame.raw.astype('int')
    sample = base.split('_')[0]
    if sample.startswith('t'):
      alts = list()
      for tube in ['a', 'b', 'c']:
        alias = tube + sample[1:]
        aliased = frame.copy()
        aliased['sample'] = alias
        alts.append(aliased)
      return pd.concat(alts, axis='index')
    else:
      frame['sample'] = sample
      return frame
  samples = [get_sample(countfile) for countfile in glob.glob(sample_pattern)]
  grid = pd.concat(samples, axis='index')
  grid.reset_index(drop=True, inplace=True)
  return grid


logging.info('Reading countfiles: {gcf.COUNT_GLOB}'.format(**vars()))
counts = read_countfiles(gcf.COUNT_GLOB)
logging.info('Reading OD data from: {gcf.OD_FRAME}'.format(**vars()))
od_data = pd.read_csv(gcf.OD_FRAME, sep='\t')
logging.info('Broadcasting OD data...'.format(**vars()))
data = pd.merge(counts, od_data, on='sample', how='left')
logging.info('Reading original/variant map: {gcf.ORIG_MAP}'.format(**vars()))
orig_map = pd.read_csv(gcf.ORIG_MAP, sep='\t')
logging.info('Broadcasting orig/variant map...'.format(**vars()))
data = pd.merge(data, orig_map, on='variant', how='left')
logging.info('Building additional annotations...'.format(**vars()))
annos = get_annotations(data, gcf.GENOME, gcf.TARGETFILE, gcf.JMPBMK_ANNOS)
logging.info('Broadcasting annotations...'.format(**vars()))
data = pd.merge(data, annos, on='variant', how='left')
data['log'] = np.log2(data.raw.clip(1))
rawlogfile = os.path.join(gcf.OUTPUT_DIR, 'lib234.rawlogs.tsv')
logging.info('Writing raw data to {rawlogfile}...'.format(**vars()))
data.to_csv(rawlogfile, sep='\t')
