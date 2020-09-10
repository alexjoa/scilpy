#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import glob
import json
import math
import os
from itertools import chain
import logging
import bisect
import functools

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
import numpy as np

from dipy.segment.clustering import qbx_and_merge
from nibabel.streamlines import ArraySequence
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def build_args_p():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('inputs', metavar='INPUT_FILES', nargs='+',
                   help='The list of result files')

    p.add_argument('output', metavar='OUTPUT_FILE',
                   help='The file where the dictionary is saved.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = build_args_p()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, args.inputs)
    assert_outputs_exist(parser, args, args.output)

    bundle_overlap = []

    for f in args.inputs:
        with open(f) as score_file:
            score = json.load(score_file)
            name = list(score['bundle_wise'].keys())
            tc_voxels = (score['bundle_wise'][str(name[0])]['tc_bundle_overlap'])
            missing_voxels = (score['bundle_wise'][str(name[0])]['tc_bundle_lacking'])
            bundle_overlap.append(float(tc_voxels/float(tc_voxels+missing_voxels)))

    final_BO_score = sum(bundle_overlap) / len(bundle_overlap)

    result = {
        "Ensemble_tractogram_Bundle_Overlap": final_BO_score
    }

    with open(args.output, 'w+') as of:
        json.dump(result, of)

if __name__ == "__main__":
    main()