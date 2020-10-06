#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np

import json
import glob
import pandas as pd
import os

from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist, add_sh_basis_args)

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('input_file', help='Input files directory.')

    p.add_argument('output_file', help='Output files directory.')

    add_overwrite_arg(p)

    return p



def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_outputs_exist(parser, args, args.output_file)

    bundle = []
    input_files = list(glob.glob(str(args.input_file) + '*/*.json'))
    with open(input_files[0]) as f:
        score = json.load(f)
        bundle = list(score['bundle_wise']['true_connections'].keys())

    tracto_tc_contribution = []
    total_tc = 0
    index = 0
    df = pd.DataFrame(columns=['bundle', 'track', 'tc_contribution', 'bundle_overlap'])
    for filename in input_files:
        with open(filename) as f:
            score = json.load(f)
            tracto_tc_contribution.append((score['trk_filename'], score['tc_streamlines']))
            total_tc += score['tc_streamlines']

            for b in bundle:
                bundle_overlap = (score['bundle_wise']['true_connections'][str(b)]['tc_bundle_overlap_PCT'])
                tc_streamlines = (score['bundle_wise']['true_connections'][str(b)]['tc_streamlines'])
                df.loc[index] = [str(b), score['trk_filename'], tc_streamlines, bundle_overlap]
                index += 1

    df = df.sort_values('bundle_overlap', ascending=False)
    grouped = df.groupby('bundle')

    contribution = {}
    for i in range(len(bundle)):
        group = grouped.get_group(str(bundle[i])).copy().reset_index(drop=True)
        total_tc_str = group['tc_contribution'].sum()
        group['tc_contribution'] = group['tc_contribution'].div(total_tc_str)
        tmp_dict = {"track": group.loc[0, 'track'], "tc_contribution": group.loc[0, 'tc_contribution'],
                    "bundle_overlap": group.loc[0, 'bundle_overlap']}
        contribution[group.loc[0, 'bundle']] = tmp_dict
        group.to_csv(args.output_file + str(bundle[i]) + "_contrib.csv", sep='\t', encoding='utf-8')

    with open(args.output_file + "top_contribution.json", 'w+') as fp:
        json.dump(contribution, fp, indent=4, sort_keys=False)


if __name__ == '__main__':
    main()
