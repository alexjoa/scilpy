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

    pd_colums = ['track', 'TC', 'FC', 'NC', 'TC_ratio', 'FC_ratio', 'NC_ratio']
    df_scores = pd.DataFrame(columns=pd_colums)

    index = 0
    for filename in list(glob.glob(str(args.input_file) + '*/*.json')):
        with open(filename) as f:
            score = json.load(f)
            input_trk = score['trk_filename']
            true_connections = score['tc_streamlines']
            false_connections = score['fc_streamlines']
            no_connections = score['mc_streamlines']
            true_connections_ratio = score['tc_streamlines_ratio']
            false_connections_ratio = score['fc_streamlines_ratio']
            no_connections_ratio = score['mc_streamlines_ratio']

            df_scores.loc[index] = [input_trk, true_connections, false_connections, no_connections,
                                    true_connections_ratio, false_connections_ratio, no_connections_ratio]
            index = index + 1

    df_scores = df_scores.sort_values('TC_ratio')

    # filtering
    filtering = False
    if filtering:
        filter_TC_null = True
        filter_high_NC_ratio = True
        high_NC_ratio = 0.7
        filter_low_TC_ratio = True
        low_TC_ratio = 0.1

        if filter_TC_null:
            df_TC_null = df_scores[df_scores.TC == 0]
            df_filtered = df_scores[df_scores.TC != 0]

        if filter_high_NC_ratio:
            df_high_NC_ratio = df_scores[df_scores.NC_ratio > high_NC_ratio]
            df_filtered = df_scores[df_scores.NC_ratio < high_NC_ratio]

        if filter_low_TC_ratio:
            df_low_TC_ratio = df_scores[df_scores.TC_ratio < low_TC_ratio]
            df_filtered = df_scores[df_scores.TC_ratio > low_TC_ratio]

        resulting_trk_list = df_filtered['track'].tolist()

    df_scores.to_csv(args.output_file, sep='\t', encoding='utf-8')
    print(df_scores)

if __name__ == '__main__':
    main()
