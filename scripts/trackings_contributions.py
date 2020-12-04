#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

import json
import glob
import pandas as pd
import os

# from scilpy.io.utils import (add_overwrite_arg,
#                              add_reference_arg,
#                              add_verbose_arg,
#                              assert_inputs_exist,
#                              assert_outputs_exist, add_sh_basis_args)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('input_file', help='Input files directory.')

    p.add_argument('output_file', help='Output files directory.')

    # add_overwrite_arg(p)

    return p


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    # assert_outputs_exist(parser, args, args.output_file)

    bundle = []
    input_files = list(glob.glob(str(args.input_file) + '*/*.json'))
    with open(input_files[0]) as f:
        score = json.load(f)
        bundle = list(score['bundle_wise']['true_connections'].keys())

    tracto_contrib = []
    total_tc = 0
    index = 0
    df = pd.DataFrame(columns=['bundle', 'track', 'tc_contribution', 'bundle_overlap'])
    for filename in input_files:
        with open(filename) as f:
            score = json.load(f)
            tracto_contrib.append([score['trk_filename'], score['tc_streamlines'], score['tractogram_overlap']])
            total_tc += score['tc_streamlines']

            for b in bundle:
                bundle_overlap = (score['bundle_wise']['true_connections'][str(b)]['tc_bundle_overlap_PCT'])
                tc_streamlines = (score['bundle_wise']['true_connections'][str(b)]['tc_streamlines'])
                df.loc[index] = [str(b), score['trk_filename'], tc_streamlines, bundle_overlap]
                index += 1

    tracto_contrib = list(zip(*tracto_contrib))
    

    labels = tracto_contrib[0]
    bo = tracto_contrib[2]
    y_pos = np.arange(len(labels))
    rb = mcolors.LinearSegmentedColormap.from_list("", [(0, "red"), (0.5, "yellow"), (1, "green")])
    plt.bar(y_pos, bo, color=rb(y_pos/len(labels)))
    plt.title("Tractograms Bundle Overlap")
    plt.xlabel('Tractograms')  
    plt.ylabel('Bundle Overlap')
    plt.xticks(y_pos, y_pos +1)
    plt.subplots_adjust(bottom=0.2, top=1)
    patches = [Patch(color=rb(k/len(labels)), label=str(k)+". "+labels[k]) for k in range(0, len(labels))]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.savefig(args.output_file + '_BO_plot.png', bbox_inches='tight')

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
        group.to_csv(args.output_file + str(bundle[i]) + ".csv", sep='\t', encoding='utf-8')

    with open(args.output_file + ".json", 'w+') as fp:
        json.dump(contribution, fp, indent=4, sort_keys=False)


if __name__ == '__main__':
    main()
