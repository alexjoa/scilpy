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
                   help='The list of files that contain the ' +
                        'streamlines to operate on.')

    p.add_argument('output', metavar='OUTPUT_FILE',
                   help='The file where the dictionary is saved.')

    p.add_argument('--algo', default='bundle', choices=['bundle', 'voxel'],
                   help='Method used to identify parameters combinations that produces unique streamlines, bundle: uses'
                        ' quick bundle to identify unique bundle(small amount of streamlines). voxel: identifies'
                        ' streamlines passing trough unique voxel')

    p.add_argument('--top', default='10',
                   help='Represents the top n percent combinations that produces unique streamlines'
                        ' must be a value from 0 to 100')

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

    trk_names = []
    nb_input = []
    nb_unique_streamline = []

    if args.algo == 'bundle':

        # Load all input streamlines.
        trk_intervals = [] #fused streamlines intervals (i.e. from 0 to x the streamline comes from the first tracto)
        fused_streamlines = []

        for f in args.inputs:
            tmp_sft = load_tractogram_with_reference(parser, args, f)
            
            basename = os.path.basename(f)
            gt_id = int(basename[basename.find("gt")+2:basename.find("_tc")])-1
            while gt_id >= len(trk_names):
                trk_names.append([])
                trk_intervals.append([0])
                fused_streamlines.append([])

            trk_names[gt_id].append(basename[:basename.find(".trk")])

            fused_streamlines[gt_id].extend(tmp_sft.streamlines)

            tmp_val = trk_intervals[gt_id][-1] + len(tmp_sft)
            trk_intervals[gt_id].append(tmp_val)

        for row in trk_intervals:
            del row[0]

        for gt in range(len(trk_intervals)):

            thresholds = [32, 16, 8, 4, 2, 1]
            cluster_map = qbx_and_merge(ArraySequence(fused_streamlines[gt]), thresholds, 12, verbose=False)

            # identify clusters with "unique" streamlines
            nb_input.append(len(trk_intervals[gt]))
            unique_max = 0.01 * (len(fused_streamlines[gt]))
            unique_min = math.ceil(0.1 * unique_max)
            nb_unique_streamline.append([])
            nb_unique_streamline[gt] = [0] * nb_input[gt]

            for i in cluster_map.clusters:
                if unique_min < len(i) <= unique_max:
                    for j in i.indices:
                        t = bisect.bisect_left(trk_intervals[gt], j)
                        nb_unique_streamline[gt][t] = nb_unique_streamline[gt][t] + 1

    elif args.algo == 'voxel':

        all_masks = []
        for f in args.inputs:
            basename = os.path.basename(f)
            gt_id = int(basename[basename.find("gt") + 2:basename.find("_tc")]) - 1
            while gt_id >= len(trk_names):
                trk_names.append([])
                all_masks.append([])
            trk_names[gt_id].append(basename[:basename.find(".trk")])

            tmp_sft = load_tractogram_with_reference(parser, args, f)
            tmp_sft.to_vox()
            tmp_sft.to_corner()
            _, dimensions, _, _ = tmp_sft.space_attribute
            voxel_mask = compute_tract_counts_map(tmp_sft.streamlines, dimensions).astype(np.int16)
            voxel_mask[voxel_mask > 0] = 1
            all_masks[gt_id].append(voxel_mask)

        for gt in range(len(all_masks)):
            nb_input.append(len(all_masks[gt]))
            nb_unique_streamline.append([])
            for i in range(nb_input[gt]):
                masks_copy = copy.deepcopy(all_masks[gt])
                mask = masks_copy[i]
                del masks_copy[i]
                masks_copy.insert(0, mask)
                unique_voxel_mask = functools.reduce(lambda a, b: a - (a * b), masks_copy)
                non_null_voxel = np.count_nonzero(unique_voxel_mask == 1)
                nb_unique_streamline[gt].append(non_null_voxel)

    # find the params that produces the most unique true positives
    top_trk_dict = {}
    for gt in range(len(nb_input)):
        nb_trk_to_keep = math.ceil(0.01 * int(args.top) * nb_input[gt])
        top_trk_params = {}
        for i in range(0, nb_trk_to_keep):
            max_index = nb_unique_streamline[gt].index(max(nb_unique_streamline[gt]))
            nb_unique_streamline[gt][max_index] = 0

            # filename = os.path.expanduser('~') + '/Data/projet_stage/results/ensemble_tracking/' + trk_names[gt][
            #     max_index] + '.json'
            # with open(filename) as f:
            #     params = json.load(f)
            #     top_trk_params[str(i)] = params
            name = trk_names[gt][max_index]
            top_trk_params[str(i)] = {
                "trk_name": name,
                "seeding": "gm",
                "algo": "prob",
                "npv": name[name.find("npv") + 3:name.find("_theta")],
                "theta": name[name.find("theta") + 5:name.find("_step")],
                "step_size": name[name.find("step") + 4:name.find("_sphere")],
                "sphere": name[name.find("sphere") + 6:]
            }
            # os.system("python scil_compute_local_tracking.py $fodf $seeding_mask $tracking_mask retrack_npv100_theta${theta}_step${step_size}_sphere${sphere}.trk --algo prob --npv ${npv} --theta ${theta} --step ${step_size} --sphere ${sphere}  --compress 0.2 -f")

        top_trk_dict['gt' + str(gt + 1)] = top_trk_params

    with open(args.output, 'w+') as fp:
        json.dump(top_trk_dict, fp, indent=4)

    print("test")


if __name__ == "__main__":
    main()
