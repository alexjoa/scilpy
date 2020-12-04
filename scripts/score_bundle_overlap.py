#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram
from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_verbose_arg,
                             add_json_args,
                             add_reference_arg, assert_output_dirs_exist_and_empty)
import numpy as np

import itertools
from scilpy.image.operations import intersection, difference
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.reproducibility_measures import (get_endpoints_density_map,
                                                           compute_dice_voxel)
from scilpy.segment.streamlines import filter_grid_roi
import nibabel as nib
from sklearn.cluster import KMeans
import json
from dipy.tracking.utils import length
from scilpy.utils.streamlines import (perform_streamlines_operation,
                                      difference)
from scilpy.tractanalysis.features import remove_loops_and_sharp_turns
import os
from scilpy.utils.filenames import split_name_with_nii
from nibabel.streamlines import ArraySequence
import pprint
from time import time


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Compute bundle centroid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('in_tractogram',
                   help='')
    p.add_argument('gt_bundles', nargs='+',
                   help='')
    p.add_argument('--gt_endpoints', nargs='+',
                   help='')

    p.add_argument('--gt_tails', nargs='+',
                   help='')
    p.add_argument('--gt_heads', nargs='+',
                   help='')

    # p.add_argument('--dilate_endpoints', metavar='NB_PASS',
    #                help='heuristic')

    p.add_argument('--gt_config', metavar='FILE',
                   help='heuristic')
    p.add_argument('--out_dir', default='gt_out/',
                   help='heuristic')

    p.add_argument('--wrong_path_as_separate', action='store_true',
                   help='heuristic')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_json_args(p)

    return p


def split_heads_tails_kmeans(data):
    X = np.argwhere(data)
    k_means = KMeans(n_clusters=2).fit(X)
    mask_1 = np.zeros(data.shape)
    mask_2 = np.zeros(data.shape)

    mask_1[tuple(X[np.where(k_means.labels_ == 0)].T)] = 1
    mask_2[tuple(X[np.where(k_means.labels_ == 1)].T)] = 1

    return mask_1, mask_2


def find_tc_pos(tc_filenames, filename):
    if filename in tc_filenames:
        return True, tc_filenames.index(filename)
    return False, None


def get_binary_maps(streamlines, dimensions, sft):
    if not len(streamlines):
        return np.zeros(dimensions), np.zeros(dimensions)
    elif len(streamlines) == 1:
        streamlines = [streamlines]
    tmp_sft = StatefulTractogram.from_sft(streamlines, sft)
    tmp_sft.to_vox()
    tmp_sft.to_corner()
    tmp_sft.remove_invalid_streamlines()

    if len(tmp_sft) == 1:
        return np.zeros(dimensions), np.zeros(dimensions)

    bundles_voxels = compute_tract_counts_map(tmp_sft.streamlines,
                                              dimensions).astype(np.int16)

    endpoints_voxels = get_endpoints_density_map(tmp_sft.streamlines,
                                                 dimensions).astype(np.int16)

    bundles_voxels[bundles_voxels > 0] = 1
    endpoints_voxels[endpoints_voxels > 0] = 1

    return bundles_voxels, endpoints_voxels


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    if (args.gt_tails and not args.gt_heads) \
            or (args.gt_heads and not args.gt_tails):
        parser.error('')
    if not args.gt_endpoints and (not args.gt_tails and not args.gt_heads):
        parser.error('')

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    gt_bundle_masks = []
    gt_bundle_inv_masks = []

    for gt_bundle in args.gt_bundles:

        # Support ground truth as streamlines or masks
        # Will be converted to binary masks immediately
        _, ext = split_name_with_nii(gt_bundle)
        if ext in ['.gz', '.nii.gz']:
            gt_img = nib.load(gt_bundle)
            gt_mask = gt_img.get_fdata().astype(np.int16)
            affine = gt_img.affine
            dimensions = gt_mask.shape
        else:
            gt_sft = load_tractogram_with_reference(parser, args, gt_bundle)
            gt_sft.remove_invalid_streamlines()
            gt_sft.to_vox()
            gt_sft.to_corner()
            affine, dimensions, _, _ = gt_sft.space_attributes
            gt_mask = compute_tract_counts_map(gt_sft.streamlines,
                                               dimensions).astype(np.int16)
        gt_inv_mask = np.zeros(dimensions, dtype=np.int16)
        gt_inv_mask[gt_mask == 0] = 1
        gt_mask[gt_mask > 0] = 1
        gt_bundle_masks.append(gt_mask)
        gt_bundle_inv_masks.append(gt_inv_mask)
    
    # If endpoints without heads/tails are loaded, split them and continue
    # normally after. Q/C of the output is important
    if args.gt_endpoints:
        tails = []
        heads = []
        for mask_filename in args.gt_endpoints:
            mask = nib.load(mask_filename).get_fdata().astype(np.int16)
            head, tail = split_heads_tails_kmeans(mask)

            basename = os.path.basename(
                split_name_with_nii(mask_filename)[0])
            tail_filename = os.path.join(
                args.out_dir, '{}_tail.nii.gz'.format(basename))
            head_filename = os.path.join(
                args.out_dir, '{}_head.nii.gz'.format(basename))
            nib.save(nib.Nifti1Image(head, affine), head_filename)
            nib.save(nib.Nifti1Image(tail, affine), tail_filename)

            tails.append(tail_filename)
            heads.append(head_filename)

        args.gt_tails, args.gt_heads = tails, heads
        args.gt_endpoints = None

    # Load the endpoints heads/tails, keep the correct combinations
    # separately from all the possible combinations
    all_data = []
    tc_filenames = list(zip(args.gt_tails, args.gt_heads))

    for mask_1_filename, mask_2_filename in tc_filenames:
        mask_1 = nib.load(mask_1_filename).get_fdata().astype(np.int16)
        mask_2 = nib.load(mask_2_filename).get_fdata().astype(np.int16)
        all_data.extend([mask_1, mask_2])

    tc_streamlines_list = []

    # Again keep the keep the correct combinations
    comb_filename = list(itertools.combinations(
        itertools.chain(*zip(args.gt_tails, args.gt_heads)), r=2))
    comb_data = list(itertools.combinations(all_data, r=2))

    # Go through all the possible combinations of endpoints masks
    for i, roi in enumerate(comb_data):
        timer = time()
        tmp_sft, _ = filter_grid_roi(sft, roi[0], 'either_end', False)
        tmp_sft, _ = filter_grid_roi(tmp_sft, roi[1], 'either_end', False)
        streamlines = tmp_sft.streamlines

        # Different processing for the true connections and the false connections
        is_tc, tc_pos = find_tc_pos(tc_filenames, comb_filename[i])
        if is_tc:
            tc_streamlines = streamlines
        else:
            tc_streamlines = []
    
        tc_streamlines_list.append(tc_streamlines)

    final_results = {}
    final_results['trk_filename'] = str(args.in_tractogram)
    final_results['tractogram_overlap'] = 0.0
    final_results["bundle_wise"] = {}
    final_results["bundle_wise"]["true_connections"] = {}
    tractogram_overlap = 0.0

    # Bundle-wise statistics, useful for more complex phantom
    for i, filename in enumerate(comb_filename):
        timer = time()
        is_tp, tc_pos = find_tc_pos(tc_filenames, filename)
        current_tc_streamlines = tc_streamlines_list[i]
        current_tc_voxels, current_tc_endpoints_voxels = get_binary_maps(
            current_tc_streamlines, dimensions, sft)
    
        tmp_dict = {}
        if is_tp:
            bundle_overlap = gt_bundle_masks[tc_pos] * current_tc_voxels
            bundle_overreach = np.zeros(dimensions)
            bundle_lacking = np.zeros(dimensions)
            bundle_lacking[np.where((gt_bundle_masks[tc_pos] == 1)
                                    & (current_tc_voxels == 0))] = 1

            tmp_dict['tc_bundle_overlap'] = np.count_nonzero(bundle_overlap)
            tmp_dict['tc_bundle_lacking'] = np.count_nonzero(bundle_lacking)
            tmp_dict['tc_bundle_overlap_PCT'] = tmp_dict['tc_bundle_overlap'] / \
                                                (tmp_dict['tc_bundle_overlap'] + tmp_dict['tc_bundle_lacking'])
            tractogram_overlap += tmp_dict['tc_bundle_overlap_PCT']
            final_results["bundle_wise"]["true_connections"][str(filename)] = tmp_dict
        
    final_results['tractogram_overlap'] = tractogram_overlap / len(gt_bundle_masks)
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(final_results, f,
                  indent=args.indent,
                  sort_keys=args.sort_keys)


if __name__ == '__main__':
    main()