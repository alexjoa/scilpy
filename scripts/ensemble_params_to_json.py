#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

from dipy.data import SPHERE_FILES

from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist, add_sh_basis_args)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('output_file', help='Output file directory (must be .json).')

    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument(
        '--trk_name', help='.trk file name')
    track_g.add_argument(
        '--seeding', default='gm', choices=['gm', 'wm', 'interface'],
        help='Seeding method used')
    track_g.add_argument(
        '--algo', default='prob', choices=['det', 'prob'],
        help='Algorithm to use (must be "det" or "prob"). [%(default)s]')
    track_g.add_argument(
        '--step', dest='step_size', type=float, default=0.5,
        help='Step size in mm. [%(default)s]')
    track_g.add_argument(
        '--theta', type=float,
        help='Maximum angle between 2 steps. ["eudx"=60, det"=45, "prob"=20]')
    add_sh_basis_args(track_g)

    seed_group = p.add_argument_group(
        'Seeding options',
        'When no option is provided, uses --npv 1.')
    seed_sub_exclusive = seed_group.add_mutually_exclusive_group()
    seed_sub_exclusive.add_argument(
        '--npv', type=int,
        help='Number of seeds per voxel.')
    seed_sub_exclusive.add_argument(
        '--nt', type=int,
        help='Total number of seeds to use.')

    p.add_argument(
        '--sphere', choices=sorted(SPHERE_FILES.keys()),
        default='symmetric724',
        help='Set of directions to be used for tracking')

    out_g = p.add_argument_group('Output options')
    add_overwrite_arg(out_g)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_outputs_exist(parser, args, args.output_file)

    track_params = {
        "trk_name": args.trk_name,
        "seeding": args.seeding,
        "algo": args.algo,
        "npv": args.npv,
        "theta": args.theta,
        "step": args.step_size,
        "sphere": args.sphere
    }

    with open(args.output_file, 'w+') as fp:
        json.dump(track_params, fp)


if __name__ == "__main__":
    main()