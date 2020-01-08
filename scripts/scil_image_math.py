#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from copy import copy
import logging
from numbers import Number
import os
from functools import reduce

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.image.operations import (is_float, absolute_value, around,
                                     addition, ceil, closing, convert,
                                     difference, dilation, division, erosion,
                                     floor, gaussian_blur, intersection, invert,
                                     lower_clip, lower_threshold, mean,
                                     multiplication, normalize_max,
                                     normalize_sum, opening, std, subtraction,
                                     union, upper_threshold, upper_clip,
                                     get_operation_doc)
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_outputs_exist)

OPERATIONS = {
    'lower_threshold': lower_threshold,
    'upper_threshold': upper_threshold,
    'lower_clip': lower_clip,
    'upper_clip': upper_clip,
    'absolute_value': absolute_value,
    'round': around,
    'ceil': ceil,
    'floor': floor,
    'normalize_sum': normalize_sum,
    'normalize_max': normalize_max,
    'convert': convert,
    'invert': invert,
    'addition': addition,
    'subtraction': subtraction,
    'multiplication': multiplication,
    'division': division,
    'mean': mean,
    'std': std,
    'union': union,
    'intersection': intersection,
    'difference': difference,
    'dilation': dilation,
    'erosion': erosion,
    'closing': closing,
    'opening': opening,
    'blur': gaussian_blur
}

DESCRIPTION = """
Performs an operation on a list of mask images. The supported
operations are:

Some operations such as multiplication or addition accept float value as
parameters instead of images.
> scil_image_math.py multiplication img.nii.gz 10 mult_10.nii.gz
"""

DESCRIPTION += get_operation_doc()


def build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    p.add_argument('operation',
                   help='The type of operation to be performed on the '
                   'masks.')

    p.add_argument('inputs', nargs='+',
                   help='The list of image files or parameters.')

    p.add_argument('--data_type',
                   help='Data type of the output image.')

    p.add_argument('output',
                   help='Output image path.')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def load_data(arg):
    if is_float(arg):
        data = float(arg)
    else:
        if not os.path.isfile(arg):
            logging.error('Input file %s does not exist', path)
            raise ValueError
        data = nib.load(arg).get_data()

        if data.ndim > 3:
            logging.warning('%s has %s dimensions, be careful', arg, data.ndim)

    return data


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_outputs_exist(parser, args, args.output)

    # Binary operations require specific verifications
    binary_op = ['union', 'intersection', 'difference', 'invert',
                 'dilation', 'erosion', 'closing', 'opening']

    if args.operation not in OPERATIONS.keys():
        parser.error('Operation {} not implement'.format(args.operation))

    # Find at least one image for reference
    for input_arg in args.inputs:
        if not is_float(input_arg):
            ref_img = nib.load(input_arg)
            break

    # Load all input masks.
    input_data = []
    for input_arg in args.inputs:
        if not is_float(input_arg) and \
                not is_header_compatible(ref_img, input_arg):
            parser.error('Input do not have a compatible header')
        data = load_data(input_arg)
        if isinstance(data, np.ndarray) and \
            data.dtype != ref_img.get_data_dtype() and \
                not args.data_type:
            parser.error('Input do not have a compatible data type.'
                         'Use --data_type to specified output datatype.')
        if args.operation in binary_op and isinstance(data, np.ndarray):
            unique = np.unique(data)
            if not len(unique)  == 2:
                parser.error('Binary operations can only be performed with '
                             'binary masks')

            if not (unique == [0, 1]).all():
                logging.warning('Input data for binary operation are not '
                                'binary array, will be converted. '
                                'Non-zeros will be set to ones.')
                data[data != 0] = 1

        input_data.append(data.astype(np.float64))

    if args.operation == 'convert' and not args.data_type:
        parser.error('Convert operation must be used with --data_type')

    output_data = OPERATIONS[args.operation](input_data)

    if args.data_type:
        output_data = output_data.astype(args.data_type)
        ref_img.header.set_data_dtype(args.data_type)
    else:
        output_data = output_data.astype(ref_img.get_data_dtype())
    new_img = nib.Nifti1Image(output_data, ref_img.affine,
                              header=ref_img.header)
    nib.save(new_img, args.output)


if __name__ == "__main__":
    main()
