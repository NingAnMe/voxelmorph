#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch

# import voxelmorph with sphere backend
os.environ['VXM_BACKEND'] = 'sphere'
import voxelmorph as vxm   # nopep8

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
# parser.add_argument('--normalize_type', default='std',  help='select the data normalization processing type')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

def meannormalize(sub_data):
    mean = np.mean(sub_data)
    std = np.std(sub_data)
    norm = (sub_data - mean) / std
    return norm, mean, std

def backmeannormalize(input, mean, std):
    output = input * std + mean
    return output

def minmaxnormalize(sub_data):
    max = np.max(sub_data)
    min = np.min(sub_data)
    norm = (sub_data - min) / (max - min)
    return norm, max, min

def backminmaxnormalize(input, max, min):
    output = input * (max - min) + min
    return output

# def normalize_forword(data, type="std"):
#     if type == "std":
#         return meannormalize(data)
#     elif type == "min_max":
#         return minmaxnormalize(data)
#     else:
#         raise KeyError("type is error")
#
# def normalize_backword(data, a, b, type="std"):
#     if type == "std":
#         return backmeannormalize(data, a, b)
#     elif type == "min_max":
#         return backminmaxnormalize(data, a, b)
#     else:
#         raise KeyError("type is error")

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

# set up normalize type
# normalize_type = args.normalize_type
# normalize_type = "min_max"

# set up tensors and permute
# moving, a_moving, b_moving = normalize_forword(moving, type=normalize_type)
# fixed, a_fixed, b_fixed = normalize_forword(fixed, type=normalize_type)

moving, a_moving, b_moving = meannormalize(moving)
fixed, a_fixed, b_fixed = meannormalize(fixed)

input_moving = torch.from_numpy(moving).to(device).float().permute(0, 3, 1, 2)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 3, 1, 2)

# predict
moved, warp = model(input_moving, input_fixed, registration=True)
# moved = normalize_backword(moved, a_moving, b_moving, type=normalize_type)
moved = backmeannormalize(moved, a_moving, b_moving)

# save moved image
if args.moved:
    moved = moved.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

# save warp
if args.warp:
    warp = warp.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)


