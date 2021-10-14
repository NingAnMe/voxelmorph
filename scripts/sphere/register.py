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
import matplotlib.pyplot as plt
# third party
import numpy as np
import nibabel as nib
import torch
from scipy.interpolate import RegularGridInterpolator
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

# import voxelmorph with sphere backend
os.environ['VXM_BACKEND'] = 'sphere'
import voxelmorph as vxm  # nopep8
import math

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', help='warped image output filename')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
# parser.add_argument('--normalize_type', default='std',  help='select the data normalization processing type')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('--sphere_sub', help='sphere_sub image filename')
parser.add_argument('--sphere_atlas', help='sphere_atlas image filename')
parser.add_argument('--sphere_reg', help='sphere.reg image output filename')
parser.add_argument('--sulc_sub', help='silc_sub image filename')
parser.add_argument('--sulc_atlas', help='silc_atlas image filename')
parser.add_argument('--sphere_freesurfer', help='sphere_freesurfer image filename')
parser.add_argument('--plot_image', help='show time image output filename')
parser.add_argument('--plot_image_dif_1', help='show dif image output filename')
parser.add_argument('--plot_image_dif_2', help='show dif image output filename')
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
    zeros = sub_data == 0
    max = np.max(sub_data)
    min = np.min(sub_data)
    norm = (sub_data - min) / (max - min)
    norm[zeros] = 0
    return norm


def backminmaxnormalize(input, max, min):
    output = input * (max - min) + min
    return output


def domainnorm(sub_data):
    domain = 33
    norm = sub_data / domain
    return norm


def backdomainnorm(sub_data):
    domain = 33
    output = sub_data * domain
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

def interpolate(warp_file, lh_sphere):
    x = np.linspace(-128, 128, 256)  # phi ###
    y = np.linspace(0, 512, 512)  # theta ###

    # print(warp_file.files)
    warp = warp_file.squeeze()
    warp = warp.permute(0, 2, 1)
    warp = warp.detach().numpy()
    # warp = warp_file['vol']
    # warp = np.moveaxis(warp, 1, -1)

    interpolate_function_x = RegularGridInterpolator((x, y), -warp[0])  # x-axis
    interpolate_function_y = RegularGridInterpolator((x, y), -warp[1])  # y-axis

    coords, faces = nib.freesurfer.read_geometry(lh_sphere)
    r, phi, theta = cartesian_to_spherical(coords[:, 0], coords[:, 1], coords[:, 2])
    p = phi.degree
    t = theta.degree

    theta_bins = 512
    phi_bins = 256
    theta_width = math.degrees(2 * np.pi) / theta_bins
    t /= theta_width
    phi_width = math.degrees(np.pi) / phi_bins
    p /= phi_width
    t = t.reshape(-1, 1)
    p = p.reshape(-1, 1)
    pts = np.concatenate((p, t), axis=1)

    new_pts_x = interpolate_function_x(pts)
    new_pts_y = interpolate_function_y(pts)
    x_prime = pts.T[0] + new_pts_x
    y_prime = pts.T[1] + new_pts_y

    x_prime *= phi_width
    y_prime *= theta_width
    y_prime = np.clip(y_prime, 0, 360)
    x_prime = np.clip(x_prime, -90, 90)

    t_prime = [math.radians(i) for i in y_prime]
    p_prime = [math.radians(i) for i in x_prime]
    t_prime = np.array(t_prime)
    p_prime = np.array(p_prime)

    return r, p_prime, t_prime

# save 4 image
def save4image(lh_sphere_sub, lh_sphere_atlas, lh_sulc_sub, lh_sulc_atlas, lh_sphere_freesurfer, phi_prime, theta_prime,
               imagesavefilename):
    lh_morph_sulc_sub = nib.freesurfer.read_morph_data(lh_sulc_sub)
    lh_morph_sulc_atlas = nib.freesurfer.read_morph_data(lh_sulc_atlas)

    coords_sub, faces_sub = nib.freesurfer.read_geometry(lh_sphere_sub)
    r_sub, phi_sub, theta_sub = cartesian_to_spherical(coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2])
    coords_atlas, faces_atlas = nib.freesurfer.read_geometry(lh_sphere_atlas)
    r_atlas, phi_atlas, theta_atlas = cartesian_to_spherical(coords_atlas[:, 0], coords_atlas[:, 1], coords_atlas[:, 2])
    coords_freesurfer, faces_freesurfer = nib.freesurfer.read_geometry(lh_sphere_freesurfer)
    r_reg, phi_reg, theta_reg = cartesian_to_spherical(coords_freesurfer[:, 0], coords_freesurfer[:, 1],
                                                       coords_freesurfer[:, 2])

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(141)
    ax.scatter(phi_sub.degree, theta_sub.degree, s=0.1,
               c=lh_morph_sulc_sub)  # phi.degree: [-90, 90], theta.degree: [0, 360]
    plt.title('Moving')

    ax = fig.add_subplot(142)
    ax.scatter(phi_atlas.degree, theta_atlas.degree, s=0.1, c=lh_morph_sulc_atlas)
    plt.title('Fixed')

    ax = fig.add_subplot(143)
    phi_prime = [math.degrees(p) for p in phi_prime]
    thtea_prime = [math.degrees(t) for t in theta_prime]
    ax.scatter(phi_prime, thtea_prime, s=0.1, c=lh_morph_sulc_sub)  # (256, 512)
    plt.title('Moved')

    ax = fig.add_subplot(144)
    ax.scatter(phi_reg.degree, theta_reg.degree, s=0.1, c=lh_morph_sulc_sub)  # (256, 512)
    plt.title('Moved FreeSurfer')

    plt.savefig(imagesavefilename)


def xyz2degree(lh_sphere, lh_sulc):
    # coords: return (x, y, z) coordinates
    # faces: defining mesh triangles
    coords, faces = nib.freesurfer.read_geometry(lh_sphere)

    # (r: radius, phi: latitude, theta: longitude) in radians
    r, phi, theta = cartesian_to_spherical(coords[:, 0], coords[:, 1], coords[:, 2])

    lat = phi.degree + 90
    lon = theta.degree
    # resize to (512, 256)
    y_bins = 512
    x_bins = 256
    y_width = math.degrees(2 * np.pi) / y_bins
    ys = lon // y_width
    x_width = math.degrees(np.pi) / x_bins
    xs = lat // x_width

    ys = np.clip(ys, 0, 511)
    xs = np.clip(xs, 0, 255)

    # load curv and sulc info
    lh_morph_sulc = nib.freesurfer.read_morph_data(lh_sulc)
    xs = xs.astype(np.int32)
    ys = ys.astype(np.int32)

    # values store [theta, phi, sulc value, curv value]
    values = np.zeros((512, 256))
    values[ys, xs] = lh_morph_sulc
    #     values[1, ys, xs] = lh_morph_curv

    return values

def xyz2degree2(phi, theta, lh_sulc):

    lat = phi + 90
    lon = theta
    # resize to (512, 256)
    y_bins = 512
    x_bins = 256
    y_width = math.degrees(2 * np.pi) / y_bins
    ys = lon // y_width
    x_width = math.degrees(np.pi) / x_bins
    xs = lat // x_width

    ys = np.clip(ys, 0, 511)
    xs = np.clip(xs, 0, 255)

    # load curv and sulc info
    lh_morph_sulc = nib.freesurfer.read_morph_data(lh_sulc)
    xs = xs.astype(np.int32)
    ys = ys.astype(np.int32)

    # values store [theta, phi, sulc value, curv value]
    values = np.zeros((512, 256))
    values[ys, xs] = lh_morph_sulc
    #     values[1, ys, xs] = lh_morph_curv

    return values

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

# moving = domainnorm(moving)
moving = minmaxnormalize(moving)
fixed = minmaxnormalize(fixed)

input_moving = torch.from_numpy(moving).to(device).float().permute(0, 3, 1, 2)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 3, 1, 2)

# predict
moved, warp = model(input_moving, input_fixed, registration=True)
# moved = normalize_backword(moved, a_moving, b_moving, type=normalize_type)
# moved = backdomainnorm(moved)


if args.sphere_sub:
    c, faces = nib.freesurfer.read_geometry(args.sphere_sub)
    coords = np.empty(shape=c.shape)
    r, phi_prime, theta_prime = interpolate(warp, args.sphere_sub)
    coords[:, 0], coords[:, 1], coords[:, 2] = spherical_to_cartesian(r, phi_prime, theta_prime)
    nib.freesurfer.io.write_geometry(args.sphere_reg, coords, faces)

if args.plot_image:
    lh_sphere_sub = args.sphere_sub
    lh_sphere_atlas = args.sphere_atlas
    lh_sulc_sub = args.sulc_sub
    lh_sulc_atlas = args.sulc_atlas
    lh_sphere_freesurfer = args.sphere_freesurfer
    imagesavefilename = args.plot_image
    save4image(lh_sphere_sub, lh_sphere_atlas, lh_sulc_sub, lh_sulc_atlas, lh_sphere_freesurfer, phi_prime, theta_prime,
               imagesavefilename)
if args.plot_image_dif_1 or args.plot_image_dif_2:
    imagesavefilenamedif_1 = args.plot_image_dif_1
    imagesavefilenamedif_2 = args.plot_image_dif_2
    dif_moving = xyz2degree(lh_sphere_sub, lh_sulc_sub)
    dif_moved = xyz2degree2(phi_prime, theta_prime, lh_sulc_sub)
    dif_freesurfer = xyz2degree(lh_sphere_freesurfer, lh_sulc_sub)
    dif_moved_moving = dif_moved - dif_moving
    print(np.nanmax(dif_moved_moving), np.nanmin(dif_moved_moving), np.nanmean(dif_moved_moving))
    dif_freesurfer_moved = dif_freesurfer - dif_moved

    plt.figure(figsize=(14, 7))
    plt.imshow(dif_moved_moving)
    plt.title('moved_moving')
    plt.colorbar()
    plt.savefig(imagesavefilenamedif_1)

    plt.figure(figsize=(14, 7))
    plt.imshow(dif_freesurfer_moved)
    plt.title('freesurfer_moved')
    plt.colorbar()
    plt.savefig(imagesavefilenamedif_2)


# save moved image
if args.moved:
    moved = moved.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

# save warp
if args.warp:
    warp = warp.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)
