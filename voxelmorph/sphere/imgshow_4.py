import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import nibabel as nib
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
import pylab
import math


def load_file_npz(input_npz):
    img_temp = io.imread(input_npz)
    img = np.array(img_temp)
    return img


def load_file_tif(input_tif):
    data = np.zeros((512, 256), dtype=np.float)

    img_temp = io.imread(input_tif)
    tif_data = np.array(img_temp)

    data[:, :] = tif_data[3]

    return data


def load_file_sphere(lh_sphere, lh_sulc):
    coords, faces = nib.freesurfer.read_geometry(lh_sphere)

    r, phi, theta = cartesian_to_spherical(coords[:, 0], coords[:, 1], coords[:, 2])

    # resize to (512, 256)
    theta_bins = 512
    phi_bins = 256
    theta_width = math.degrees(2 * np.pi) / theta_bins
    ys = theta.degree // theta_width
    phi_width = math.degrees(np.pi) / phi_bins
    xs = phi.degree // phi_width

    # load sulc info
    lh_morph_sulc = nib.freesurfer.read_morph_data(lh_sulc)
    xs = xs.astype(np.int32)
    ys = ys.astype(np.int32)

    # values store [y-axis, x-axis, sulc value]
    values = np.zeros((512, 256))
    values[ys, xs] = lh_morph_sulc

    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving', required=True, help='moving image (source) filename')
    parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
    parser.add_argument('--moved_model', required=True, help='')
    parser.add_argument('--moved_freesurfer_sphere', required=True, help='')
    parser.add_argument('--moved_freesurfer_sulc', required=True, help='')

    args = parser.parse_args()
    moving = args.moving
    fixed = args.fixed
    moved_model = args.moved_model
    moved_freesurfer_sphere = args.moved_freesurfer_sphere
    moved_freesurfer_sulc = args.moved_freesurfer_sphere_sulc

    # moving = '/home/zhenyu/SurfReg/moving_sub/lh_sub031475/lh_sphere.npz'
    # moved_model = '/home/zhenyu/SurfReg/moved_result/meannorm_true/lh_sub031475_meannorm_fav6.npz'
    # fixed = '/home/zhenyu/SurfReg/atlas/lh_sphere_fav6.npz'

    # moved_freesurfer_sphere = '/home/zhenyu/SurfReg/freesurfer_reg/lh.sphere.reg'
    # moved_freesurfer_sulc = '/home/zhenyu/SurfReg/freesurfer_reg/lh.sulc'

    fig = plt.figure(figsize=(36, 9))
    ax = fig.add_subplot(1, 4, 1)
    moving_img = load_file_npz(moving)
    plt.title("moving")
    plt.imshow(moving_img)

    ax = fig.add_subplot(1, 4, 2)
    fixed_img = load_file_npz(fixed)
    plt.title("fixed")
    plt.imshow(fixed_img)

    ax = fig.add_subplot(1, 4, 3)
    moved_img = load_file_npz(moved_model)
    plt.title("moved_model")
    plt.imshow(moved_img)

    ax = fig.add_subplot(1, 4, 4)
    moved_freesurfer_img = load_file_sphere(moved_freesurfer_sphere, moved_freesurfer_sulc)
    plt.title("moved_freesurfer")
    plt.imshow(moved_freesurfer_img)

