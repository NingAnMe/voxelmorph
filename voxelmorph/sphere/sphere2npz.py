import os
import nibabel as nib
import numpy as np
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
import matplotlib.pyplot as plt
import math

input_sphere = "/mnt/ngshare/PersonData/zhenyu/dataset/Freesurfer_858"
output_npz = '/mnt/ngshare/PersonData/zhenyu/dataset/SurfReg_parameterization_858'

# select files start with "sub-"
file_names = [file for file in os.listdir(input_sphere) if file.startswith("sub-")]
print("{} of subjects".format(len(file_names)))


def xyz2degree(lh_sphere, lh_sulc, lh_curv):
    # coords: return (x, y, z) coordinates
    # faces: defining mesh triangles
    coords, faces = nib.freesurfer.read_geometry(lh_sphere)

    # (r: radius, phi: latitude, theta: longitude) in radians
    r, phi, theta = cartesian_to_spherical(coords[:, 0], coords[:, 1], coords[:, 2])

    # resize to (512, 256)
    theta_bins = 512
    phi_bins = 256
    theta_width = math.degrees(2 * np.pi) / theta_bins
    ys = theta.degree // theta_width
    phi_width = math.degrees(np.pi) / phi_bins
    xs = phi.degree // phi_width

    # load curv and sulc info
    lh_morph_sulc = nib.freesurfer.read_morph_data(lh_sulc)
    lh_morph_curv = nib.freesurfer.read_morph_data(lh_curv)
    xs = xs.astype(np.int32)
    ys = ys.astype(np.int32)

    # values store [y-axis, x-axis, sulc value, curv valus]
    values = np.zeros((512, 256))
    values[ys, xs] = lh_morph_sulc
    # values[ys, xs, 1] = lh_morph_curv

    return values, xs, ys, r

def sphere2npz(sphere, sulc, curv, npz = 'lh_sphere.npz'):
    values, _, _, _ = xyz2degree(sphere, sulc, curv)
    print(npz)

    dir = os.path.dirname(npz)
    if not os.path.exists(dir):
        os.mkdir(dir)

    # save as .npz
    np.savez(npz, values)
    print(f"values have been successfully saved as .npz format: {npz}")

count = 0
for file in file_names:
    print("Working on {}th subject".format(count))
    count += 1
    lh_sphere = os.path.join(input_sphere, file, 'surf/lh.sphere')
    lh_sulc = os.path.join(input_sphere, file, 'surf/lh.sulc')
    lh_curv = os.path.join(input_sphere, file, 'surf/lh.curv')

    rh_sphere = os.path.join(input_sphere, file, 'surf/rh.sphere')
    rh_sulc = os.path.join(input_sphere, file, 'surf/rh.sulc')
    rh_curv = os.path.join(input_sphere, file, 'surf/rh.curv')

    sphere2npz(lh_sphere, lh_sulc, lh_curv, npz = os.path.join(output_npz, file, 'lh_sphere.npz'))