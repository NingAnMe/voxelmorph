import argparse
import nibabel as nib
from voxelmorph.sphere.idw import tree


def sphere2sphere_163842(coords_moving, morph_sulc_moving, coords_fixed):
    idw_tree = tree(coords_moving, morph_sulc_moving)
    sulc_moving_resampled_by_fixed = idw_tree(coords_fixed)

    return sulc_moving_resampled_by_fixed


if __name__ == '__main__':
    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('--sphere-moving', required=True, help='input sphere moving file path',
                        default='/home/anning/project/surfreg/Irene_test/lh.sphere')
    parser.add_argument('--sulc-moving', required=True, help='input sphere moving sulcus file path',
                        default='/home/anning/project/surfreg/Irene_test/lh.sulc')
    parser.add_argument('--sphere-fixed', required=True, help='optional input image file prefix',
                        default='/home/anning/project/surfreg/fsaverage/lh.sphere')
    parser.add_argument('--sphere-163842', required=True, help='resampled sphere_163842 output path',
                        default='/home/anning/project/surfreg/Irene_test/resample_163842/lh.sphere')
    args = parser.parse_args()

    # output path
    sulc_moving_resample_163842 = "/home/anning/project/surfreg/Irene_test/resample_163842/lh.sphere"

    # read input file
    coords_moving, _ = nib.freesurfer.read_geometry(args.sphere_moving)
    morph_sulc_moving = nib.freesurfer.read_morph_data(args.sulc_moving)
    coords_fixed, _ = nib.freesurfer.read_geometry(args.sphere_fixed)

    # save the resampled sphere_163842
    values_moving_resampled_by_fixed_points_3d = sphere2sphere_163842(coords_moving, morph_sulc_moving, coords_fixed)
    nib.freesurfer.write_morph_data(args.sphere_163842, values_moving_resampled_by_fixed_points_3d)
