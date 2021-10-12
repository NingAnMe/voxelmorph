import argparse
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from astropy.coordinates import cartesian_to_spherical


def normalize(data, norm_method='SD', mean=None, std=None, mi=None, ma=None):
    """
    data: 163842 * 1, numpy array
    """
    if norm_method == 'SD':
        data = data - np.median(data)
        data = data / np.std(data)

        index = np.where(data < -3)[0]
        data[index] = -3 - (1 - np.exp(3 - np.abs(data[index])))
        index = np.where(data > 3)[0]
        data[index] = 3 + (1 - np.exp(3 - np.abs(data[index])))

        data = data / np.std(data)
        index = np.where(data < -3)[0]
        data[index] = -3 - (1 - np.exp(3 - np.abs(data[index])))
        index = np.where(data > 3)[0]
        data[index] = 3 + (1 - np.exp(3 - np.abs(data[index])))

    elif norm_method == 'MinMax':
        mi = data.min()
        ma = data.max()
        return normalize(data, norm_method='PriorMinMax', mi=mi, ma=ma)
    elif norm_method == 'Gaussian':
        data = (data - data.mean()) / data.std()
    elif norm_method == 'PriorGaussian':
        assert mean is not None and std is not None, "PriorGaussian needs prior mean and std"
        data = (data - mean) / std
    elif norm_method == 'PriorMinMax':
        assert mi is not None and ma is not None, "PriorMinMax needs prior min and max"
        data = (data - mi) / (ma - mi) * 2. - 1.
    else:
        raise NotImplementedError('e')

    return data


def xyz2lonlat(vertices):
    # coords: return (x, y, z) coordinates
    # faces: defining mesh triangles

    # (r: radius; phi, x, col: latitude; theta, y, row: longitude) in radians
    r, phi, theta = cartesian_to_spherical(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    lat = phi.degree + 90
    lon = theta.degree
    return lat, lon


def xyz2lonlat_img(vertices, data, shape=(512, 256)):
    # coords: return (x, y, z) coordinates
    # faces: defining mesh triangles

    # (r: radius; phi, x, col: latitude; theta, y, row: longitude) in radians
    r, phi, theta = cartesian_to_spherical(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    lat = phi.degree + 90
    lon = theta.degree

    # resize to (512, 256)
    y_bins = shape[0]
    x_bins = shape[1]
    y_width = math.degrees(2 * np.pi) / y_bins
    ys = lon // y_width
    x_width = math.degrees(np.pi) / x_bins
    xs = lat // x_width

    ys = np.clip(ys, 0, shape[0] - 1)
    xs = np.clip(xs, 0, shape[1] - 1)

    xs = xs.astype(np.int32)
    ys = ys.astype(np.int32)

    values = np.zeros(shape)
    values[ys, xs] = data

    return values, xs, ys, r, phi, theta


def get_rot_mat_zyx(z1, y2, x3):
    """
    first x3, then y2, lastly z1
    """
    return np.array([[np.cos(z1) * np.cos(y2), np.cos(z1) * np.sin(y2) * np.sin(x3) - np.sin(z1) * np.cos(x3),
                      np.sin(z1) * np.sin(x3) + np.cos(z1) * np.cos(x3) * np.sin(y2)],

                     [np.cos(y2) * np.sin(z1), np.cos(z1) * np.cos(x3) + np.sin(z1) * np.sin(y2) * np.sin(x3),
                      np.cos(x3) * np.sin(z1) * np.sin(y2) - np.cos(z1) * np.sin(x3)],

                     [-np.sin(y2), np.cos(y2) * np.sin(x3), np.cos(y2) * np.cos(x3)]])


if __name__ == '__main__':
    count = 0
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--sphere-moving', required=True, help='moving image (source) filename')
    parser.add_argument('--sulc-moving', required=True, help='fixed image (target) filename')
    parser.add_argument('--sphere-fixed', required=True, help='warped image output filename')
    parser.add_argument('--sulc-fixed', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('--sphere-moved', required=True, help='output warp deformation filename')
    args = parser.parse_args()

    # 数据准备
    sphere_moving = args.sphere_moving
    sulc_moving = args.sulc_moving
    sphere_fixed = args.sphere_fixed
    sulc_fixed = args.sulc_fixed
    sphere_moved = args.sphere_moved

    # # 数据准备
    # sphere_moving = "Irene_test/lh.sphere"
    # sulc_moving = "Irene_test/lh.sulc"
    # sphere_fixed = "fsaverage/lh.sphere"
    # sulc_fixed = "fsaverage/lh.sulc"
    # sphere_moved = "Irene_test/lh.rigid.sphere"

    # 加载数据
    vertices_moving, faces_moving = nib.freesurfer.read_geometry(sphere_moving)
    data_moving = nib.freesurfer.read_morph_data(sulc_moving)
    vertices_fixed, faces_fixed = nib.freesurfer.read_geometry(sphere_fixed)
    data_fixed = nib.freesurfer.read_morph_data(sulc_fixed)

    # 归一化data
    data_moving = normalize(data_moving)  # [3, n]
    data_fixed = normalize(data_fixed)  # [3, n]

    # fixed坐标转2D_lonlat.计算energy时，moving和fixed都转到lonlat平面
    shape_img2d = (720, 360)
    img2d_fixed, _, _, _, phi, theta = xyz2lonlat_img(vertices_fixed, data_fixed, shape=shape_img2d)

    # 建立fixed的插值网格
    y = np.arange(0, shape_img2d[0])
    x = np.arange(0, shape_img2d[1])
    rgi = RegularGridInterpolator((y, x), img2d_fixed)

    energies = []  # 记录loss的变化，最优的rot值
    import time
    time_start = time.time()

    # 超参数
    # search_widths = [360, 80, 40, 16, 8, 4, 2, 0.08, 0.04, 0.02, 0.008, 0.004, 0.002, 0.0008]  # 最大遍历范围
    # # num_intervals = [9, 4, 5, 4, 4, 4, 5, 4, 4, 5, 4, 4, 5]  # 每个范围遍历的步数
    # search_widths = [180, 40, 20, 10, 5, 2, 0.8, 0.4, 0.02, 0.008, 0.004, 0.002, 0.0008]  # 最大遍历范围
    # num_intervals = [9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  # 每个范围遍历的步数
    # search_widths = [180, 40, 20, 10, 5, 2, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]  # 最大遍历范围
    # num_intervals = [9, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 5]  # 每个范围遍历的步数
    # search_widths = [180, 40, 20, 10, 5, 2, 1, 0.5, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]  # 最大遍历范围
    # num_intervals = [9, 4, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5]  # 每个范围遍历的步数
    # search_widths = [180, 40, 40, 20, 20, 10, 10, 5, 2, 1, 0.5, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]  # 最大遍历范围
    # num_intervals = [9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  # 每个范围遍历的步数
    # search_widths = [180, 40, 40, 20, 20, 10, 10, 5, 2, 1, 0.5, 0.02, 0.01, 0.005, 0.002, 0.001]  # 最大遍历范围
    # num_intervals = [  9,  5,  4,  5,  4,  5,  4, 5, 5, 5,   5,    5,    5,     5,     5,     5]  # 每个范围遍历的步数
    search_widths = [9, 1]  # 最大遍历范围
    num_intervals = [9, 9]  # 每个范围遍历的步数

    # 遍历角度
    center_alpha = 0.
    best_alpha = 0.
    center_beta = 0.
    best_beta = 0.
    center_gamma = 0.
    best_gamma = 0.
    best_vertices_moving_rigid = vertices_moving

    best_energy = float('inf')
    for search_width, num_interval in zip(search_widths, num_intervals):
        # search_width = search_width / 2
        search_width = search_width / 180 * np.pi
        num_interval = num_interval + 1
        print(center_alpha, center_beta, center_gamma)
        print(np.linspace(center_alpha - search_width, center_alpha + search_width, num=num_interval))
        for alpha in np.linspace(center_alpha - search_width, center_alpha + search_width, num=num_interval):
            for beta in np.linspace(center_beta - search_width, center_beta + search_width, num=num_interval):
                for gamma in np.linspace(center_gamma - search_width, center_gamma + search_width, num=num_interval):
                    count += 1
                    # 旋转
                    curr_rot = get_rot_mat_zyx(alpha, beta, gamma)
                    curr_vertices_moving = curr_rot.dot(np.transpose(vertices_moving))
                    curr_vertices_moving = np.transpose(curr_vertices_moving)  # 新的顶点坐标_3D

                    lat_moving, lon_moving = xyz2lonlat(curr_vertices_moving)  # 新的顶点坐标_经纬度

                    # 在fixed图像，采样新坐标位置的值
                    lon_moving = lon_moving / 2
                    lat_moving = lat_moving / 2
                    lonlat = np.stack((lon_moving, lat_moving), axis=1)
                    data_fixed_resample_moving = rgi(lonlat)

                    # 计算energy
                    energy = mean_squared_error(data_moving, data_fixed_resample_moving)

                    # print(count, time.time() - time_tmp)  # 每次刚性配准需要0.05秒
                    # 保存最优的变换参数
                    if energy < best_energy:
                        print(count, energy, alpha, beta, gamma)
                        best_energy = energy
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma
                        best_vertices_moving_rigid = curr_vertices_moving

        # 在最优角度的更优范围内进行搜索
        center_alpha = best_alpha
        center_beta = best_beta
        center_gamma = best_gamma
        print(time.time() - time_start)
    # 保存结果
    nib.freesurfer.write_geometry(sphere_moved, best_vertices_moving_rigid, faces_moving)
    print(time.time() - time_start, best_alpha, best_beta, best_gamma, best_energy)
    print(sphere_moved)
