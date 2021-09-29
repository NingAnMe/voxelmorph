import math
import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import cartesian_to_spherical
# from astropy.coordinates import spherical_to_cartesian


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
        raise NotImplementedError('e')
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


def plot_loss_img(losses, out_img_file):
    plt.figure()
    x = range(len(losses))
    y = losses
    plt.plot(x, y)
    plt.savefig(out_img_file, dpi=200)
    print(f"save image: {out_img_file}")


def xyz2lonlat(vertices, data):
    # coords: return (x, y, z) coordinates
    # faces: defining mesh triangles

    # (r: radius, phi: latitude, theta: longitude) in radians
    r, phi, theta = cartesian_to_spherical(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    # resize to (512, 256)
    theta_bins = 512
    phi_bins = 256
    theta_width = math.degrees(2 * np.pi) / theta_bins
    ys = theta.degree // theta_width
    phi_width = math.degrees(np.pi) / phi_bins
    xs = phi.degree // phi_width

    xs = xs.astype(np.int32)
    ys = ys.astype(np.int32)

    values = np.zeros((512, 256))
    values[ys, xs] = data

    return values, xs, ys, r
