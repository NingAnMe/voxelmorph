# zhenyu
import copy
import os.path
import numpy as np
from skimage import io
from PIL import Image
import imageio

def tif2npz(tif, npz):
    data = np.zeros((512, 256), dtype=np.float)

    img = io.imread(tif)
    tif_data = np.array(img)

    data[:, :] = tif_data[3]
    # data[:, :, 1] = tif_data[6]

    np.savez(npz, data)

def getmedian(input):
    input.sort()
    median = input[589824]
    return median

def median_normalization(tif):
    img = io.imread(tif)
    tif_data = np.array(img)
    tif_data_temp = tif_data.flatten()
    tif_median = copy.deepcopy(tif_data_temp)
    median = getmedian(tif_median)

    print(median)

    vtotal = 0

    for i in range(0, 512 * 256 * 9):
        std = tif_data_temp[i] - median
        var = std * std
        vtotal += var
    vtotal = vtotal / (512 * 256 * 9)
    vtotal = np.sqrt(vtotal)
    for i in range(0, 512 * 256 * 9):
        tif_data_temp[i] = (tif_data_temp[i] - median)/vtotal

    tif_data_temp = tif_data_temp.reshape(512, 256, 9)
    print(tif_data_temp)



def lookshape(filename):
    img = io.imread(filename)
    # print(img)

    file_data = np.array(img)
    print(file_data)
    a = file_data.flatten()
    print(max(a), min(a))


if __name__ == '__main__':
    input = "/home/sunzhenyu/download/SurfReg/lh.average.curvature.filled.buckner40.tif"
    output = "/home/sunzhenyu/download/SurfReg/lh.atlas.npz"
    test_data = '/home/zhenyu/SurfReg/lh.reg.template.tif'

    lookshape(test_data)
    # median_normalization(input)
    # tif2npz(input, output)
