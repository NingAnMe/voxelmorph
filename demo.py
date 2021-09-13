import numpy as np
from skimage import io

img = io.imread('/home/sunzhenyu/download/SurfReg/lh.average.curvature.filled.buckner40.tif')
data = np.array(img)
a = data[3]
b = data[6]
c = np.stack((a, b))

print(c)
io.imsave('/home/sunzhenyu/download/SurfReg/lh.tif', np.float32(c))
