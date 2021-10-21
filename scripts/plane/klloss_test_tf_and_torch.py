import numpy as np
import torch
import tensorflow as tf

from voxelmorph.tf.losses import KL
from voxelmorph.plane.losses import KL as KLTorch

x = np.ones((1, 3, 3, 4), dtype=np.float32)

klloss_tf = KL(0.1, (3, 3)).loss
x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
result = klloss_tf(x_tf, x_tf)

print(result)

x_torch = torch.from_numpy(x)
x_torch = x_torch.permute(0, 3, 1, 2)
kltorch = KLTorch(0.1).loss
result = kltorch(x_torch, x_torch)
print(result)

