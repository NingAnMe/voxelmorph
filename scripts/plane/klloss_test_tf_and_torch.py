import numpy as np
import torch
import tensorflow as tf

from voxelmorph.tf.losses import KL
from voxelmorph.plane.losses import KL as KLTorch, NCC

x = np.ones((1, 16, 8, 4), dtype=np.float32)

klloss_tf = KL(0.1, (16, 8)).loss
x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
result = klloss_tf(x_tf, x_tf)

print(result)

x_torch = torch.from_numpy(x).to("cuda")
x_torch = x_torch.permute(0, 3, 1, 2)
kltorch = KLTorch(0.1).loss
result = kltorch(x_torch, x_torch, weights=True)
print(result)

ncc_torch = NCC().loss
result = ncc_torch(x_torch[:, :1, ...], x_torch[:, :1, ...] + 1, weights=False)
print(result)
