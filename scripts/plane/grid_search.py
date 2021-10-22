import os
from multiprocessing import Pool

cmds = list()
for lam in range(-7, 7):
    lam = 1 * 10 ** lam
    kllams = list(range(-7, 7)) + [0]
    for kllam in kllams:
        kllam = 1 * 10 ** kllam
        for loss in ['mse', 'ncc']:
            cmd = f"python3 /home/sunzhenyu/SurfReg/voxelmorph2D/voxelmorph/scripts/plane/train.py --img-list /home/sunzhenyu/SurfReg/list/plane1/train.txt  " \
                  f"--img-list-val /home/sunzhenyu/SurfReg/list/plane1/val.txt " \
                  f"--atlas /home/sunzhenyu/SurfReg/atlas/fsave/lh_sphere_fsave.npz  " \
                  f"--model-dir /home/sunzhenyu/SurfReg/models/models_fsave_minmaxnorm_103_mse_{lam}_kl_{kllam}_epoch_300 --epochs 300 " \
                  f"--steps-per-epoch 250 --image-loss {loss} --lambda {lam} --use-probs --kl-lambda {kllam}"
            cmds.append(cmd)

p = Pool(4)
p.map(os.system, cmds)
p.close()
p.join()
