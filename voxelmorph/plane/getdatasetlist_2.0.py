# zhenyu

import os.path
import numpy as np
from sklearn.model_selection import KFold


def getlist_train_test(data):
    dataset = []
    for subject in os.listdir(data):
        dataset.append(f'{data}/{subject}/lh_sphere.npz')
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits((dataset))

    datasets = np.array(dataset)
    count = 1
    for train_index, test_index in kf.split(dataset):

        train = datasets[train_index]
        test = datasets[test_index]

        with open(f"/mnt/ngshare/PersonData/zhenyu/dataset/train_{count}.txt", "w") as output_train:
            for sub_train in train:
                output_train.write(f'{sub_train}\n')
        with open(f"/mnt/ngshare/PersonData/zhenyu/dataset/test_{count}.txt", "w") as output_test:
            for sub_test in test:
                output_test.write(f'{sub_test}\n')
        count += 1

if __name__ == '__main__':
    data = '/mnt/ngshare/PersonData/zhenyu/dataset/SurfReg_parameterization_2D_858'
    getlist_train_test(data)
