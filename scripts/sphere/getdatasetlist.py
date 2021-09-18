# zhenyu

import os.path


def getlist_train_test(data, count, train, test):
    for subject in os.listdir(data):
        if count < 400:
            train.append(f'{data}/{subject}/lh_sphere.npz')
        else:
            test.append(f'{data}/{subject}/lh_sphere.npz')
        count += 1

    with open("/mnt/ngshare/PersonData/zhenyu/dataset/subjects_train.txt", "w") as output_train:
        for sub_train in train:
            output_train.write(f'{sub_train}\n')

            # output_train.close()
    with open("/mnt/ngshare/PersonData/zhenyu/dataset/subjects_test.txt", "w") as output_test:
        for sub_test in test:

            output_test.write(f'{sub_test}\n')

        # output_test.close()

if __name__ == '__main__':
    data = '/mnt/ngshare/PersonData/zhenyu/dataset/SurfReg_parameterization_2D_858'
    count = 0
    train = []
    test = []
    getlist_train_test(data, count, train, test)
