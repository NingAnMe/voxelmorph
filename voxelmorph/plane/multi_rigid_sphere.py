import os
from multiprocessing import Pool


if __name__ == '__main__':
    input_sphere = "/mnt/ngshare/PersonData/zhenyu/dataset/FreeSurfer_904"
    input_sphere_fixed = "/usr/local/freesurfer/subjects/fsaverage"
    # 多线程
    multi = 10  # 线程数量，大于1为多线程
    # select files start with "sub-"
    file_names = [file for file in os.listdir(input_sphere) if file.startswith("sub-")]
    print("{} of subjects".format(len(file_names)))
    count = 0
    cmds = []
    for file in file_names:
        count = 0
        print("Working on {}th subject".format(count))
        lh_sphere_moving = os.path.join(input_sphere, file, 'lh.sphere')
        lh_sulc_moving = os.path.join(input_sphere, file, 'lh.sulc')

        lh_sphere_fixed = os.path.join(input_sphere_fixed, 'surf/lh.sphere')
        lh_sulc_fixed = os.path.join(input_sphere_fixed, 'surf/lh.sulc')

        lh_sphere_moved = os.path.join(input_sphere, file, 'lh.sphere.rigid')

        cmd = f"python3 /home/zhenyu/SurfReg/voxelmorph2D/voxelmorph/voxelmorph/sphere/rigid_sphere.py " \
              f"--sphere-moving {lh_sphere_moving} --sulc-moving {lh_sulc_moving} " \
              f"--sphere-fixed {lh_sphere_fixed} --sulc-fixed {lh_sulc_fixed} --sphere-moved {lh_sphere_moved}"
        print(f"cmd : {cmd}")
        count += 1
        print(f"第{count}组")
        cmds.append(cmd)
    if multi > 1:
        p = Pool(multi)
        p.map(os.system, cmds)
        p.close()
        p.join()
    else:
        for cmd in cmds:
            print(cmd)
            os.system(cmd)
