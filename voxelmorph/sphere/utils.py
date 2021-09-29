from matplotlib import pyplot as plt
import numpy as np


def plot_loss_img(losses, out_img_file):
    plt.figure()
    x = range(len(losses))
    y = losses
    plt.plot(x, y)
    plt.savefig(out_img_file, dpi=200)
    print(f"save image: {out_img_file}")


def minmaxnormalize(sub_data):
    zeros = sub_data == 0
    max = np.max(sub_data)
    min = np.min(sub_data)
    norm = (sub_data - min) / (max - min)
    norm[zeros] = 0
    return norm


def domainnorm(sub_data):
    domain = 33
    norm = sub_data / domain

    return norm
