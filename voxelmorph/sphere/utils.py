import torch
from matplotlib import pyplot as plt


def plot_loss_img(losses, out_img_file):
    plt.figure()
    x = range(len(losses))
    y = losses
    plt.plot(x, y)
    plt.savefig(out_img_file, dpi=200)
    print(f"save image: {out_img_file}")
