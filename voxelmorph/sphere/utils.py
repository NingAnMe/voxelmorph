import torch
import matplotlib.pyplot as plt


def plot_loss_img(losses, out_img_file):
    ax = plt.figure()
    x = range(len(losses))
    y = losses
    ax.plot(x, y)
    plt.savefig(out_img_file, dpi=200)
    print(f"save image: {out_img_file}")
