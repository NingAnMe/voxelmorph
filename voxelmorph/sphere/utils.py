<<<<<<< HEAD
import torch
=======
>>>>>>> 01d6d2540feaccf9f1999d751f7235258189eca7
from matplotlib import pyplot as plt


def plot_loss_img(losses, out_img_file):
    plt.figure()
    x = range(len(losses))
    y = losses
    plt.plot(x, y)
    plt.savefig(out_img_file, dpi=200)
    print(f"save image: {out_img_file}")
