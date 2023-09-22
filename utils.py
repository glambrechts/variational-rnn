import torch
import matplotlib.pyplot as plt


def auto_device():
    if torch.backends.mps.is_available():
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def show(*images, block=False):

    plt.close('all')

    nrows = len(images)
    ncols = images[0].size(0)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))

    for i, batch in enumerate(images):
        for j, image in enumerate(batch):
            ax = axs[j] if nrows == 1 else axs[i][j]
            ax.set_aspect('equal')
            ax.imshow(image, vmin=-0.7, vmax=0.7)
            ax.axis('off')

    plt.tight_layout()

    if block:
        plt.show()

    return fig
