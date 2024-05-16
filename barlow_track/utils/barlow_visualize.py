import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def visualize_model_performance(c, save_fname=None):
    """
    Plots the correlation matrix of the model's output, which should be close to diagonal (if trained)

    Parameters
    ----------
    c

    Returns
    -------

    """
    all_vals = c.cpu().numpy()

    vmin = np.clip(np.nanmin(all_vals), -1, 0)
    vmax = np.clip(np.nanmax(all_vals), 0, 1)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    fig = plt.figure()
    plt.imshow(all_vals[:10, :10], norm=norm, cmap='PiYG')
    plt.colorbar()
    # plt.show()
    if save_fname is not None:
        plt.savefig(save_fname)

    return fig
