import matplotlib.pyplot as plt
import itertools
import numpy as np

"""
Credit to Jaroslav Knotek for the following functions:
- visu_cols
- def _draw_line(plt, a, b):
- draw_grid
"""
def visu_cols(imgs, figsize=None):
    if figsize is None:
        figsize = (6*len(imgs), 80)

    fig, ax_row = plt.subplots(1, len(imgs), figsize=figsize)
    for ax, img in zip(ax_row, imgs):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)


def _draw_line(plt, a, b):
    if np.isnan([a, b]).any():
        return
    plt.plot([a[0], b[0]], [a[1], b[1]], c='orange', alpha=.4)


def draw_grid(img, grid):
    plt.figure(figsize=(12, 60))
    plt.imshow(img, cmap='gray')

    for a, b, c in grid:
        _draw_line(plt, a, b)
        _draw_line(plt, c, b)

    for col_id in range(3):
        col = grid[:, col_id]
        for a, b in itertools.pairwise(col):
            _draw_line(plt, a, b)
    all_pts = grid.reshape((-1, 2))
    plt.plot(all_pts[:, 0], all_pts[:, 1], 'x', c='r')

# for one img and two grids
def draw_grid_combined(grid1, grid2, image):
    plt.figure(figsize=(12, 60))

    plt.imshow(image, cmap='gray')

    for a, b, c in grid1:
        _draw_line(plt, a, b)
        _draw_line(plt, c, b)

    for col_id in range(3):
        col = grid1[:, col_id]
        for a, b in itertools.pairwise(col):
            _draw_line(plt, a, b)
    all_pts = grid1.reshape((-1, 2))
    plt.plot(all_pts[:, 0], all_pts[:, 1], 'x', c='r')

    # Plot grid2
    for a, b, c in grid2:
        _draw_line(plt, a, b)
        _draw_line(plt, c, b)

    for col_id in range(3):
        col = grid2[:, col_id]
        for a, b in itertools.pairwise(col):
            _draw_line(plt, a, b)
    all_pts = grid2.reshape((-1, 2))
    plt.plot(all_pts[:, 0], all_pts[:, 1], 'x', c='r')
