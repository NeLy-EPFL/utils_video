import numpy as np
from matplotlib import pyplot as plt

from .utils import grid_size, colorbar, add_colorbar

def dff_trials(snippets):
    """
    This function returns a generator that
    yields frames with the given snippets
    of dff stacks.

    Parameters
    ----------
    snippets : list of 3D numpy arrays
        Each array is a snippet for a single
        trial.

    Returns
    -------
    frame_generator : generator
        A generator that yields individual
        video frames.
    """
    # Check that all snippets have the same frame size.
    if not len(set([snippet.shape[1] for snippet in snippets])) and len(set([snippet.shape[2] for snippet in snippets])):
        raise ValueError("Snippets do not have the same frame size.")

    n_snippets = len(snippets)
    max_length = max([len(stack) for stack in snippets])
    frame_size = snippets[0].shape[1:]
    n_rows, n_cols = grid_size(n_snippets, frame_size)
    
    frames = np.zeros((max_length, frame_size[0] * n_rows, frame_size[1] * n_cols))
    for i, stack in enumerate(snippets):
        row_idx = int(i / n_cols)
        col_idx = i % n_cols
        frames[: len(stack), row_idx * frame_size[0] : (row_idx + 1) * frame_size[0], col_idx * frame_size[1] : (col_idx + 1) * frame_size[1]] = stack
        frames[len(stack) :, row_idx * frame_size[0] : (row_idx + 1) * frame_size[0], col_idx * frame_size[1] : (col_idx + 1) * frame_size[1]] = stack[-1]

    vmin = np.percentile(frames, 0.5)
    vmax = np.percentile(frames, 99.5)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.jet
    cbar = colorbar(norm, cmap, (frames.shape[1], -1))
    frames = cmap(norm(frames))
    frames = (frames * 255).astype(np.uint8)
    frames = add_colorbar(frames, cbar, 'right')

    def frame_generator():
        for frame in frames:
            yield frame

    return frame_generator()
