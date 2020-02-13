import numpy as np
from matplotlib import pyplot as plt

from .utils import grid_size, colorbar, add_colorbar

def dff_trials(snippets, synchronization_indices=None):
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
    frames = _grid_frames(snippets, synchronization_indices)
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


def beh_trials(snippets, synchronization_indices=None):
    """
    This function returns a generator that
    yields frames with the given snippets
    of behaviour stacks.

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
    frames = _grid_frames(snippets, synchronization_indices)

    def frame_generator():
        for frame in frames:
            frame = np.tile(frame[:, :, np.newaxis], (1, 1, 4))
            frame[:, :, 3] = 255
            yield frame

    return frame_generator()


def _grid_frames(snippets, synchronization_indices=None):
    # Check that all snippets have the same frame size.
    if not len(set([snippet.shape[1] for snippet in snippets])) and len(set([snippet.shape[2] for snippet in snippets])):
        raise ValueError("Snippets do not have the same frame size.")

    n_snippets = len(snippets)
    lengths = [len(stack) for stack in snippets]
    max_length = max(lengths)
    frame_size = snippets[0].shape[1:]
    dtype = snippets[0].dtype
    n_rows, n_cols = grid_size(n_snippets, frame_size)

    if synchronization_indices is None:
        synchronization_indices = [np.clip(np.arange(max_length), None, length - 1) for length in lengths]
    elif len(synchronization_indices) != n_snippets:
        raise ValueError("Number of synchronization_indices provided doesn't match the number of snippets.")
    
    frames = np.zeros((max_length, frame_size[0] * n_rows, frame_size[1] * n_cols), dtype=dtype)
    for i, stack in enumerate(snippets):
        row_idx = int(i / n_cols)
        col_idx = i % n_cols
        frames[:, row_idx * frame_size[0] : (row_idx + 1) * frame_size[0], col_idx * frame_size[1] : (col_idx + 1) * frame_size[1]] = stack[synchronization_indices[i]]
    return frames


def synchronization_indices(offsets, length, snippet_lengths):
    indices = np.zeros((len(offsets), length), dtype=np.uint)
    for i, offset in enumerate(offsets):
        indices[i, offset:] = np.arange(length - offset)
        indices[i] = np.clip(indices[i], None, snippet_lengths[i] - 1)
    return indices


def beh_overlay(snippets, synchronization_indices):
    lengths = [len(stack) for stack in snippets]
    max_length = max(lengths)
    frame_size = snippets[0].shape[1:]
    dtype = snippets[0].dtype
    frames = np.zeros((max_length,) + frame_size)
    denominator = np.zeros(max_length)
    for i, snippet in enumerate(snippets):
        start = np.where(synchronization_indices[i] == 0)[0][-1]
        stop = np.where(synchronization_indices[i] == synchronization_indices[i][-1])[0][0] + 1
        #frames[start : stop] += snippet
        frames[start : stop] = np.maximum(frames[start : stop], snippet)
        denominator[start : stop] += 1
    #frames = frames / denominator[:, np.newaxis, np.newaxis]
    frames = frames.astype(dtype)
    def frame_generator():
        for frame in frames:
            yield frame
    return frame_generator()
