import glob
import itertools

import numpy as np
from matplotlib import pyplot as plt
import cv2

from .utils import (
    grid_size,
    colorbar,
    add_colorbar,
    load_video,
    natsorted,
    resize_shape,
    find_greatest_common_resolution,
)


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
    frames = add_colorbar(frames, cbar, "right")

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


def merge_videos(paths, synchronization_indices=None, sort=False):
    """
    This function returns a generator that
    yields frames with the given files.

    Parameters
    ----------
    paths : list of strings
        List of the paths to the video files that should be merged.
    sort : boolean
        Whether to sort the paths according to natural order.

    Returns
    -------
    frame_generator : generator
        A generator that yields individual
        video frames.
    """
    if sort:
        paths = natsorted(paths)
    snippets = [load_video(path) for path in paths]

    frames = _grid_frames(snippets, synchronization_indices)

    def frame_generator():
        for frame in frames:
            yield frame

    return frame_generator()


def _grid_frames(snippets, synchronization_indices=None):
    # Check that all snippets have the same frame size.
    if not len(set([snippet.shape[1] for snippet in snippets])) and len(
        set([snippet.shape[2] for snippet in snippets])
    ):
        raise ValueError("Snippets do not have the same frame size.")

    n_snippets = len(snippets)
    lengths = [len(stack) for stack in snippets]
    max_length = max(lengths)
    frame_size = snippets[0].shape[1:]
    if np.all([snippet.ndim == 4 for snippet in snippets]):
        n_channels = snippets[0].shape[-1]
        frame_size = frame_size[:-1]
    elif np.all([snippet.ndim == 3 for snippet in snippets]):
        n_channels = 1
    else:
        raise ValueError("Snippets don't have the same number of channels.")
    dtype = snippets[0].dtype
    n_rows, n_cols = grid_size(n_snippets, frame_size)

    if synchronization_indices is None:
        synchronization_indices = [
            np.clip(np.arange(max_length), None, length - 1) for length in lengths
        ]
    elif len(synchronization_indices) != n_snippets:
        raise ValueError(
            "Number of synchronization_indices provided doesn't match the number of snippets."
        )

    frames = np.zeros(
        (max_length, frame_size[0] * n_rows, frame_size[1] * n_cols, n_channels),
        dtype=dtype,
    )
    frames = np.squeeze(frames)
    for i, stack in enumerate(snippets):
        row_idx = int(i / n_cols)
        col_idx = i % n_cols
        frames[
            :,
            row_idx * frame_size[0] : (row_idx + 1) * frame_size[0],
            col_idx * frame_size[1] : (col_idx + 1) * frame_size[1],
        ] = stack[synchronization_indices[i]]
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
        stop = (
            np.where(synchronization_indices[i] == synchronization_indices[i][-1])[0][0]
            + 1
        )
        # frames[start : stop] += snippet
        frames[start:stop] = np.maximum(frames[start:stop], snippet)
        denominator[start:stop] += 1
    # frames = frames / denominator[:, np.newaxis, np.newaxis]
    frames = frames.astype(dtype)

    def frame_generator():
        for frame in frames:
            yield frame

    return frame_generator()


def images(path):
    images = glob.glob(path)
    if len(images) == 0:
        raise FileNotFoundError(f"No files match {path}.")
    images = natsorted(images)
    for image_path in images:
        yield cv2.imread(image_path)


def add_text(
    generator,
    text,
    pos=(10, 240),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=1,
    color=(255, 255, 255),
    line_type=2,
):
    for i, img in enumerate(generator):
        for j, line in enumerate(text[i].split("\n")):
            cv2.putText(
                img, line, (pos[0], pos[1] + j * 40), font, scale, color, line_type
            )
        yield img


def stack(generators):
    def frame_generator():
        # Extract shapes of images
        shapes = []
        for i, generator in enumerate(generators):
            img = next(generator)
            shapes.append(img.shape[:2])
            generators[i] = itertools.chain([img,], generator)

        # Find target shapes
        shapes = find_greatest_common_resolution(shapes, axis=1)

        for imgs in zip(*generators):
            # Resize images
            imgs = list(imgs)
            for i, (img, shape) in enumerate(zip(imgs, shapes)):
                imgs[i] = cv2.resize(img, shape[::-1])

            yield np.concatenate(imgs, axis=0)

    return frame_generator()
