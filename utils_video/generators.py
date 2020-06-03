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
    match_greatest_resolution,
    plot_df3d_pose,
    ridge_line_plot,
)

def dff(stack):
    vmin = np.percentile(stack, 0.5)
    vmax = np.percentile(stack, 99.5)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.jet
    cbar = colorbar(norm, cmap, (stack.shape[1], -1))

    def frame_generator():
        for frame in stack:
            frame = cmap(norm(frame))
            frame = (frame * 255).astype(np.uint8)
            frame = add_colorbar(frame, cbar, "right")
            yield frame

    return frame_generator()


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
    return dff(frames)


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
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        yield img


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


def stack(generators, axis=0):
    def frame_generator():
        # Extract shapes of images
        shapes = []
        for i, generator in enumerate(generators):
            img = next(generator)
            shapes.append(img.shape[:2])
            generators[i] = itertools.chain([img,], generator)

        # Find target shapes
        shapes = match_greatest_resolution(shapes, axis=(axis + 1) % 2)

        for imgs in zip(*generators):
            # Resize images
            imgs = list(imgs)
            for i, (img, shape) in enumerate(zip(imgs, shapes)):
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                imgs[i] = cv2.resize(img, shape[::-1])

            yield np.concatenate(imgs, axis=axis)

    return frame_generator()

def df3d_3d_points(points3d):
    def generator():
        for pose in points3d:
            img = plot_df3d_pose(pose)
            yield img

    return generator()


def ridge_line(dff_traces, frame_times_2p, frame_times_beh, dt):
    ylim = (np.nanmin(dff_traces), np.nanmax(dff_traces))
    def frame_generator():
        for t in frame_times_beh:
            indices = np.where((frame_times_2p > t - dt) & (frame_times_2p < t + dt))[0]
            start = max(indices[0] - 1, 0)
            stop = min(indices[-1] + 1, dff_traces.shape[1])
            signals = dff_traces[:, start : stop]
            times = frame_times_2p[start : stop]
            
            xlim = (t - dt, t + dt)

            frame = ridge_line_plot(signals, times, vline=t, ylim=ylim, xlim=xlim)
            yield frame

    return frame_generator()


def static_image(image, n_frames):
    def frame_generator():
        for i in range(n_frames):
            yield image

    return frame_generator()


def resample(generator, indices):
    def resampled_generator():
        current_frame_index = 0
        frame = next(generator)
        for i in indices:
            while i > current_frame_index:
                frame = next(generator)
                current_frame_index += 1
            yield frame
    return resampled_generator()

def pad(generator, top, botom, left, right):
    def padded_generator():
        for img in generator:
            padded_image = np.pad(img, ((top, botom), (left, right), (0, 0)), "constant", constant_values=0)
            yield padded_image
    return padded_generator()
