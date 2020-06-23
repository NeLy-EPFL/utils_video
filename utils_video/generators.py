import glob
import itertools
import math

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
    dynamics_3D_plot,
    rgb,
    process_2p_rgb,
    plot_coxa_positions,
)

def dff(stack, size=None, font_size=16):
    vmin = np.percentile(stack, 0.5)
    vmax = np.percentile(stack, 99.5)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.jet

    if size is None:
        image_shape = stack.shape[1:3]
        cbar_shape = (stack.shape[1], max(math.ceil(stack.shape[2] * 0.1), 150))
    else: 
        cbar_width = max(math.ceil(size[1] * 0.1), 150)
        image_shape = (size[0], size[1] - cbar_width)
        image_shape = resize_shape(image_shape, stack.shape[1:3])
        cbar_shape = (image_shape[0], cbar_width)
    
    cbar = colorbar(norm, cmap, cbar_shape, font_size=font_size)

    def frame_generator():
        for frame in stack:
            frame = cmap(norm(frame))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.resize(frame, image_shape[::-1])
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


def images(path, size=None, start=0):
    images = glob.glob(path)
    if len(images) == 0:
        raise FileNotFoundError(f"No files match {path}.")
    images = natsorted(images)
    images = images[start:]
    for image_path in images:
        img = cv2.imread(image_path)
        if size is not None:
            shape = resize_shape(size, img.shape[:2])
            img = cv2.resize(img, shape[::-1])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        yield img


def video(path, size=None, start=0):
    try:
        cap = cv2.VideoCapture(path)

        if (cap.isOpened() == False):
            raise RuntimeError(f"Error opening video stream or file at {path}.")

        current_frame = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True and current_frame >= start:
                if size is not None:
                    shape = resize_shape(size, frame.shape[:2])
                    frame = cv2.resize(frame, shape[::-1])
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                yield frame
            elif ret == False:
                break
    finally:
        cap.release()


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


def ridge_line(dff_traces, frame_times_2p, frame_times_beh, dt, size=(432, 720), font_size=16):
    ylim = (np.nanmin(dff_traces), np.nanmax(dff_traces))
    def frame_generator():
        for t in frame_times_beh:
            indices = np.where((frame_times_2p > t - dt) & (frame_times_2p < t + dt))[0]
            start = max(indices[0] - 1, 0)
            stop = min(indices[-1] + 1, dff_traces.shape[1])
            signals = dff_traces[:, start : stop]
            times = frame_times_2p[start : stop]
            
            xlim = (t - dt, t + dt)

            frame = ridge_line_plot(signals, times, vline=t, ylim=ylim, xlim=xlim, size=size, font_size=font_size)
            yield frame

    return frame_generator()


def static_image(image, n_frames, size=None):
    if size is not None:
        shape = resize_shape(size, image.shape[:2])
        image = cv2.resize(image, shape[::-1])
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

def dynamics_3D(points, n, size=(432, 216), font_size=16):
    minimums = np.min(points, axis=0)
    maximums = np.max(points, axis=0)
    def generator():
        for i in range(points.shape[0]):
            image = dynamics_3D_plot(points[max(0, i - n) : i + 1], minimums=minimums, maximums=maximums, size=size, font_size=font_size)
            yield image
    return generator()

def frames_2p(red_stack, green_stack, percentiles=(5, 95)):
    channels = []
    v_max = []
    v_min = []
    if red_stack is not None:
        channels.append("r")
        v_max.append(np.percentile(red_stack, percentiles[1]))
        v_min.append(np.percentile(red_stack, percentiles[0]))
    if green_stack is not None:
        channels.append("g")
        v_max.append(np.percentile(green_stack, percentiles[1]))
        v_min.append(np.percentile(green_stack, percentiles[0]))
        channels.append("b")
        v_max.append(v_max[-1])
        v_min.append(v_min[-1])

    
    for red_frame, green_frame in zip(red_stack, green_stack):
        frame = rgb(red_frame, green_frame, green_frame, None)
        frame = process_2p_rgb(frame, channels, v_max, v_min)
        frame = frame.astype(np.uint8)
        yield frame

def coxa_locations(points3d, labels=None):
    # allow for multiple experiments to be shown
    if points3d.ndim == 3:
        points3d = points3d[np.newaxis,]

    n_exp = points3d.shape[0]
        
    coxa_indices = np.array([0, 5, 10, 19, 24, 29])
    coxa_points = points3d[:, :, coxa_indices, :]
    coxa_points = coxa_points.reshape((-1, 3))
    centroid = np.mean(coxa_points, axis=0)
    coxa_points = coxa_points - centroid
    U, S, VT = np.linalg.svd(np.transpose(coxa_points))
    #print("U:", U)
    projected_coxa_points = np.transpose(np.dot(np.transpose(U), np.transpose(coxa_points)))
    mins = np.min(projected_coxa_points, axis=0)
    maxs = np.max(projected_coxa_points, axis=0)
    projected_coxa_points = projected_coxa_points.reshape([n_exp, -1, len(coxa_indices), 3])
    for frame_idx in range(projected_coxa_points.shape[1]):
        yield plot_coxa_positions(projected_coxa_points[:, frame_idx], mins, maxs, labels) 
