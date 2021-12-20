import glob
import itertools
import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors
import cv2
import PIL
import PIL.ImageFont
import PIL.Image
import PIL.ImageDraw

from .utils import (
    grid_size,
    fig_to_array,
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
    add_dot,
    plot_coxa_positions,
    get_generator_shape,
    plot_df3d_scatter,
    plot_df3d_lines,
)


def dff(stack, size=None, font_size=16, vmin=None, vmax=None, log=False, background="black"):
    if vmin is None:
        vmin = np.percentile(stack, 0.5)
    if vmax is None:
        vmax = np.percentile(stack, 99.5)
    if log:
        if vmin > 0:
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.SymLogNorm(linthresh=0.5, vmin=vmin, vmax=vmax)
    else:
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

    cbar = colorbar(norm, cmap, cbar_shape, font_size=font_size, background=background)

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


def grid(generators, ratio=4/3, allow_different_length=False):
    # Check that all generators have the same frame size.
    shape, generators[0] = get_generator_shape(generators[0])
    frame_size = shape[:2]
    for i, generator in enumerate(generators[1:]):
        current_shape, generators[i + 1] = get_generator_shape(generator)
        current_frame_size = current_shape[:2]
        if not np.all(frame_size == current_frame_size):
            raise ValueError("Generators do not have the same frame size.")

    n_generators = len(generators)
    n_rows, n_cols = grid_size(n_generators, frame_size, ratio=ratio)

    black_image = np.zeros(shape, dtype=np.uint8)
    black_frame_generator = static_image(black_image)
    for i in range(n_rows * n_cols - n_generators):
        generators.append(black_frame_generator)

    rows = []
    for row in range(n_rows):
        row_generator = stack(generators[row * n_cols : (row + 1) * n_cols], axis=1, allow_different_length=allow_different_length)
        rows.append(row_generator)

    grid_generator = stack(rows, axis=0, allow_different_length=allow_different_length)
    return grid_generator


def synchronization_indices(offsets, length, snippet_lengths):
    indices = np.zeros((len(offsets), length), dtype=np.uint)
    for i, offset in enumerate(offsets):
        indices[i, offset:] = np.arange(length - offset)
        indices[i] = np.clip(indices[i], None, snippet_lengths[i] - 1)
    return indices


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


def video(path, size=None, start=0, stop=np.inf):
    try:
        cap = cv2.VideoCapture(path)

        if cap.isOpened() == False:
            raise RuntimeError(f"Error opening video stream or file at {path}.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start
        while cap.isOpened() and frame_idx < stop:
            ret, frame = cap.read()
            if ret == True:
                if size is not None:
                    shape = resize_shape(size, frame.shape[:2])
                    frame = cv2.resize(frame, shape[::-1])
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                yield frame
                frame_idx += 1
            elif ret == False:
                break
    finally:
        cap.release()


def add_text(
    generator,
    text,
    pos=(10, 240),
    font=cv2.FONT_HERSHEY_DUPLEX,
    scale=1,
    color=(255, 255, 255),
    line_type=2,
):
    for i, img in enumerate(generator):
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        if type(text) == str:
            frame_text = text
        else:
            frame_text = text[i]

        for j, line in enumerate(frame_text.split("\n")):
            cv2.putText(
                img, line, (pos[0], pos[1] + j * 40), font, scale, color, line_type
            )
        yield img

def add_text_PIL(
    generator,
    text,
    pos=(10, 240),
    font_file="Arial.ttf",
    color="#FFF",
    size=50,
):
    font = PIL.ImageFont.truetype(font_file, size)
    for i, img in enumerate(generator):
        img = PIL.Image.fromarray(img)
        draw = PIL.ImageDraw.Draw(img)
        if type(text) == str:
            frame_text = text
        else:
            frame_text = text[i]

        draw.text(pos, frame_text, font=font, fill=color)
        draw = PIL.ImageDraw.Draw(img)
        
        img = np.array(img)

        yield img


def _backup_generator(generator):
    """
    keeps a copy of the last output and keeps returning it
    if the next causes an error.
    """
    backup_frame = None
    try:
        for frame in generator:
            backup_frame = frame.copy()
            yield frame
    except Exception as err:
        print("The following exception occurred when accessing the next element of a generator. Returning the previous element indefinitely.")
        print(err)
    while True:
        yield backup_frame


def crop_generator(generator, n):
    """
    Limit the number of returned elements to n.
    """
    for i, item in enumerate(generator):
        if i < n:
            yield item
        else:
            return


def stack(generators, axis=0, allow_different_length=False):
    """
    CAUTION: if you allow_different_length you MUST terminate
    the video using the n_frames parameter of the make_video function.
    Alternatively you can use crop_generator.
    """
    if allow_different_length:
        generators = list(map(_backup_generator, generators))
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


def ridge_line(
    dff_traces, frame_times_2p, frame_times_beh, dt, size=(432, 720), font_size=16
):
    ylim = (np.nanmin(dff_traces), np.nanmax(dff_traces))

    def frame_generator():
        for t in frame_times_beh:
            indices = np.where((frame_times_2p > t - dt) & (frame_times_2p < t + dt))[0]
            start = max(indices[0] - 1, 0)
            stop = min(indices[-1] + 1, dff_traces.shape[1])
            signals = dff_traces[:, start:stop]
            times = frame_times_2p[start:stop]

            xlim = (t - dt, t + dt)

            frame = ridge_line_plot(
                signals,
                times,
                vline=t,
                ylim=ylim,
                xlim=xlim,
                size=size,
                font_size=font_size,
            )
            yield frame

    return frame_generator()


def static_image(image, n_frames=np.inf, size=None):
    if size is not None:
        shape = resize_shape(size, image.shape[:2])
        image = cv2.resize(image, shape[::-1])

    i = 0
    while i < n_frames:
        yield image
        i += 1


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
            padded_image = np.pad(
                img,
                ((top, botom), (left, right), (0, 0)),
                "constant",
                constant_values=0,
            )
            yield padded_image

    return padded_generator()


def dynamics_3D(points, n, size=(432, 216), font_size=16):
    minimums = np.min(points, axis=0)
    maximums = np.max(points, axis=0)

    def generator():
        for i in range(points.shape[0]):
            image = dynamics_3D_plot(
                points[max(0, i - n) : i + 1],
                minimums=minimums,
                maximums=maximums,
                size=size,
                font_size=font_size,
            )
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
    else:
        red_stack = [None for i in range(len(green_stack))]
    if green_stack is not None:
        channels.append("g")
        v_max.append(np.percentile(green_stack, percentiles[1]))
        v_min.append(np.percentile(green_stack, percentiles[0]))
        channels.append("b")
        v_max.append(v_max[-1])
        v_min.append(v_min[-1])
    else:
        green_stack = [None for i in range(len(red_stack))]

    for red_frame, green_frame in zip(red_stack, green_stack):
        frame = rgb(red_frame, green_frame, green_frame, None)
        frame = process_2p_rgb(frame, channels, v_max, v_min)
        frame = frame.astype(np.uint8)
        yield frame


def change_points(generator, change_points, n_pause=1):
    for i, image in enumerate(generator):
        if i in change_points:
            image = add_dot(image)
            for j in range(n_pause):
                yield image
        else:
            yield image


def add_stimulus_dot(generator, stimulus, radius=35, center=(50, 50)):
    for i, image in enumerate(generator):
        if stimulus[i]:
            yield add_dot(image, radius=radius, center=center)
        else:
            yield image


def coxa_locations(points3d, labels=None):
    # allow for multiple experiments to be shown
    if points3d.ndim == 3:
        points3d = points3d[
            np.newaxis,
        ]

    n_exp = points3d.shape[0]

    coxa_indices = np.array([0, 5, 10, 19, 24, 29])
    coxa_points = points3d[:, :, coxa_indices, :]
    coxa_points = coxa_points.reshape((-1, 3))
    centroid = np.mean(coxa_points, axis=0)
    coxa_points = coxa_points - centroid
    U, S, VT = np.linalg.svd(np.transpose(coxa_points))
    # print("U:", U)
    projected_coxa_points = np.transpose(
        np.dot(np.transpose(U), np.transpose(coxa_points))
    )
    mins = np.min(projected_coxa_points, axis=0)
    maxs = np.max(projected_coxa_points, axis=0)
    projected_coxa_points = projected_coxa_points.reshape(
        [n_exp, -1, len(coxa_indices), 3]
    )
    for frame_idx in range(projected_coxa_points.shape[1]):
        yield plot_coxa_positions(
            projected_coxa_points[:, frame_idx], mins, maxs, labels
        )


def df3d_scatter_plots(points3D):
    limits = (
            (np.min(points3D[:, :, 0]), np.max(points3D[:, :, 0])),
            (np.min(points3D[:, :, 1]), np.max(points3D[:, :, 1])),
            (np.min(points3D[:, :, 2]), np.max(points3D[:, :, 2])),
            )
    for frame_points in points3D:
        frame = plot_df3d_scatter(frame_points, limits)
        yield frame

def df3d_line_plots(points3D, connections, colors):
    limits = (
            (np.min(points3D[:, :, 0]), np.max(points3D[:, :, 0])),
            (np.min(points3D[:, :, 1]), np.max(points3D[:, :, 1])),
            (np.min(points3D[:, :, 2]), np.max(points3D[:, :, 2])),
            )
    for frame_points in points3D:
        frame = plot_df3d_lines(frame_points, limits, connections, colors)
        yield frame


def df3d_line_plots_aligned(aligned, fixed_coxa=True):
    n_frames = len(aligned["LF_leg"]["Femur"]["raw_pos_aligned"])
    points3D = np.zeros((n_frames, 30, 3))
    for i, leg in enumerate(["LF_leg", "LM_leg", "LH_leg", "RF_leg", "RM_leg", "RH_leg"]):
        for j, joint in enumerate(["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]):
            if fixed_coxa and joint == "Coxa":
                points3D[:, i * 5 + j, :] = np.tile(aligned[leg][joint]["fixed_pos_aligned"], (n_frames, 1))
            else:
                points3D[:, i * 5 + j, :] = aligned[leg][joint]["raw_pos_aligned"]

    connections = [
              np.array((0, 1, 2, 3, 4), dtype=np.int),
              np.array((5, 6, 7, 8, 9), dtype=np.int),
              np.array((10, 11, 12, 13, 14), dtype=np.int),
              np.array((15, 16, 17, 18, 19), dtype=np.int),
              np.array((20, 21, 22, 23, 24), dtype=np.int),
              np.array((25, 26, 27, 28, 29), dtype=np.int),
              ]
    colors = ["r", "g", "b", "c", "m", "y"]
    return df3d_line_plots(points3D, connections, colors)


def df3d_line_plots_df(df):
    n_frames = df.shape[0]
    points3D = np.zeros((n_frames, 30, 3))
    for i, leg in enumerate(["LF_leg", "LM_leg", "LH_leg", "RF_leg", "RM_leg", "RH_leg"]):
        for j, joint in enumerate(["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]):
            for k, axes in enumerate(["x", "y", "z"]):
                points3D[:, i * 5 + j, k] = df["_".join(["Pose_", leg, joint, axes])].values

    connections = [
              np.array((0, 1, 2, 3, 4), dtype=np.int),
              np.array((5, 6, 7, 8, 9), dtype=np.int),
              np.array((10, 11, 12, 13, 14), dtype=np.int),
              np.array((15, 16, 17, 18, 19), dtype=np.int),
              np.array((20, 21, 22, 23, 24), dtype=np.int),
              np.array((25, 26, 27, 28, 29), dtype=np.int),
              ]
    colors = ["r", "g", "b", "c", "m", "y"]
    return df3d_line_plots(points3D, connections, colors)


def df3d_line_plots_comparison(points3D, connections, colors, linestyles, labels, title=None):
    if type(points3D) != list:
        raise TypeError("points3D must be a list of numpy arrays.")
    points3D = np.array(points3D)

    limits = (
            (np.nanmin(points3D[:, :, :, 0]), np.nanmax(points3D[:, :, :, 0])),
            (np.nanmin(points3D[:, :, :, 1]), np.nanmax(points3D[:, :, :, 1])),
            (np.nanmin(points3D[:, :, :, 2]), np.nanmax(points3D[:, :, :, 2])),
            )
    for frame_index in range(points3D.shape[1]):
        frame_points = [points3D[i, frame_index] for i in range(points3D.shape[0])]
        frame = plot_df3d_lines(frame_points, limits, connections, colors, labels=labels, linestyles=linestyles, title=title)
        yield frame


def optical_flow(pitch, yaw, roll, times, window_size, ylims=None, frame_times=None):
    if frame_times is None:
        frame_times = times

    fig, axes = plt.subplots(3, 1, sharex=True)
  
    if ylims is None:
        ylims = (
                 (np.min(pitch), np.max(pitch)),
                 (np.min(yaw), np.max(yaw)),
                 (np.min(roll), np.max(roll)),
                )
    for i in range(3):
        axes[i].set_ylim(ylims[i])
   
    axes[0].set_ylabel("Pitch")
    axes[1].set_ylabel("Yaw")
    axes[2].set_ylabel("Roll")

    axes[2].set_xlabel("Time [s]")

    axes[0].plot(times, pitch)
    axes[1].plot(times, yaw)
    axes[2].plot(times, roll)
   
    vlines = [axes[i].plot([frame_times[0], frame_times[0]], [ylims[i][0], ylims[i][1]], linestyle="dashed")[0] for i in range(3)]

    dt = window_size / 2
    for i, t in enumerate(frame_times):
        for i in range(3):
            axes[i].set_xlim(t-dt, t+dt)
            vlines[i].set_data([t, t], [ylims[i][0], ylims[i][1]])
        frame = fig_to_array(fig)    
        if i == len(times) - 1:
            plt.close()
        yield frame

