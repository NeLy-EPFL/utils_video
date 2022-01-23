import math
import re
import itertools

import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np

#from pandas.plotting._tools import _subplots, _flatten

import deepfly.plot_util

dpi = 100
img3d_aspect = (2, 2)


def get_generator_shape(generator):
    img = next(generator)
    shape = img.shape
    generator = itertools.chain([img,], generator)
    return shape, generator


def resize_shape(shape, original_shape, allow_upsampling=False):
    """
    This function converts an image shape into
    a new size respecting the ratio between
    width and hight.

    Parameters
    ----------
    shape : tuple of two integers
        Desired shape. The tuple and contain one, two
        or no -1 entry. If no entry is -1, this argument
        is returned. If both entries are -1, `original_shape`
        is returned. If only one of the entires is -1, its new
        value in `new_shape` is calculated preserving the ratio
        of `original_shape`.
    original_shape : tuple of two integers
        Original shape.

    Returns
    -------
    new_shape : tuple of two integers
        Resized shape.
    """
    if len(shape) != 2:
        raise ValueError("shape has to be of length 2.")
    if len(original_shape) != 2:
        raise ValueError("original_shape has to be of length 2.")
    if shape[0] % 1 != 0 or shape[1] % 1 != 0:
        raise ValueError("Entries of shape have to be integers.")
    if original_shape[0] % 1 != 0 or original_shape[1] % 1 != 0:
        raise ValueError("Entries of original_shape have to be integers.")
    if np.any(np.array(shape) < -1):
        raise ValueError("The values of shape cannot be smaller than -1.")
    if np.any(np.array(original_shape) < -1):
        raise ValueError("The values of original_shape cannot be smaller than -1.")

    if shape[0] == -1 and shape[1] == -1:
        new_shape = original_shape
    elif shape[0] == -1:
        ratio = original_shape[0] / original_shape[1]
        new_shape = (int(shape[1] * ratio), shape[1])
    elif shape[1] == -1:
        ratio = original_shape[1] / original_shape[0]
        new_shape = (shape[0], int(shape[0] * ratio))
    else:
        new_shape = shape
    if not allow_upsampling:
        if new_shape[0] > original_shape[0] and new_shape[1] > original_shape[1]:
            return original_shape
    return new_shape


def match_greatest_resolution(shapes, axis):
    """
    This function finds the greatest resolution of
    all the given shapes along the given axis.
    It then converts all the other shapes to match the
    greatest resolution along the given axis, respecting
    the aspect ratio when resizing the other axes.

    Parameters
    ----------
    shapes : list or tuple of 2-tuples of integers
        List of the shapes of the different images.
    axis : int
        The integer specifying the axis

    Returns
    -------
    shapes : list of 2-tuples of integers
        New resized shapes with all shapes along axis
        being equal to the maximum along this axis.
    """
    # Find highest resolution
    target_resolution = np.max(shapes, axis=0)

    # Prepare resolution for resize_shape based on axis
    desired_resolution = [-1, -1]
    desired_resolution[axis] = target_resolution[axis]

    # make shapes mutable so they can be overwritten by resized shapes
    shapes = list(shapes)

    # find shapes for resizing
    for i, shape in enumerate(shapes):
        shapes[i] = resize_shape(desired_resolution, shape, allow_upsampling=True)

    return shapes


def grid_size(n_elements, element_size, ratio=4 / 3):
    """
    This function computes the number of rows and
    columns to fit elements next to each other
    while trying to be as close to the given ratio
    as possible (default is 4:3).

    Parameters
    ----------
    n_elements : int
        Number of elements to fit.
    element_size : tuple of two integers
        Shape of a single element.
    ratio : float
        Target ratio. Default is 4:3.

    Returns
    -------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    """
    if not isinstance(n_elements, int):
        raise ValueError("n_elements has to be of type int.")
    if not len(element_size) == 2:
        raise ValueError("element_size has to be of length 2.")
    target_ratio = ratio / (element_size[1] / element_size[0])
    n_rows = int(round(np.sqrt(n_elements / target_ratio)))
    n_cols = math.ceil(n_elements / n_rows)
    return n_rows, n_cols


def fig_to_array(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def colorbar(norm, cmap, size, orientation="vertical", font_size=16, background="black",
             label=r"%$\frac{\Delta F}{F}$"):
    if orientation not in ["horizontal", "vertical"]:
        raise ValueError("""orientation can only be "horizontal" or "vertical".""")

    figsize = (size[1] / dpi, size[0] / dpi)

    if background == "black":
        config = {
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "black",
            "font.size": font_size,
            "text.color": "white",
        }
    elif background == "white":
        config = {
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "font.size": font_size,
            "text.color": "black",
        }
    else:
        raise ValueError(f"Unknown background color {background}")

    with plt.rc_context(
        config
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plt.imshow(np.random.rand(100).reshape((10, 10)))
        color_bar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation=orientation,
            fraction=1,
        )
        color_bar.ax.tick_params(labelsize=font_size)
        if orientation == "horizontal":
            color_bar.ax.set_xlabel(label, rotation=0, color="white", fontsize=font_size)
        else:
            color_bar.ax.set_ylabel(
                label, rotation=0, color=config["text.color"], labelpad=15, fontsize=font_size
            )
        ax.remove()
        data = fig_to_array(fig)
        plt.close()
        data = cv2.resize(data, size[::-1])
    return data


def add_colorbar(frames, cbar, pos, alpha=255):
    if frames.dtype != cbar.dtype:
        raise ValueError("frames and cbar need to have the same dtype.")
    # If frames is a single image, convert it to a sequence with one image.
    if frames.ndim < 4:
        frames = frames[np.newaxis]
    # Check that frames and cbar both have 4 channels.
    if frames.shape[-1] == 3:
        extra_channel = np.ones(frames.shape[:-1] + (1,), dtype=frames.dtype) * alpha
        frames = np.concatenate((frames, extra_channel), axis=-1)
    if cbar.shape[-1] == 3:
        extra_channel = np.ones(cbar.shape[:-1] + (1,), dtype=cbar.dtype) * alpha
        cbar = np.concatenate((cbar, extra_channel), axis=-1)
    cbar = np.tile(cbar, (frames.shape[0], 1, 1, 1))
    if pos == "right":
        with_color_bar = np.concatenate((frames, cbar), axis=2)
    elif pos == "left":
        with_color_bar = np.concatenate((cbar, frames), axis=2)
    elif pos == "bottom":
        with_color_bar = np.concatenate((frames, cbar), axis=1)
    elif pos == "top":
        with_color_bar = np.concatenate((cbar, frames), axis=1)
    return np.squeeze(with_color_bar)


def load_video(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened() == False:
        raise ValueError(f"Could not open {path}.")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    cap.release()
    return np.array(frames)


def natsorted(list_of_strs):
    """
    Sorts a list of strings in natural order.

    Parameters
    ----------
    list_of_strs : list of strings
        List to be sorted.

    Returns
    -------
    sorted_l : list of strings
        Naturally sorted list.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(list_of_strs, key=alphanum_key)


def plot_df3d_pose(points3d):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=img3d_aspect, dpi=dpi)
    fig.tight_layout(pad=0)
    ax3d = Axes3D(fig)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    deepfly.plot_util.plot_drosophila_3d(ax3d, points3d.copy(), cam_id=2, lim=2)

    data = fig_to_array(fig)
    plt.close()
    return data


def roi_image(background, mask, connectivity=4, min_size=0, cm="autumn"):
    """
    This function overlays the ROIs in the mask on the background
    and gives each ROI a different color.

    Parameters
    ----------
    background : numpy array 2D
        Background image.
    mask : numpy array 2D
        ROI mask.
    connectivity : int
        Connectivity of connected components. Can be 4 or 8.
    min_size : int
        Minimum size of a connected component to be considered a ROI.
    cm : string
        Name of the matplotlib color map used.
    
    Returns
    -------
    img : numpy array 3D
        Output image.
    colors : numpy array 2D
        Colors in the order of the ROIs.
    """
    # Make sure mask has only one channel
    if mask.ndim == 3:
        mask = np.sum(mask, axis=-1)

    # Make sure background only has one channel
    if background.ndim == 3:
        background = np.sum(background, axis=-1)

    # Check that background and mask are image of the same size
    if background.shape != mask.shape:
        raise ValueError(
            "Dimensions of the image given for background and mask do not match."
        )

    background = background / np.max(background) * 255
    img = np.zeros(background.shape + (3,))
    img[:, :, 0] = background
    img[:, :, 1] = background
    img[:, :, 2] = background

    # Ensure mask is uint8 for cv2's connected components
    mask = mask.astype(np.uint8)

    _, label_img, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    (selected_components,) = np.where(np.array(stats)[:, 4] > min_size)
    n_rois = len(selected_components)
    cm = plt.get_cmap(cm)
    colors = [cm(1.0 * i / n_rois) for i in range(n_rois)]
    for roi_num, label in enumerate(selected_components):
        if label == 0:
            continue
        mask_one_roi = np.where(label_img == label)
        img[mask_one_roi + (np.ones(len(mask_one_roi[0]), dtype=np.int) * 0,)] = (
            colors[roi_num][0] * 255
        )
        img[mask_one_roi + (np.ones(len(mask_one_roi[0]), dtype=np.int) * 1,)] = (
            colors[roi_num][1] * 255
        )
        img[mask_one_roi + (np.ones(len(mask_one_roi[0]), dtype=np.int) * 2,)] = (
            colors[roi_num][2] * 255
        )
    return img.astype(np.uint8), colors


def ridge_line_plot(
    signals,
    t,
    colormap=plt.get_cmap("autumn"),
    ylim="max",
    xlim="fit",
    vline=False,
    overlap=1,
    size=(720, 432),
    font_size=16,
    background="black",
):
    """
    This function creates a ridge line plot of all signals.
    Parts of the code are from the joypy module.

    Parameters
    ----------
    signals : 2D numpy array
        First dimension: neuron id
        Second dimension: time
    t : numpy array
        Times.
    color_map : matplotlib colormap
        Colormap used to color the separate lines.
    ylim : string or tuple of float
        The y limits of the shared y axes.
        If ylim is 'max', the y limits are set to the maximum and minimum
        of all signals.
        Default is 'max'.
    tick_spacing : float
        Distance between two ticks on the x axis.
        Default is 0.5.
    vline : float
        Plot a vertical line to mark the given value.
    overlap : float
        Amount of overlap between figures.
    figsize : tuple
        See matplot lib documentation for details.

    Returns
    -------
    fig : pyplot figure
        Figure with the ridge_line_plot.
    """

    if ylim == "max":
        ylim = (min(signals), max(signals))
    if xlim == "fit":
        xlim = (np.min(t), np.max(t))
    num_axes = signals.shape[0]
    clip_on = True

    if background == "black":
        config = {
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "black",
            "font.size": font_size,
            "text.color": "white",
        }
    elif background == "white":
        config = {
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "font.size": font_size,
            "text.color": "black",
        }
    else:
        raise ValueError(f"Unknown background color {background}.")

    with plt.rc_context(config):
        fig_ridge, axes = _subplots(
            naxes=num_axes,
            squeeze=False,
            sharex=True,
            sharey=False,
            figsize=(size[1] / dpi, size[0] / dpi),
            dpi=dpi,
            layout_type="vertical",
        )
        _axes = _flatten(axes)
        for i in range(num_axes):
            a = _axes[i]
            a.fill_between(
                t, 0.0, signals[i], clip_on=clip_on, color=colormap(i / num_axes)
            )
            a.plot(t, [0.0] * len(t), clip_on=clip_on, color=colormap(i / num_axes))
            a.plot(t, signals[i], clip_on=clip_on, color="k")
            a.set_ylim(ylim)
            a.set_xlim(xlim)
            if not i % 10:
                a.set_yticks([0])
                a.set_yticklabels([str(i)])
            else:
                a.set_yticks([])
            a.patch.set_alpha(0)
            a.tick_params(axis="both", which="both", length=0, pad=10)
            a.xaxis.set_visible(False)
            a.set_frame_on(False)

        _axes[-1].xaxis.set_visible(True)
        _axes[-1].tick_params(
            axis="x", which="both", length=5, pad=10, labelrotation=45
        )

        _axes[-1].set_xlabel("Time (s)", color="white")
        _axes[int(num_axes / 2)].set_ylabel("Neuron", color=config["text.color"], labelpad=40)

        h_pad = 5 + (-5 * (1 + overlap))
        fig_ridge.tight_layout(h_pad=h_pad)

        if vline is not None:
            for i in range(0, len(axes), 1):
                _axes[i].axvline(
                    vline, linestyle="-", color="white", linewidth=1.5, clip_on=False
                )
    data = fig_to_array(fig_ridge)
    plt.close()
    return data


def dynamics_3D_plot(points, minimums, maximums, size=(432, 720), font_size=16):
    with plt.rc_context(
        {
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "black",
            "font.size": font_size,
        }
    ):

        fig = plt.figure(figsize=(size[1] / dpi, size[0] / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d", facecolor="black")
        # Some times warns because of zero divide (seems to be related with plotting several points at once if
        # all points are plotted with separate calls of ax.scatter no warning is raised.
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.arange(points.shape[0]))
        ax.scatter(minimums[0], points[-1, 1], points[-1, 2], c="k")
        ax.scatter(points[-1, 0], maximums[1], points[-1, 2], c="k")
        ax.scatter(points[-1, 0], points[-1, 1], minimums[2], c="k")
        ax.plot(points[:, 0], points[:, 1], points[:, 2])
        ax.scatter(
            points[:-1, 0],
            points[:-1, 1],
            points[:-1, 2],
            c=np.arange(points.shape[0] - 1),
            s=1,
        )
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c="r")
        ax.set_xlim3d(minimums[0], maximums[0])
        ax.set_ylim3d(minimums[1], maximums[1])
        ax.set_zlim3d(minimums[2], maximums[2])
        ax.set_xlabel("PC 1", color="white", labelpad=15)
        ax.set_ylabel("PC 2", color="white", labelpad=15)
        ax.set_zlabel("PC 3", color="white", rotation=90, labelpad=15)

        data = fig_to_array(fig)
        plt.close()
    return data


def process_2p_rgb(img, channel, v_max, v_min):
    if isinstance(channel, str):
        channel = (channel,)
        v_max = (v_max,)
        v_min = (v_min,)
    else:
        assert len(channel) == len(v_max)
        assert len(channel) == len(v_min)
    for i, c in enumerate(channel):
        if c == "r":
            img[:, :, 0] = (
                (np.clip(img[:, :, 0], v_min[i], v_max[i]) - v_min[i])
                / (v_max[i] - v_min[i])
                * 255
            )
        if c == "g":
            img[:, :, 1] = (
                (np.clip(img[:, :, 1], v_min[i], v_max[i]) - v_min[i])
                / (v_max[i] - v_min[i])
                * 255
            )
        elif c == "b":
            img[:, :, 2] = (
                (np.clip(img[:, :, 2], v_min[i], v_max[i]) - v_min[i])
                / (v_max[i] - v_min[i])
                * 255
            )
    return img


def rgb(red, green, blue, alpha):
    """
    Merges channels by creating a new
    axis of lengths 3 or four.

    Parameters
    ----------
    red, green, blue, alpha : numpy arrays
        Different channels. Can be None.

    Returns
    -------
    rgb_array : numpy array
        The RGB stack.
    """
    n_channels = 4 if alpha is not None else 3

    if red is not None:
        shape = red.shape
        dtype = red.dtype
    else:
        shape = green.shape
        dtype = green.dtype

    rgb_array = np.zeros((n_channels,) + shape, dtype=dtype)
    channel_axis = len(shape)
    if red is not None:
        rgb_array[0] = red
    if green is not None:
        rgb_array[1] = green
    if blue is not None:
        rgb_array[2] = blue
    if alpha is not None:
        rgb_array[3] = alpha
    rgb_array = np.rollaxis(rgb_array, 0, len(rgb_array.shape))
    return rgb_array


def add_dot(image, radius=35, center=(50, 50), color=(255, 0, 0)):
    image = cv2.circle(image, center, radius, color, -1)
    return image


def plot_coxa_positions(points, mins, maxs, labels=None):
    padding = np.abs(maxs) * 0.1
    mins = mins - padding
    maxs = maxs + padding
    if points.ndim == 2:
        points = points[
            np.newaxis,
        ]
    if labels == None:
        labels = [None,] * points.shape[0]
    fig, axes = plt.subplots(2, 1, sharex=True)
    for i, exp_points in enumerate(points):
        axes[0].scatter(exp_points[:, 0], exp_points[:, 1], label=labels[i])
        axes[0].set_xlim((mins[0], maxs[0]))
        axes[0].set_ylim((mins[1], maxs[1]))
        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
        axes[1].scatter(exp_points[:, 0], exp_points[:, 2])
        axes[1].set_xlim((mins[0], maxs[0]))
        axes[1].set_ylim((mins[2], maxs[2]))
    data = fig_to_array(fig)
    plt.close()
    return data


def plot_df3d_scatter(points, limits):
    fig, axes = plt.subplots(1, 3, figsize=(27,9))
    plot_axes = {0: (0, 1), 1: (0, 2), 2: (1, 2)}
    half = int(points.shape[0] / 2)
    for i in range(3):
        x = points[:, plot_axes[i][0]]
        y = points[:, plot_axes[i][1]]
        axes[i].scatter(x[:half], y[:half])
        axes[i].scatter(x[half:], y[half:])
        x_limits = limits[plot_axes[i][0]]
        y_limits = limits[plot_axes[i][1]]
        axes[i].set_xlim(x_limits)
        axes[i].set_ylim(y_limits)
    data = fig_to_array(fig)
    plt.close()
    return data

def plot_df3d_lines(points, limits, connections, colors, labels=None, linestyles=["solid",], title=None):
    """
    coonections: [np.array((0, 1), dtype=np.int), np.array((2, 3), dtype=np.int)
    colors: one color for each connection
    """
    fig, axes = plt.subplots(1, 3, figsize=(27,9))
    plot_axes = {0: (0, 1), 1: (0, 2), 2: (1, 2)}

    if type(points) == list:
        points = np.array(points)
    elif type(points) == np.ndarray:
        points = np.expand_dims(points, axis=0)
    else:
        raise TypeError("points has to be a list of numpy array if multiple flies should be plotted on top of each other or a numpy array for a single fly.")

    if type(colors[0]) != list:
        colors = [colors,] * len(points)

    for i in range(3):
        for j in range(points.shape[0]):
            x = points[j, :, plot_axes[i][0]]
            y = points[j, :, plot_axes[i][1]]
            for c, connection in enumerate(connections):
                axes[i].plot(x[connection], y[connection], color=colors[j][c], linestyle=linestyles[j], linewidth=4)
            x_limits = limits[plot_axes[i][0]]
            y_limits = limits[plot_axes[i][1]]
            axes[i].set_xlim(x_limits)
            axes[i].set_ylim(y_limits)
    if labels is not None:
        if len(labels) != len(linestyles):
            raise ValueError("Length of labels and linestyles has to match.")
        for linestyle, label in zip(linestyles, labels):
            axes[2].plot((x_limits[0], x_limits[0]), y_limits, color="k", linestyle=linestyle, label=label, alpha=1)
        lgd = axes[2].legend(fontsize=16)
        #for lh in lgd.legendHandles:
        #    lh._legmarker.set_alpha(1)
    if title is not None:
        fig.suptitle(title, fontsize=24)
    data = fig_to_array(fig)
    plt.close()
    return data

