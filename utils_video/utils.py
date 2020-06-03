import math
import re

import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from pandas.plotting._tools import _subplots, _flatten

import deepfly.plot_util

img3d_dpi = 100
img3d_aspect = (2, 2)


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


def grid_size(n_elements, element_size, ratio=4/3):
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


def colorbar(norm, cmap, size, orientation="vertical"):
    if orientation == "horizontal":
        figsize = (math.ceil(size[1] / 100), 10)
    elif orientation == "vertical":
        figsize = (10, math.ceil(size[0] / 100))
    else:
        raise ValueError("""orientation can only be "horizontal" or "vertical".""")

    with plt.rc_context({"axes.edgecolor": "white", "xtick.color": "white", "ytick.color": "white", "figure.facecolor": "black", "font.size": 18,}):
        fig, ax = plt.subplots(figsize=figsize)
        plt.imshow(np.random.rand(100).reshape((10, 10)))
        color_bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation=orientation)
        if orientation == "horizontal":
            color_bar.ax.set_xlabel(r"%$\frac{\Delta F}{F}$", rotation=0, color="white")
        else:
            color_bar.ax.set_ylabel(r"%$\frac{\Delta F}{F}$", rotation=0, color="white")
        ax.remove()
        data = fig_to_array(fig)
        plt.close()
    if orientation == "horizontal":
        data = data[750:900, :]
    else:
        data = data[:, 750:]
    size = resize_shape(size, data.shape[:-1])
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
    fig = plt.figure(figsize=img3d_aspect, dpi=img3d_dpi)
    fig.tight_layout(pad=0)
    ax3d = Axes3D(fig)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    deepfly.plot_util.plot_drosophila_3d(ax3d, points3d.copy(), cam_id=2, lim=2)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
        raise ValueError("Dimensions of the image given for background and mask do not match.")

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
    selected_components, = np.where(np.array(stats)[:, 4] > min_size)
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
    figsize=(6, 10),
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
    with plt.rc_context(
        {
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "black",
            "font.size": 16,
        }
    ):
        fig_ridge, axes = _subplots(
            naxes=num_axes,
            squeeze=False,
            sharex=True,
            sharey=False,
            figsize=figsize,
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
        _axes[-1].tick_params(axis="x", which="both", length=5, pad=10)

        _axes[-1].set_xlabel("time [s]", color="white")
        _axes[int(num_axes / 2)].set_ylabel("Neuron", color="white")

        h_pad = 5 + (-5 * (1 + overlap))
        fig_ridge.tight_layout(h_pad=h_pad)

        if vline is not None:
            for i in range(0, len(axes), 1):
                _axes[i].axvline(
                    vline, linestyle="-", color="white", linewidth=1.5, clip_on=False
                )
    fig_ridge.canvas.draw()
    data = np.frombuffer(fig_ridge.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig_ridge.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
