import math
import re

import cv2
from matplotlib import pyplot as plt
import numpy as np


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
        

def find_greatest_common_resolution(shapes, axis):
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
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
    if orientation == "horizontal":
        data = data[750:900, :]
    else:
        data = data[:, 750:]
    size = resize_shape(size, data.shape)
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
        return np.concatenate((frames, cbar), axis=2)
    elif pos == "left":
        return np.concatenate((cbar, frames), axis=2)
    elif pos == "bottom":
        return np.concatenate((frames, cbar), axis=1)
    elif pos == "top":
        return np.concatenate((cbar, frames), axis=1)

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
