from typing import Dict, List, Optional, Union

import numpy as np
import cv2
from matplotlib import cm

from habitat.utils.visualizations.utils import tile_images


def colorize_depth(
    depth: np.ndarray,
    d_min=None,
    d_max=None,
    colormap="viridis",
):
    """Colorize depth map.

    Args:
        depth: [H, W]
        d_min: If None, computed from @depth
        d_max: If None, computed from @depth
        colormap: color map name supported by matplotlib

    Returns:
        np.ndarray: [H, W, 3] np.uint8 image

    References:
        https://github.com/dwofk/fast-depth/blob/master/deploy/data/visualize.py
    """
    cmap = cm.get_cmap(colormap)
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_color = (255 * cmap(depth_relative)[..., :3]).clip(0, 255)
    return depth_color.astype(np.uint8)


def draw_border(
    img: np.ndarray,
    color=(255, 0, 0),
    width: Union[float, int] = 0.05,
    alpha=1.0,
):
    """Draw translucent strips on the border (in-place).

    Args:
        img: [H, W, 3]
        color: [3]
        width: float for ratio and int for value
        alpha: opacity of strip. 1 is completely non-transparent.

    Returns:
        np.ndarray: @img drawn with strips.

    References:
        habitat-lab/habitat/utils/visualizations/utils.py::draw_collision
    """
    if isinstance(width, float):
        width = int(min(img.shape[0:2]) * width)

    mask = np.ones(img.shape[0:2], dtype=bool)
    mask[width:-width, width:-width] = 0
    # NOTE(jigu): dtype conversion will happen implicitly during assignment.
    img[mask] = np.clip(
        alpha * np.array(color) + ((1.0 - alpha) * img)[mask], 0, 255
    )
    return img


def put_text_on_image(image: np.ndarray, lines: List[str]):
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()

    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 255, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def append_text_to_image(image: np.ndarray, lines: List[str]):
    r"""Appends text left to an image of size (height, width, channels).
    The returned image has white text on a black background.

    Args:
        image: the image to put text
        text: a string to display

    Returns:
        A new image with text inserted left to the input image

    See also:
        habitat.utils.visualization.utils
    """
    # h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    # text_image = blank_image[0 : y + 10, 0:w]
    # final = np.concatenate((image, text_image), axis=0)
    final = np.concatenate((blank_image, image), axis=1)
    return final


def put_info_on_image(
    image, info: Dict[str, float], extras=None, overlay=True
):
    lines = [f"{k}: {v:.3f}" for k, v in info.items()]
    if extras is not None:
        lines.extend(extras)
    if overlay:
        return put_text_on_image(image, lines)
    else:
        return append_text_to_image(image, lines)


def to_array(x):
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    return x


def observations_to_image(observations, extra_images=None):
    render_obs_images: List[np.ndarray] = []
    if extra_images is not None:
        render_obs_images.extend(extra_images)

    for sensor_name in observations:
        if "rgb" in sensor_name and "pose" not in sensor_name:
            rgb = to_array(observations[sensor_name])
            render_obs_images.append(rgb)
        elif "depth" in sensor_name and "pose" not in sensor_name:
            depth_map = to_array(observations[sensor_name])  # [H, W, 1]
            # depth map is assumed to be normalized.
            depth_map = (depth_map * 255).astype(np.uint8)
            depth_map = np.repeat(depth_map, 3, axis=-1)
            render_obs_images.append(depth_map)
        elif "semantic" in sensor_name:
            semantic_map = to_array(observations[sensor_name])  # [H, W]
            semantic_map = (semantic_map * 255).astype(np.uint8)
            semantic_map = np.repeat(semantic_map[..., np.newaxis], 3, axis=-1)
            render_obs_images.append(semantic_map)

    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    return render_frame


def add_goal_to_map_2d(
    map_2d: np.ndarray,
    goal: List[float],
    value,
    size: int = 0,
):
    if goal is None:
        return False
    assert len(goal) == 2, goal
    assert len(map_2d.shape) == 2, map_2d.shape
    if all(0 <= x < s for x, s in zip(goal, map_2d.shape)):
        map_2d[
            max(goal[0] - size, 0) : goal[0] + size,
            max(goal[1] - size, 0) : goal[1] + size,
        ] = value
        return True
    else:
        return False
