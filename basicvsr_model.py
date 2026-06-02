from __future__ import annotations

import numpy as np


def rgb_to_ycbcr(image: np.ndarray, only_y: bool = False) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    if only_y:
        y = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
        return (y / 255.0).astype(np.float32)

    ycbcr = np.matmul(
        image,
        [
            [65.481, -37.797, 112.0],
            [128.553, -74.203, -93.786],
            [24.966, 112.0, -18.214],
        ],
    ) + [16.0, 128.0, 128.0]
    return (ycbcr / 255.0).astype(np.float32)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) * 255.0
    rgb = (
        np.matmul(
            image,
            [
                [0.00456621, 0.00456621, 0.00456621],
                [0.0, -0.00153632, 0.00791071],
                [0.00625893, -0.00318811, 0.0],
            ],
        )
        * 255.0
        + [-222.921, 135.576, -276.836]
    )
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def extract_y_channel(image: np.ndarray) -> np.ndarray:
    return (rgb_to_ycbcr(image, only_y=True) * 255.0).astype(np.float32)
