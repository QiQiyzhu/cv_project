from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter


RESAMPLE_MAP = {
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def resize_frame(frame: np.ndarray, scale: int, method: str) -> np.ndarray:
    if method not in RESAMPLE_MAP:
        raise ValueError(f"Unsupported resize method: {method}")

    image = Image.fromarray(frame.astype(np.uint8))
    width, height = image.size
    resized = image.resize((width * scale, height * scale), RESAMPLE_MAP[method])
    return np.asarray(resized, dtype=np.uint8)


def spatial_baseline_sequence(frames: Iterable[np.ndarray], scale: int, method: str) -> list[np.ndarray]:
    return [resize_frame(frame, scale=scale, method=method) for frame in frames]


def spatial_baseline_center_frame(frames: list[np.ndarray], scale: int, method: str) -> np.ndarray:
    if not frames:
        raise ValueError("At least one frame is required.")
    center_index = len(frames) // 2
    return resize_frame(frames[center_index], scale=scale, method=method)


def temporal_average_sequence(
    frames: list[np.ndarray],
    scale: int,
    radius: int = 1,
    weights: list[float] | None = None,
    sharpen: bool = False,
    sharpen_radius: float = 1.5,
    sharpen_percent: int = 120,
) -> list[np.ndarray]:
    if radius < 0:
        raise ValueError("Temporal radius must be non-negative.")
    if not frames:
        return []

    upscaled = spatial_baseline_sequence(frames, scale=scale, method="bicubic")
    if weights is None:
        weights = [1.0] * (2 * radius + 1)
    if len(weights) != 2 * radius + 1:
        raise ValueError("Temporal weights must match the selected radius.")

    outputs: list[np.ndarray] = []
    num_frames = len(upscaled)
    for center_index in range(num_frames):
        accum: np.ndarray | None = None
        total_weight = 0.0
        for offset, weight in zip(range(-radius, radius + 1), weights):
            frame_index = min(max(center_index + offset, 0), num_frames - 1)
            current = upscaled[frame_index].astype(np.float32)
            accum = current * weight if accum is None else accum + current * weight
            total_weight += weight

        averaged = np.clip(accum / max(total_weight, 1e-8), 0, 255).astype(np.uint8)
        if sharpen:
            image = Image.fromarray(averaged)
            averaged = np.asarray(
                image.filter(
                    ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_percent, threshold=3)
                ),
                dtype=np.uint8,
            )
        outputs.append(averaged)
    return outputs


def temporal_average_center_frame(
    frames: list[np.ndarray],
    scale: int,
    radius: int = 1,
    weights: list[float] | None = None,
    sharpen: bool = False,
    sharpen_radius: float = 1.5,
    sharpen_percent: int = 120,
) -> np.ndarray:
    if radius < 0:
        raise ValueError("Temporal radius must be non-negative.")
    if not frames:
        raise ValueError("At least one frame is required.")

    upscaled = spatial_baseline_sequence(frames, scale=scale, method="bicubic")
    if weights is None:
        weights = [1.0] * (2 * radius + 1)
    if len(weights) != 2 * radius + 1:
        raise ValueError("Temporal weights must match the selected radius.")

    center_index = len(upscaled) // 2
    accum: np.ndarray | None = None
    total_weight = 0.0
    for offset, weight in zip(range(-radius, radius + 1), weights):
        frame_index = min(max(center_index + offset, 0), len(upscaled) - 1)
        current = upscaled[frame_index].astype(np.float32)
        accum = current * weight if accum is None else accum + current * weight
        total_weight += weight

    averaged = np.clip(accum / max(total_weight, 1e-8), 0, 255).astype(np.uint8)
    if not sharpen:
        return averaged

    image = Image.fromarray(averaged)
    return np.asarray(
        image.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_percent, threshold=3)),
        dtype=np.uint8,
    )
