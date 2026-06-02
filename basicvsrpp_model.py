from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vsr_part1.color_utils import extract_y_channel
from vsr_part1.io_utils import list_images, list_images_from_sequence_list


RESAMPLE_MAP = {
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        array = array[..., None]
    contiguous = np.array(array.transpose(2, 0, 1), copy=True, order="C")
    return torch.from_numpy(contiguous).float() / 255.0


def _resize(array: np.ndarray, size: tuple[int, int], method: str) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8))
    return np.array(image.resize(size, RESAMPLE_MAP[method]), dtype=np.uint8, copy=True)


class SRCNNPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        hr_dir: str | Path,
        sequence_list_file: str | Path | None = None,
        scale: int = 4,
        patch_size: int = 32,
        samples_per_image: int = 1,
        downsample_method: str = "bicubic",
        color_space: str = "y",
        augment: bool = True,
        recursive: bool = True,
        seed: int = 42,
    ) -> None:
        if patch_size % scale != 0:
            raise ValueError("Patch size must be divisible by the scale factor.")
        if color_space not in {"rgb", "y"}:
            raise ValueError(f"Unsupported color space: {color_space}")

        if sequence_list_file is not None:
            self.image_paths = list_images_from_sequence_list(hr_dir, sequence_list_file)
        else:
            self.image_paths = list_images(hr_dir, recursive=recursive)
        self.scale = scale
        self.patch_size = patch_size
        self.samples_per_image = samples_per_image
        self.downsample_method = downsample_method
        self.color_space = color_space
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index % len(self.image_paths)]
        rng = random.Random(self.seed + index)
        with Image.open(image_path) as image:
            hr = np.asarray(image.convert("RGB"), dtype=np.uint8)

        height, width = hr.shape[:2]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Frame {image_path} is smaller than patch size {self.patch_size}. "
                f"Received {width}x{height}."
            )

        top = rng.randint(0, height - self.patch_size)
        left = rng.randint(0, width - self.patch_size)
        hr_patch = hr[top : top + self.patch_size, left : left + self.patch_size]

        if self.augment:
            if rng.random() < 0.5:
                hr_patch = np.flip(hr_patch, axis=0)
            if rng.random() < 0.5:
                hr_patch = np.flip(hr_patch, axis=1)

        if self.color_space == "y":
            hr_patch = extract_y_channel(hr_patch)

        lr_size = (self.patch_size // self.scale, self.patch_size // self.scale)
        lr_patch = _resize(hr_patch, lr_size, method=self.downsample_method)
        upsampled_patch = _resize(lr_patch, (self.patch_size, self.patch_size), method="bicubic")
        return _to_tensor(upsampled_patch), _to_tensor(hr_patch.copy())
