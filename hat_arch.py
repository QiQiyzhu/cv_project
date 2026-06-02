from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_nonempty_lines(path: str | Path) -> list[str]:
    file_path = Path(path)
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(input_dir: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    if recursive:
        images = sorted(path for path in root.rglob("*") if is_image_file(path))
    else:
        images = sorted(path for path in root.iterdir() if is_image_file(path))
    if not images:
        raise FileNotFoundError(f"No image frames were found in: {root}")
    return images


def find_sequence_dirs(input_root: str | Path) -> list[Path]:
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    if any(is_image_file(path) for path in root.iterdir()):
        return [root]

    sequence_dirs = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_dir() and any(is_image_file(child) for child in path.iterdir())
    ]
    if not sequence_dirs:
        raise FileNotFoundError(f"No image sequence directories were found in: {root}")
    return sequence_dirs


def resolve_sequences_root(input_root: str | Path) -> Path:
    root = Path(input_root)
    sequences_root = root / "sequences"
    return sequences_root if sequences_root.is_dir() else root


def find_sequence_dirs_from_list(input_root: str | Path, list_file: str | Path) -> list[Path]:
    sequences_root = resolve_sequences_root(input_root)
    sequence_dirs = [sequences_root / relative for relative in read_nonempty_lines(list_file)]
    missing_dirs = [path for path in sequence_dirs if not path.is_dir()]
    if missing_dirs:
        preview = ", ".join(str(path) for path in missing_dirs[:3])
        raise FileNotFoundError(f"Some listed sequence directories do not exist: {preview}")
    return sequence_dirs


def list_images_from_sequence_list(input_root: str | Path, list_file: str | Path) -> list[Path]:
    image_paths: list[Path] = []
    for sequence_dir in find_sequence_dirs_from_list(input_root, list_file):
        image_paths.extend(list_images(sequence_dir))
    if not image_paths:
        raise FileNotFoundError(f"No image frames were found from sequence list: {list_file}")
    return image_paths


def load_frame(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_frame(frame: np.ndarray, path: str | Path) -> None:
    Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)


def load_sequence(input_dir: str | Path) -> list[np.ndarray]:
    return [load_frame(path) for path in list_images(input_dir)]


def save_sequence(
    frames: Iterable[np.ndarray],
    output_dir: str | Path,
    prefix: str = "frame",
    suffix: str = ".png",
    start_index: int = 0,
) -> None:
    output_root = ensure_dir(output_dir)
    for index, frame in enumerate(frames, start=start_index):
        save_frame(frame, output_root / f"{prefix}_{index:04d}{suffix}")


def parse_weights(raw_weights: str | None, radius: int) -> list[float]:
    if raw_weights is None:
        return [1.0] * (2 * radius + 1)

    weights = [float(value.strip()) for value in raw_weights.split(",") if value.strip()]
    expected = 2 * radius + 1
    if len(weights) != expected:
        raise ValueError(f"Expected {expected} temporal weights for radius={radius}, got {len(weights)}")
    return weights
