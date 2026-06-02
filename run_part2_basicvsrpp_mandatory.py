#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vsr_part1.io_utils import (
    ensure_dir,
    find_sequence_dirs_from_list,
    list_images,
    load_frame,
    save_frame,
)


RESAMPLE_MAP = {
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LR frame sequences from an HR dataset sequence list.")
    parser.add_argument("--input-root", required=True, help="HR dataset root that contains a sequences/ directory.")
    parser.add_argument("--sequence-list-file", required=True, help="Sequence list file with relative paths.")
    parser.add_argument("--output-root", required=True, help="Output root for generated LR sequences.")
    parser.add_argument("--scale", type=int, default=4, help="Downsampling scale factor.")
    parser.add_argument("--method", choices=sorted(RESAMPLE_MAP), default="bicubic")
    parser.add_argument("--summary-json", default=None, help="Optional JSON path for generation summary.")
    return parser.parse_args()


def downsample_frame(frame, scale: int, method: str):
    image = Image.fromarray(frame)
    width, height = image.size
    lr_size = (max(1, width // scale), max(1, height // scale))
    return np.asarray(image.resize(lr_size, RESAMPLE_MAP[method]).convert("RGB"), dtype=np.uint8)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = ensure_dir(args.output_root)
    sequence_dirs = find_sequence_dirs_from_list(input_root, args.sequence_list_file)

    summaries: list[dict[str, object]] = []
    total_frames = 0
    for sequence_dir in sequence_dirs:
        relative = sequence_dir.relative_to(input_root / "sequences")
        save_dir = ensure_dir(output_root / relative)
        frame_paths = list_images(sequence_dir)
        for frame_path in frame_paths:
            lr_image = downsample_frame(load_frame(frame_path), scale=args.scale, method=args.method)
            save_frame(lr_image, save_dir / frame_path.name)
        summaries.append({"sequence": relative.as_posix(), "num_frames": len(frame_paths), "output_dir": str(save_dir)})
        total_frames += len(frame_paths)

    result = {
        "input_root": str(input_root),
        "sequence_list_file": str(args.sequence_list_file),
        "output_root": str(output_root),
        "scale": args.scale,
        "method": args.method,
        "num_sequences": len(summaries),
        "num_frames": total_frames,
        "sequences": summaries,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
