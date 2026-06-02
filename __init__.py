#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vsr_part1.baselines import resize_frame
from vsr_part1.color_utils import extract_y_channel, rgb_to_ycbcr, ycbcr_to_rgb
from vsr_part1.io_utils import ensure_dir, find_sequence_dirs, find_sequence_dirs_from_list, list_images, load_frame, save_frame
from vsr_part1.srcnn import SRCNN, normalize_srcnn_state_dict, srcnn_kwargs_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRCNN inference on all sequences under a dataset root.")
    parser.add_argument("--input-root", required=True, help="Root directory containing LR frame sequences.")
    parser.add_argument("--output-root", required=True, help="Root directory for SRCNN outputs.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained SRCNN checkpoint.")
    parser.add_argument("--scale", type=int, default=4, help="Upsampling scale factor.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sequence-list-file",
        default=None,
        help="Optional relative sequence list file. When provided, only listed sequences are processed.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional limit for smoke tests or partial runs.",
    )
    return parser.parse_args()


def to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    if frame.ndim == 2:
        frame = frame[..., None]
    tensor = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    if image.shape[0] == 1:
        return (image[0] * 255.0).round().astype(np.uint8)
    image = image.transpose(1, 2, 0)
    return (image * 255.0).round().astype(np.uint8)


def resolve_sequences_root(input_root: Path) -> Path:
    sequences_root = input_root / "sequences"
    return sequences_root if sequences_root.is_dir() else input_root


def sequence_output_dir(output_root: Path, sequences_root: Path, sequence_dir: Path) -> Path:
    relative = sequence_dir.relative_to(sequences_root)
    return output_root if relative == Path(".") else output_root / relative


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    device = torch.device(args.device)
    sequences_root = resolve_sequences_root(input_root)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SRCNN(**srcnn_kwargs_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(normalize_srcnn_state_dict(checkpoint["model_state_dict"]))
    model.eval()
    color_space = checkpoint.get("config", {}).get("color_space", "y" if model.in_channels == 1 else "rgb")

    if args.sequence_list_file is not None:
        sequence_dirs = find_sequence_dirs_from_list(input_root, args.sequence_list_file)
    else:
        sequence_dirs = find_sequence_dirs(input_root)
    if args.max_sequences is not None:
        sequence_dirs = sequence_dirs[: args.max_sequences]

    summary = []
    with torch.no_grad():
        for sequence_dir in sequence_dirs:
            sequence_output = ensure_dir(sequence_output_dir(output_root, sequences_root, sequence_dir))
            frame_paths = list_images(sequence_dir)
            for image_path in frame_paths:
                lr_frame = load_frame(image_path)
                bicubic = resize_frame(lr_frame, scale=args.scale, method="bicubic")
                if color_space == "y":
                    bicubic_y = extract_y_channel(bicubic)
                    prediction_y = model(to_tensor(bicubic_y, device))
                    prediction = to_image(prediction_y).astype(np.float32) / 255.0
                    bicubic_ycbcr = rgb_to_ycbcr(bicubic)
                    bicubic_ycbcr[..., 0] = prediction
                    save_frame(ycbcr_to_rgb(bicubic_ycbcr), sequence_output / image_path.name)
                else:
                    prediction = model(to_tensor(bicubic, device))
                    save_frame(to_image(prediction), sequence_output / image_path.name)

            relative = sequence_dir.relative_to(sequences_root)
            summary.append(
                {
                    "sequence": "." if relative == Path(".") else relative.as_posix(),
                    "num_frames": len(frame_paths),
                }
            )

    print(json.dumps({"num_sequences": len(summary), "sequences": summary}, indent=2))


if __name__ == "__main__":
    main()
