#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vsr_part1.baselines import spatial_baseline_sequence, temporal_average_sequence
from vsr_part1.io_utils import find_sequence_dirs, load_sequence, parse_weights, save_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 1 baselines on all sequences under a dataset root.")
    parser.add_argument("--input-root", required=True, help="Root directory containing one or more LR sequences.")
    parser.add_argument("--output-root", required=True, help="Root directory for all output sequences.")
    parser.add_argument("--scale", type=int, default=4, help="Upsampling scale factor.")
    parser.add_argument("--temporal-radius", type=int, default=1, help="Radius for temporal averaging.")
    parser.add_argument(
        "--temporal-weights",
        default=None,
        help="Comma-separated weights. Example for radius 1: 1,2,1",
    )
    parser.add_argument(
        "--save-sharpened",
        action="store_true",
        help="Also save an unsharp-mask version of temporal averaging.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional limit for smoke tests or partial runs.",
    )
    return parser.parse_args()


def sequence_output_dir(output_root: Path, input_root: Path, sequence_dir: Path, method: str) -> Path:
    relative = sequence_dir.relative_to(input_root)
    method_root = output_root / method
    return method_root if relative == Path(".") else method_root / relative


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    temporal_weights = parse_weights(args.temporal_weights, radius=args.temporal_radius)

    sequence_dirs = find_sequence_dirs(input_root)
    if args.max_sequences is not None:
        sequence_dirs = sequence_dirs[: args.max_sequences]

    summary = []
    for sequence_dir in sequence_dirs:
        frames = load_sequence(sequence_dir)

        save_sequence(
            spatial_baseline_sequence(frames, scale=args.scale, method="bicubic"),
            sequence_output_dir(output_root, input_root, sequence_dir, "bicubic"),
        )
        save_sequence(
            spatial_baseline_sequence(frames, scale=args.scale, method="lanczos"),
            sequence_output_dir(output_root, input_root, sequence_dir, "lanczos"),
        )
        save_sequence(
            temporal_average_sequence(
                frames,
                scale=args.scale,
                radius=args.temporal_radius,
                weights=temporal_weights,
                sharpen=False,
            ),
            sequence_output_dir(output_root, input_root, sequence_dir, "temporal_average"),
        )

        if args.save_sharpened:
            save_sequence(
                temporal_average_sequence(
                    frames,
                    scale=args.scale,
                    radius=args.temporal_radius,
                    weights=temporal_weights,
                    sharpen=True,
                ),
                sequence_output_dir(output_root, input_root, sequence_dir, "temporal_average_unsharp"),
            )

        relative = sequence_dir.relative_to(input_root)
        summary.append(
            {
                "sequence": "." if relative == Path(".") else relative.as_posix(),
                "num_frames": len(frames),
            }
        )

    print(json.dumps({"num_sequences": len(summary), "sequences": summary}, indent=2))


if __name__ == "__main__":
    main()
