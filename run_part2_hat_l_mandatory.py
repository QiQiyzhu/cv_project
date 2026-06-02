#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vsr_part1.io_utils import ensure_dir, find_sequence_dirs, list_images, load_frame, save_frame


RESAMPLE_MAP = {
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare wild video frame folders into a benchmark-style dataset with "
            "gt_sequences and lr_sequences."
        )
    )
    parser.add_argument(
        "--frames-root",
        required=True,
        help="Root directory containing extracted wild frame folders such as wild_01, wild_02, ...",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output dataset root. Script will create gt_sequences and lr_sequences inside it.",
    )
    parser.add_argument("--gt-width", type=int, default=448)
    parser.add_argument("--gt-height", type=int, default=256)
    parser.add_argument("--lr-width", type=int, default=112)
    parser.add_argument("--lr-height", type=int, default=64)
    parser.add_argument(
        "--method",
        choices=sorted(RESAMPLE_MAP),
        default="bicubic",
        help="Resampling method used for both GT resize and LR downsampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output frames if they already exist.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional explicit path for the output summary JSON. Defaults to <output-root>/summary.json.",
    )
    parser.add_argument(
        "--sequence-list-file",
        default=None,
        help="Optional explicit path for the output sequence list file. Defaults to <output-root>/sequence_list.txt.",
    )
    return parser.parse_args()


def resize_frame(frame: np.ndarray, width: int, height: int, method: str) -> np.ndarray:
    image = Image.fromarray(frame)
    resized = image.resize((width, height), RESAMPLE_MAP[method]).convert("RGB")
    return np.asarray(resized, dtype=np.uint8)


def canonical_sequence_name(sequence_dir: Path) -> str:
    matches = re.findall(r"\d+", sequence_dir.name)
    if matches:
        return str(int(matches[-1]))
    return sequence_dir.name


def frame_output_name(index: int) -> str:
    return f"{index:08d}.png"


def main() -> None:
    args = parse_args()
    if args.gt_width != args.lr_width * 4 or args.gt_height != args.lr_height * 4:
        raise ValueError("Expected GT size to be exactly 4x LR size for this x4 benchmark preparation pipeline.")

    frames_root = Path(args.frames_root)
    output_root = ensure_dir(args.output_root)
    gt_root = ensure_dir(output_root / "gt_sequences")
    lr_root = ensure_dir(output_root / "lr_sequences")

    sequence_dirs = find_sequence_dirs(frames_root)
    mappings: list[dict[str, object]] = []
    total_frames = 0

    for sequence_dir in sequence_dirs:
        sequence_name = canonical_sequence_name(sequence_dir)
        gt_dir = ensure_dir(gt_root / sequence_name)
        lr_dir = ensure_dir(lr_root / sequence_name)
        frame_paths = list_images(sequence_dir)

        if not args.overwrite:
            gt_existing = sorted(path.name for path in gt_dir.glob("*.png"))
            lr_existing = sorted(path.name for path in lr_dir.glob("*.png"))
            expected = [frame_output_name(index) for index in range(len(frame_paths))]
            if gt_existing == expected and lr_existing == expected:
                mappings.append(
                    {
                        "source": sequence_dir.name,
                        "target": sequence_name,
                        "num_frames": len(frame_paths),
                        "gt_dir": str(gt_dir),
                        "lr_dir": str(lr_dir),
                        "skipped_existing": True,
                    }
                )
                total_frames += len(frame_paths)
                continue

        for index, frame_path in enumerate(frame_paths):
            source = load_frame(frame_path)
            gt_frame = resize_frame(source, width=args.gt_width, height=args.gt_height, method=args.method)
            lr_frame = resize_frame(gt_frame, width=args.lr_width, height=args.lr_height, method=args.method)
            output_name = frame_output_name(index)
            save_frame(gt_frame, gt_dir / output_name)
            save_frame(lr_frame, lr_dir / output_name)

        mappings.append(
            {
                "source": sequence_dir.name,
                "target": sequence_name,
                "num_frames": len(frame_paths),
                "gt_dir": str(gt_dir),
                "lr_dir": str(lr_dir),
                "skipped_existing": False,
            }
        )
        total_frames += len(frame_paths)

    mappings.sort(key=lambda item: int(str(item["target"])) if str(item["target"]).isdigit() else str(item["target"]))
    sequence_names = [str(item["target"]) for item in mappings]

    summary = {
        "frames_root": str(frames_root),
        "output_root": str(output_root),
        "gt_root": str(gt_root),
        "lr_root": str(lr_root),
        "gt_size": [args.gt_width, args.gt_height],
        "lr_size": [args.lr_width, args.lr_height],
        "scale": 4,
        "method": args.method,
        "num_sequences": len(mappings),
        "num_frames": total_frames,
        "sequences": mappings,
    }

    summary_path = Path(args.summary_json) if args.summary_json else output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sequence_list_path = Path(args.sequence_list_file) if args.sequence_list_file else output_root / "sequence_list.txt"
    sequence_list_path.write_text("\n".join(sequence_names) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
