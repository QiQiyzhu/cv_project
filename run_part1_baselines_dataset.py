#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vsr_part1.io_utils import find_sequence_dirs
from vsr_part1.metrics import evaluate_frame_directories, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted dataset outputs against GT sequence folders.")
    parser.add_argument("--pred-root", required=True, help="Root directory containing predicted frame sequences.")
    parser.add_argument("--gt-root", required=True, help="Root directory containing GT frame sequences.")
    parser.add_argument("--crop-border", type=int, default=0, help="Optional border crop before evaluation.")
    parser.add_argument("--channel", choices=("rgb", "y"), default="rgb", help="Evaluate on RGB or Y channel.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the evaluation summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)

    sequence_dirs = find_sequence_dirs(gt_root)
    sequence_metrics = []
    total_frames = 0
    psnr_sum = 0.0
    ssim_sum = 0.0

    for gt_dir in sequence_dirs:
        relative = gt_dir.relative_to(gt_root)
        pred_dir = pred_root if relative == Path(".") else pred_root / relative
        metrics = evaluate_frame_directories(pred_dir, gt_dir, crop_border=args.crop_border, channel=args.channel)
        sequence_metrics.append(
            {
                "sequence": "." if relative == Path(".") else relative.as_posix(),
                "num_frames": metrics["num_frames"],
                "mean_psnr": metrics["mean_psnr"],
                "mean_ssim": metrics["mean_ssim"],
            }
        )
        total_frames += metrics["num_frames"]
        psnr_sum += metrics["mean_psnr"] * metrics["num_frames"]
        ssim_sum += metrics["mean_ssim"] * metrics["num_frames"]

    summary = {
        "num_sequences": len(sequence_metrics),
        "num_frames": total_frames,
        "channel": args.channel,
        "mean_psnr": psnr_sum / max(total_frames, 1),
        "mean_ssim": ssim_sum / max(total_frames, 1),
        "sequences": sequence_metrics,
    }

    if args.output_json:
        save_metrics(summary, args.output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
