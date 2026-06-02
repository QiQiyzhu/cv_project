#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vsr_part3 import export_dataset_videos, run_fusion_dataset_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 3 paper fusion over all mandatory datasets.")
    parser.add_argument("--base-root", required=True, help="Part 2 BasicVSR++ mandatory output root.")
    parser.add_argument("--gen-root", required=True, help="Part 2 detail-branch mandatory output root, typically HAT-L.")
    parser.add_argument("--output-root", required=True, help="Part 3 output root.")
    parser.add_argument("--checkpoint", required=True, help="Fusion checkpoint path, usually best.pt.")
    parser.add_argument("--device", default="cuda", help="Device to use, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--save-alpha", action="store_true", help="Save alpha maps.")
    parser.add_argument("--skip-videos", action="store_true", help="Skip MP4 export.")
    parser.add_argument("--summary-json", default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root)
    gen_root = Path(args.gen_root)
    output_root = Path(args.output_root)

    datasets = [
        ("vimeo_lr", base_root / "vimeo_lr", gen_root / "vimeo_lr", output_root / "vimeo_lr", 7.0),
        ("reds_sample", base_root / "reds_sample", gen_root / "reds_sample", output_root / "reds_sample", 30.0),
        (
            "wild_video",
            base_root / "wild_video" / "sr_frames",
            gen_root / "wild_video" / "sr_frames",
            output_root / "wild_video" / "sr_frames",
            30.0,
        ),
    ]

    summaries: list[dict] = []
    video_counts: dict[str, int] = {}
    for name, base_dir, gen_dir, out_dir, fps in datasets:
        alpha_root = output_root / "alphas" / name if args.save_alpha else None
        summary = run_fusion_dataset_inference(
            base_root=base_dir,
            gen_root=gen_dir,
            output_root=out_dir,
            checkpoint_path=args.checkpoint,
            device=args.device,
            save_alpha=args.save_alpha,
            alpha_root=alpha_root,
        )
        summary["dataset"] = name
        summaries.append(summary)
        if not args.skip_videos:
            video_root = output_root / f"{name}_videos"
            video_counts[name] = export_dataset_videos(out_dir, video_root, fps=fps)

    result = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "datasets": summaries,
        "video_counts": video_counts,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
