#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from vsr_part1.io_utils import ensure_dir, list_images
from vsr_part2 import run_basicvsr_dataset_inference, save_summary
from vsr_part2.basicvsr_model import BasicVSRInferencer


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
VALID_DATASETS = {"reds_sample", "vimeo_lr", "wild_video"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BasicVSR over all mandatory datasets.")
    parser.add_argument(
        "--datasets",
        default="reds_sample,vimeo_lr,wild_video",
        help="Comma-separated subset of datasets to run: reds_sample,vimeo_lr,wild_video.",
    )
    parser.add_argument(
        "--reds-root",
        default=str(PROJECT_ROOT / "data" / "official" / "reds_sample" / "lr_sequences"),
        help="Root directory of the mandatory REDS sample frames.",
    )
    parser.add_argument(
        "--vimeo-root",
        default=str(PROJECT_ROOT / "data" / "official" / "vimeo_lr" / "lr_sequences"),
        help="Root directory of the mandatory Vimeo LR sequences.",
    )
    parser.add_argument(
        "--wild-root",
        default=str(PROJECT_ROOT / "wild_video"),
        help="Root directory of the mandatory wild videos.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "part2" / "basicvsr_mandatory"),
        help="Output root for the mandatory BasicVSR run.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional BasicVSR checkpoint path or URL. If omitted, the official default checkpoint is downloaded automatically.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=100,
        help="Maximum sequence chunk length for recurrent inference.",
    )
    parser.add_argument(
        "--wild-target-width",
        type=int,
        default=None,
        help="Optional width to resize every wild_video frame before inference.",
    )
    parser.add_argument(
        "--wild-target-height",
        type=int,
        default=None,
        help="Optional height to resize every wild_video frame before inference.",
    )
    parser.add_argument(
        "--wild-prepare-only",
        action="store_true",
        help="Only extract and resize wild_video inputs, without running BasicVSR inference.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets or videos whose expected outputs already exist.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional combined summary json path. Defaults to <output-root>/mandatory_summary.json.",
    )
    return parser.parse_args()


def _list_videos(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def _parse_dataset_names(raw_value: str) -> set[str]:
    datasets = {value.strip() for value in raw_value.split(",") if value.strip()}
    unknown = datasets - VALID_DATASETS
    if unknown:
        unknown_str = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown datasets: {unknown_str}")
    if not datasets:
        raise ValueError("At least one dataset must be selected.")
    return datasets


def _read_first_frame_size(frames_dir: Path) -> tuple[int, int] | None:
    if not frames_dir.is_dir():
        return None
    try:
        first_frame = list_images(frames_dir)[0]
    except Exception:
        return None
    image = cv2.imread(str(first_frame))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def extract_video_to_frames(
    video_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    target_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    existing_frames = list_images(output_dir) if output_dir.exists() and any(output_dir.iterdir()) else []
    existing_size = _read_first_frame_size(output_dir)
    if existing_frames and not overwrite and (target_size is None or existing_size == target_size):
        return {
            "fps": None,
            "frame_count": len(existing_frames),
            "source_video": str(video_path),
            "frames_dir": str(output_dir),
            "reused_existing_frames": True,
            "source_frame_size": None,
            "prepared_frame_size": list(existing_size) if existing_size is not None else None,
        }

    for image_path in existing_frames:
        image_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_dir / f"{frame_count:08d}.png"), frame)
        frame_count += 1
    capture.release()

    prepared_size = target_size if target_size is not None else (source_width, source_height)
    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "source_video": str(video_path),
        "frames_dir": str(output_dir),
        "reused_existing_frames": False,
        "source_frame_size": [source_width, source_height],
        "prepared_frame_size": [prepared_size[0], prepared_size[1]],
    }


def encode_frames_to_video(frames_dir: Path, output_path: Path, fps: float) -> dict[str, object]:
    frame_paths = list_images(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found to encode under: {frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")
        writer.write(frame)
    writer.release()

    return {
        "output_video": str(output_path),
        "num_frames": len(frame_paths),
        "fps": fps,
        "frame_size": [width, height],
    }


def run_wild_video_inference(
    input_root: Path,
    output_root: Path,
    checkpoint: str,
    device: str,
    max_seq_len: int,
    skip_existing: bool,
    target_size: tuple[int, int] | None = None,
    prepare_only: bool = False,
) -> dict[str, object]:
    inferencer = None if prepare_only else BasicVSRInferencer(device=device, checkpoint=checkpoint)
    input_frames_root = ensure_dir(output_root / "input_frames")
    prepared_videos_root = ensure_dir(output_root / "prepared_videos")
    sr_frames_root = ensure_dir(output_root / "sr_frames")
    sr_videos_root = ensure_dir(output_root / "sr_videos")

    video_summaries: list[dict[str, object]] = []
    for video_path in _list_videos(input_root):
        stem = video_path.stem
        extracted_frames_dir = input_frames_root / stem
        prepared_video_path = prepared_videos_root / f"{stem}.mp4"
        sr_frames_dir = sr_frames_root / stem
        sr_video_path = sr_videos_root / f"{stem}.mp4"

        if prepare_only and skip_existing and prepared_video_path.is_file() and extracted_frames_dir.is_dir():
            num_frames = len(list_images(extracted_frames_dir))
            video_summaries.append(
                {
                    "source_video": str(video_path),
                    "input_frames_dir": str(extracted_frames_dir),
                    "prepared_video": str(prepared_video_path),
                    "num_frames": num_frames,
                    "skipped_existing_output": True,
                    "prepare_only": True,
                }
            )
            continue

        if not prepare_only and skip_existing and sr_video_path.is_file() and sr_frames_dir.is_dir():
            num_frames = len(list_images(sr_frames_dir))
            video_summaries.append(
                {
                    "source_video": str(video_path),
                    "input_frames_dir": str(extracted_frames_dir),
                    "prepared_video": str(prepared_video_path),
                    "output_frames_dir": str(sr_frames_dir),
                    "output_video": str(sr_video_path),
                    "num_frames": num_frames,
                    "skipped_existing_output": True,
                    "prepare_only": False,
                }
            )
            continue

        extraction = extract_video_to_frames(
            video_path,
            extracted_frames_dir,
            overwrite=not skip_existing,
            target_size=target_size,
        )
        fps = extraction["fps"] if extraction["fps"] is not None else 25.0
        prepared_video = encode_frames_to_video(extracted_frames_dir, prepared_video_path, float(fps))
        prepared_video_summary = {
            "prepared_video": str(prepared_video_path),
            "prepared_video_num_frames": prepared_video["num_frames"],
            "prepared_video_fps": prepared_video["fps"],
            "prepared_video_frame_size": prepared_video["frame_size"],
        }
        if prepare_only:
            video_summaries.append(
                {
                    "source_video": str(video_path),
                    "input_frames_dir": str(extracted_frames_dir),
                    **extraction,
                    **prepared_video_summary,
                    "prepare_only": True,
                }
            )
            continue

        if inferencer is None:
            raise RuntimeError("Inferencer was not initialized for wild_video inference.")
        inferencer.infer(
            video=extracted_frames_dir,
            result_out_dir=sr_frames_dir,
            start_idx=0,
            filename_tmpl="{:08d}.png",
            max_seq_len=max_seq_len,
            window_size=0,
        )
        encoded = encode_frames_to_video(sr_frames_dir, sr_video_path, float(fps))
        video_summaries.append(
            {
                "source_video": str(video_path),
                "input_frames_dir": str(extracted_frames_dir),
                "output_frames_dir": str(sr_frames_dir),
                **extraction,
                **prepared_video_summary,
                **encoded,
                "prepare_only": False,
            }
        )

    return {
        "dataset": "wild_video",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "target_size": list(target_size) if target_size is not None else None,
        "prepare_only": prepare_only,
        "num_videos": len(video_summaries),
        "videos": video_summaries,
    }


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    summary_path = Path(args.summary_json) if args.summary_json else output_root / "mandatory_summary.json"
    selected_datasets = _parse_dataset_names(args.datasets)
    wild_target_size: tuple[int, int] | None = None
    if args.wild_target_width is not None or args.wild_target_height is not None:
        if args.wild_target_width is None or args.wild_target_height is None:
            raise ValueError("Both --wild-target-width and --wild-target-height must be provided together.")
        wild_target_size = (args.wild_target_width, args.wild_target_height)

    combined_summary: dict[str, object] = {
        "model_name": "basicvsr",
        "device": args.device,
        "checkpoint": args.checkpoint,
        "output_root": str(output_root),
        "generated_at_unix": time.time(),
        "selected_datasets": sorted(selected_datasets),
        "wild_target_size": list(wild_target_size) if wild_target_size is not None else None,
        "datasets": {},
    }

    dataset_jobs = [
        (
            "reds_sample",
            Path(args.reds_root),
            output_root / "reds_sample",
            output_root / "reds_sample_summary.json",
        ),
        (
            "vimeo_lr",
            Path(args.vimeo_root),
            output_root / "vimeo_lr",
            output_root / "vimeo_lr_summary.json",
        ),
    ]

    for dataset_name, input_root, dataset_output_root, dataset_summary_path in dataset_jobs:
        if dataset_name not in selected_datasets:
            continue
        if args.skip_existing and dataset_summary_path.is_file():
            combined_summary["datasets"][dataset_name] = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
            continue

        summary = run_basicvsr_dataset_inference(
            input_root=input_root,
            output_root=dataset_output_root,
            model_name="basicvsr",
            device=args.device,
            model_ckpt=args.checkpoint,
            max_seq_len=args.max_seq_len,
            window_size=0,
        )
        save_summary(summary, dataset_summary_path)
        combined_summary["datasets"][dataset_name] = summary

    wild_summary_path = output_root / "wild_video_summary.json"
    if "wild_video" in selected_datasets:
        if args.skip_existing and wild_summary_path.is_file():
            combined_summary["datasets"]["wild_video"] = json.loads(wild_summary_path.read_text(encoding="utf-8"))
        else:
            wild_summary = run_wild_video_inference(
                input_root=Path(args.wild_root),
                output_root=output_root / "wild_video",
                checkpoint=args.checkpoint,
                device=args.device,
                max_seq_len=args.max_seq_len,
                skip_existing=args.skip_existing,
                target_size=wild_target_size,
                prepare_only=args.wild_prepare_only,
            )
            save_summary(wild_summary, wild_summary_path)
            combined_summary["datasets"]["wild_video"] = wild_summary

    save_summary(combined_summary, summary_path)
    print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
