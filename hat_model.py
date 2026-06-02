from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from vsr_part1.color_utils import extract_y_channel
from vsr_part1.io_utils import list_images, load_frame


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=255))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.ndim == 2:
        return float(structural_similarity(target, pred, data_range=255))
    return float(structural_similarity(target, pred, channel_axis=-1, data_range=255))


def evaluate_frame_directories(
    pred_dir: str | Path,
    gt_dir: str | Path,
    crop_border: int = 0,
    channel: str = "rgb",
) -> dict:
    if channel not in {"rgb", "y"}:
        raise ValueError(f"Unsupported evaluation channel: {channel}")

    pred_paths = list_images(pred_dir)
    gt_paths = list_images(gt_dir)
    if len(pred_paths) != len(gt_paths):
        raise ValueError(f"Mismatched frame counts: {len(pred_paths)} predictions vs {len(gt_paths)} GT frames.")

    frame_metrics = []
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred = load_frame(pred_path)
        gt = load_frame(gt_path)

        if crop_border > 0:
            pred = pred[crop_border:-crop_border, crop_border:-crop_border]
            gt = gt[crop_border:-crop_border, crop_border:-crop_border]

        if channel == "y":
            pred = extract_y_channel(pred)
            gt = extract_y_channel(gt)

        frame_metrics.append(
            {
                "frame": pred_path.name,
                "psnr": compute_psnr(pred, gt),
                "ssim": compute_ssim(pred, gt),
            }
        )

    mean_psnr = sum(item["psnr"] for item in frame_metrics) / len(frame_metrics)
    mean_ssim = sum(item["ssim"] for item in frame_metrics) / len(frame_metrics)

    return {
        "num_frames": len(frame_metrics),
        "channel": channel,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "frames": frame_metrics,
    }


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
