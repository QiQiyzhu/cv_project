#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vsr_benchmarks import get_benchmark_or_raise
from vsr_part1.color_utils import extract_y_channel
from vsr_part1.io_utils import find_sequence_dirs, list_images, load_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted dataset outputs against GT folders with PSNR, SSIM, LPIPS, FID, and tLPIPS."
    )
    parser.add_argument("--pred-root", required=True, help="Root directory containing predicted frame sequences.")
    parser.add_argument("--gt-root", default=None, help="Root directory containing GT frame sequences.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional registered GT-ready benchmark name. When set, GT root and default matching config are resolved automatically.",
    )
    parser.add_argument(
        "--workdir",
        default=str(PROJECT_ROOT),
        help="Project root used to resolve registered benchmark datasets. Defaults to the local repo root.",
    )
    parser.add_argument(
        "--sequence-source",
        choices=("pred", "gt", "intersection"),
        default=None,
        help="Which root defines the sequence list. If omitted with --dataset-name, the benchmark default is used.",
    )
    parser.add_argument(
        "--frame-match",
        choices=("intersection", "exact"),
        default=None,
        help="How to match frames inside each sequence. If omitted with --dataset-name, the benchmark default is used.",
    )
    parser.add_argument("--crop-border", type=int, default=0, help="Optional border crop before evaluation.")
    parser.add_argument(
        "--channel",
        choices=("rgb", "y"),
        default="rgb",
        help="Channel used for PSNR/SSIM. LPIPS/FID/tLPIPS remain RGB perceptual metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for Inception feature extraction.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for LPIPS/FID/tLPIPS. Use 'auto', 'cpu', 'cuda', or 'cuda:0'.",
    )
    parser.add_argument("--skip-lpips", action="store_true", help="Skip LPIPS and tLPIPS.")
    parser.add_argument("--skip-fid", action="store_true", help="Skip FID.")
    parser.add_argument(
        "--torch-home",
        default=None,
        help="Optional TORCH_HOME cache directory for pretrained LPIPS/Inception weights.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save the evaluation summary.")
    return parser.parse_args()


def resolve_dataset_config(args: argparse.Namespace) -> tuple[Path, str, str, dict[str, object] | None]:
    benchmark_info = None
    if args.dataset_name:
        benchmark = get_benchmark_or_raise(args.workdir, args.dataset_name)
        gt_root = Path(args.gt_root) if args.gt_root else benchmark.gt_root
        sequence_source = args.sequence_source or benchmark.sequence_source
        frame_match = args.frame_match or benchmark.frame_match
        benchmark_info = benchmark.to_json_dict()
    else:
        if not args.gt_root:
            raise ValueError("Either --gt-root or --dataset-name must be provided.")
        gt_root = Path(args.gt_root)
        sequence_source = args.sequence_source or "pred"
        frame_match = args.frame_match or "intersection"

    return gt_root, sequence_source, frame_match, benchmark_info


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SSIM evaluation requires scikit-image. Install it first, for example with 'pip install scikit-image'."
        ) from exc

    if pred.ndim == 2:
        return float(structural_similarity(target, pred, data_range=255))
    return float(structural_similarity(target, pred, channel_axis=-1, data_range=255))


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def to_lpips_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def to_inception_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)


def compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    if features.shape[0] < 2:
        sigma = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    else:
        sigma = np.cov(features, rowvar=False)
    return mu, sigma


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    sqrt_eigvals = np.sqrt(eigvals)
    return (eigvecs * sqrt_eigvals) @ eigvecs.T


def compute_fid_from_features(pred_features: np.ndarray, gt_features: np.ndarray) -> float:
    mu_pred, sigma_pred = compute_statistics(pred_features)
    mu_gt, sigma_gt = compute_statistics(gt_features)
    return compute_fid_from_statistics(mu_pred, sigma_pred, mu_gt, sigma_gt)


def compute_fid_from_statistics(
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    mu_gt: np.ndarray,
    sigma_gt: np.ndarray,
) -> float:
    mean_diff = mu_pred - mu_gt
    sqrt_sigma_pred = matrix_sqrt_psd(sigma_pred)
    covmean = matrix_sqrt_psd(sqrt_sigma_pred @ sigma_gt @ sqrt_sigma_pred)
    fid = mean_diff @ mean_diff + np.trace(sigma_pred) + np.trace(sigma_gt) - 2.0 * np.trace(covmean)
    return float(np.real(fid))


class RunningFeatureStats:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_outer = np.zeros((dim, dim), dtype=np.float64)

    def update(self, features: np.ndarray) -> None:
        if features.size == 0:
            return
        features64 = features.astype(np.float64, copy=False)
        self.count += features64.shape[0]
        self.sum += features64.sum(axis=0)
        self.sum_outer += features64.T @ features64

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return np.zeros(self.dim, dtype=np.float64), np.zeros((self.dim, self.dim), dtype=np.float64)
        mean = self.sum / self.count
        if self.count < 2:
            covariance = np.zeros((self.dim, self.dim), dtype=np.float64)
        else:
            centered_outer = self.sum_outer - self.count * np.outer(mean, mean)
            covariance = centered_outer / (self.count - 1)
        return mean, covariance


def load_lpips_model(device: torch.device) -> torch.nn.Module:
    try:
        import lpips
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LPIPS/tLPIPS evaluation requires the 'lpips' package. Install it first, for example with 'pip install lpips'."
        ) from exc

    model = lpips.LPIPS(net="alex")
    model.eval()
    model.to(device)
    return model


def load_inception_model(device: torch.device) -> torch.nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def collect_sequence_relatives(pred_root: Path, gt_root: Path, source: str) -> list[Path]:
    pred_relatives = {"." if path == pred_root else path.relative_to(pred_root).as_posix() for path in find_sequence_dirs(pred_root)}
    gt_relatives = {"." if path == gt_root else path.relative_to(gt_root).as_posix() for path in find_sequence_dirs(gt_root)}

    if source == "pred":
        selected = pred_relatives
    elif source == "gt":
        selected = gt_relatives
    elif source == "intersection":
        selected = pred_relatives & gt_relatives
    else:
        raise ValueError(f"Unsupported sequence source: {source}")

    if not selected:
        raise FileNotFoundError(f"No matching sequence directories found with sequence_source={source}.")
    return [Path(".") if item == "." else Path(item) for item in sorted(selected)]


def match_frame_paths(pred_paths: list[Path], gt_paths: list[Path], frame_match: str, pred_dir: Path) -> list[tuple[Path, Path]]:
    if frame_match == "exact":
        if len(pred_paths) != len(gt_paths):
            raise ValueError(
                f"Mismatched frame counts: {len(pred_paths)} predictions vs {len(gt_paths)} GT frames in {pred_dir}."
            )
        pred_names = [path.name for path in pred_paths]
        gt_names = [path.name for path in gt_paths]
        if pred_names != gt_names:
            raise ValueError(f"Mismatched frame names in {pred_dir}: {pred_names[:5]} vs {gt_names[:5]}.")
        return list(zip(pred_paths, gt_paths))

    if frame_match == "intersection":
        pred_by_name = {path.name: path for path in pred_paths}
        gt_by_name = {path.name: path for path in gt_paths}
        common_names = sorted(pred_by_name.keys() & gt_by_name.keys())
        if not common_names:
            raise ValueError(f"No common frame names found in {pred_dir}.")
        return [(pred_by_name[name], gt_by_name[name]) for name in common_names]

    raise ValueError(f"Unsupported frame_match: {frame_match}")


def evaluate_frame_pair_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    crop_border: int,
    channel: str,
) -> tuple[float, float]:
    if crop_border > 0:
        pred = pred[crop_border:-crop_border, crop_border:-crop_border]
        gt = gt[crop_border:-crop_border, crop_border:-crop_border]

    if channel == "y":
        pred_eval = extract_y_channel(pred)
        gt_eval = extract_y_channel(gt)
    else:
        pred_eval = pred
        gt_eval = gt

    return compute_psnr(pred_eval, gt_eval), compute_ssim(pred_eval, gt_eval)


def evaluate_sequence(
    pred_dir: Path,
    gt_dir: Path,
    crop_border: int,
    channel: str,
    lpips_model: torch.nn.Module | None,
    inception_model: torch.nn.Module | None,
    device: torch.device,
    batch_size: int,
    frame_match: str,
) -> tuple[dict, list[torch.Tensor], list[torch.Tensor]]:
    pred_paths = list_images(pred_dir)
    gt_paths = list_images(gt_dir)
    matched_paths = match_frame_paths(pred_paths, gt_paths, frame_match=frame_match, pred_dir=pred_dir)

    frame_metrics: list[dict] = []
    paired_pred_lpips_tensors: list[torch.Tensor] = []
    paired_gt_lpips_tensors: list[torch.Tensor] = []
    temporal_pred_tensors: list[torch.Tensor] = []
    temporal_gt_tensors: list[torch.Tensor] = []
    pred_fid_tensors: list[torch.Tensor] = []
    gt_fid_tensors: list[torch.Tensor] = []

    for pred_path, gt_path in matched_paths:
        pred = load_frame(pred_path)
        gt = load_frame(gt_path)

        psnr, ssim = evaluate_frame_pair_metrics(pred, gt, crop_border=crop_border, channel=channel)
        frame_summary = {
            "frame": pred_path.name,
            "psnr": psnr,
            "ssim": ssim,
        }

        if lpips_model is not None:
            paired_pred_lpips_tensors.append(to_lpips_tensor(pred))
            paired_gt_lpips_tensors.append(to_lpips_tensor(gt))

        if inception_model is not None:
            pred_fid_tensors.append(to_inception_tensor(pred))
            gt_fid_tensors.append(to_inception_tensor(gt))

        frame_metrics.append(frame_summary)

    mean_psnr = sum(item["psnr"] for item in frame_metrics) / len(frame_metrics)
    mean_ssim = sum(item["ssim"] for item in frame_metrics) / len(frame_metrics)

    sequence_summary: dict[str, object] = {
        "num_pred_frames": len(pred_paths),
        "num_gt_frames": len(gt_paths),
        "num_matched_frames": len(frame_metrics),
        "channel": channel,
        "frame_match": frame_match,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "frames": frame_metrics,
    }

    if lpips_model is not None:
        temporal_pred_tensors = [to_lpips_tensor(load_frame(path)) for path in pred_paths]
        temporal_gt_tensors = [to_lpips_tensor(load_frame(path)) for path in gt_paths]
        lpips_scores = compute_lpips_scores(paired_pred_lpips_tensors, paired_gt_lpips_tensors, lpips_model, device)
        pred_tlpips_scores = compute_adjacent_lpips_scores(temporal_pred_tensors, lpips_model, device)
        gt_tlpips_scores = compute_adjacent_lpips_scores(temporal_gt_tensors, lpips_model, device)
        tlpips_gap_scores = [
            abs(pred_score - gt_score) for pred_score, gt_score in zip(pred_tlpips_scores, gt_tlpips_scores)
        ]
        sequence_summary["mean_lpips"] = float(sum(lpips_scores) / len(lpips_scores))
        sequence_summary["mean_tlpips"] = (
            float(sum(pred_tlpips_scores) / len(pred_tlpips_scores)) if pred_tlpips_scores else None
        )
        sequence_summary["mean_gt_tlpips"] = (
            float(sum(gt_tlpips_scores) / len(gt_tlpips_scores)) if gt_tlpips_scores else None
        )
        sequence_summary["mean_tlpips_gap"] = (
            float(sum(tlpips_gap_scores) / len(tlpips_gap_scores)) if tlpips_gap_scores else None
        )
        sequence_summary["tlpips_scores"] = pred_tlpips_scores
        sequence_summary["gt_tlpips_scores"] = gt_tlpips_scores
        sequence_summary["tlpips_gap_scores"] = tlpips_gap_scores
        for frame_summary, score in zip(frame_metrics, lpips_scores):
            frame_summary["lpips"] = score

    return sequence_summary, pred_fid_tensors, gt_fid_tensors


def compute_lpips_scores(
    pred_tensors: list[torch.Tensor],
    gt_tensors: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    with torch.inference_mode():
        for pred_tensor, gt_tensor in zip(pred_tensors, gt_tensors):
            score = model(pred_tensor.unsqueeze(0).to(device), gt_tensor.unsqueeze(0).to(device))
            scores.append(float(score.item()))
    return scores


def compute_adjacent_lpips_scores(
    tensors: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    if len(tensors) < 2:
        return scores

    with torch.inference_mode():
        for idx in range(len(tensors) - 1):
            score = model(tensors[idx].unsqueeze(0).to(device), tensors[idx + 1].unsqueeze(0).to(device))
            scores.append(float(score.item()))
    return scores


def extract_inception_features(
    tensors: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not tensors:
        return np.empty((0, 2048), dtype=np.float32)

    features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            features.append(out.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(features, axis=0)


def main() -> None:
    args = parse_args()
    if args.torch_home:
        os.environ["TORCH_HOME"] = args.torch_home

    device = resolve_device(args.device)

    pred_root = Path(args.pred_root)
    gt_root, sequence_source, frame_match, benchmark_info = resolve_dataset_config(args)

    lpips_model = None
    if not args.skip_lpips:
        lpips_model = load_lpips_model(device)

    inception_model = None
    if not args.skip_fid:
        inception_model = load_inception_model(device)

    sequence_relatives = collect_sequence_relatives(pred_root, gt_root, sequence_source)
    sequence_metrics: list[dict] = []
    total_frames = 0
    finite_psnr_frames = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_weighted_sum = 0.0
    inf_psnr_sequences: list[str] = []
    inf_psnr_frames = 0
    tlpips_scores_all: list[float] = []
    gt_tlpips_scores_all: list[float] = []
    tlpips_gap_scores_all: list[float] = []
    pred_fid_stats = RunningFeatureStats(dim=2048) if inception_model is not None else None
    gt_fid_stats = RunningFeatureStats(dim=2048) if inception_model is not None else None

    for relative in sequence_relatives:
        gt_dir = gt_root if relative == Path(".") else gt_root / relative
        pred_dir = pred_root if relative == Path(".") else pred_root / relative
        (
            metrics,
            pred_fid_tensors,
            gt_fid_tensors,
        ) = evaluate_sequence(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            crop_border=args.crop_border,
            channel=args.channel,
            lpips_model=lpips_model,
            inception_model=inception_model,
            device=device,
            batch_size=args.batch_size,
            frame_match=frame_match,
        )

        num_matched_frames = int(metrics["num_matched_frames"])
        total_frames += num_matched_frames
        mean_psnr = float(metrics["mean_psnr"])
        if math.isfinite(mean_psnr):
            psnr_sum += mean_psnr * num_matched_frames
            finite_psnr_frames += num_matched_frames
        else:
            inf_psnr_frames += num_matched_frames
        ssim_sum += float(metrics["mean_ssim"]) * num_matched_frames

        sequence_summary = {
            "sequence": "." if relative == Path(".") else relative.as_posix(),
            "num_pred_frames": metrics["num_pred_frames"],
            "num_gt_frames": metrics["num_gt_frames"],
            "num_matched_frames": num_matched_frames,
            "mean_psnr": metrics["mean_psnr"],
            "mean_ssim": metrics["mean_ssim"],
        }

        if not math.isfinite(mean_psnr):
            inf_psnr_sequences.append(sequence_summary["sequence"])

        if "mean_lpips" in metrics:
            sequence_summary["mean_lpips"] = metrics["mean_lpips"]
            lpips_weighted_sum += float(metrics["mean_lpips"]) * num_matched_frames
        if "mean_tlpips" in metrics and metrics["mean_tlpips"] is not None:
            sequence_summary["mean_tlpips"] = metrics["mean_tlpips"]
            sequence_summary["mean_gt_tlpips"] = metrics["mean_gt_tlpips"]
            sequence_summary["mean_tlpips_gap"] = metrics["mean_tlpips_gap"]
            tlpips_scores_all.extend(metrics.get("tlpips_scores", []))
            gt_tlpips_scores_all.extend(metrics.get("gt_tlpips_scores", []))
            tlpips_gap_scores_all.extend(metrics.get("tlpips_gap_scores", []))

        sequence_metrics.append(sequence_summary)
        if inception_model is not None and pred_fid_stats is not None and gt_fid_stats is not None:
            pred_features = extract_inception_features(pred_fid_tensors, inception_model, device, args.batch_size)
            gt_features = extract_inception_features(gt_fid_tensors, inception_model, device, args.batch_size)
            pred_fid_stats.update(pred_features)
            gt_fid_stats.update(gt_features)

    summary: dict[str, object] = {
        "num_sequences": len(sequence_metrics),
        "num_matched_frames": total_frames,
        "channel": args.channel,
        "crop_border": args.crop_border,
        "sequence_source": sequence_source,
        "frame_match": frame_match,
        "device": str(device),
        "torch_home": os.environ.get("TORCH_HOME"),
        "pred_root": str(pred_root.resolve()),
        "gt_root": str(gt_root.resolve()),
        "metric_definitions": {
            "psnr": "Pixel-level fidelity against GT; higher is better.",
            "ssim": "Structural similarity against GT; higher is better.",
            "lpips": "Per-frame perceptual distance between prediction and GT; lower is better.",
            "fid": "Dataset-level Inception feature distribution distance between predictions and GT; lower is better.",
            "tlpips": "Adjacent-frame LPIPS within the predicted video; lower indicates smoother output, but natural motion also contributes.",
            "tlpips_gap": "Absolute difference between predicted adjacent-frame LPIPS and GT adjacent-frame LPIPS; lower means temporal change is closer to GT.",
        },
        "mean_psnr": psnr_sum / max(finite_psnr_frames, 1) if finite_psnr_frames else float("inf"),
        "mean_psnr_finite_only": psnr_sum / max(finite_psnr_frames, 1) if finite_psnr_frames else None,
        "num_inf_psnr_sequences": len(inf_psnr_sequences),
        "num_inf_psnr_frames": inf_psnr_frames,
        "inf_psnr_sequences": inf_psnr_sequences,
        "mean_ssim": ssim_sum / max(total_frames, 1),
        "sequences": sequence_metrics,
    }
    if benchmark_info is not None:
        summary["dataset_name"] = args.dataset_name
        summary["benchmark"] = benchmark_info

    if lpips_model is not None:
        summary["mean_lpips"] = lpips_weighted_sum / max(total_frames, 1)
        summary["mean_tlpips"] = sum(tlpips_scores_all) / len(tlpips_scores_all) if tlpips_scores_all else None
        summary["mean_gt_tlpips"] = sum(gt_tlpips_scores_all) / len(gt_tlpips_scores_all) if gt_tlpips_scores_all else None
        summary["mean_tlpips_gap"] = (
            sum(tlpips_gap_scores_all) / len(tlpips_gap_scores_all) if tlpips_gap_scores_all else None
        )

    if inception_model is not None and pred_fid_stats is not None and gt_fid_stats is not None:
        pred_mean, pred_cov = pred_fid_stats.finalize()
        gt_mean, gt_cov = gt_fid_stats.finalize()
        summary["fid"] = compute_fid_from_statistics(pred_mean, pred_cov, gt_mean, gt_cov)

    if args.output_json:
        save_metrics(summary, args.output_json)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
