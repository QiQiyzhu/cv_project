#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vsr_part3 import FusionSequenceDataset, MotionCompensatedFusionNet, warp_sequence_with_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Part 3 Motion-Compensated Gated Residual Fusion model."
    )
    parser.add_argument("--train-gt-root", required=True, help="GT root for training sequences.")
    parser.add_argument("--train-base-root", required=True, help="Video SR base root, ideally BasicVSR++ outputs.")
    parser.add_argument("--train-gen-root", required=True, help="Detail branch root, typically HAT-L outputs.")
    parser.add_argument("--train-list-file", default=None, help="Optional sequence list for training.")
    parser.add_argument("--val-gt-root", default=None, help="Optional GT root for validation.")
    parser.add_argument("--val-base-root", default=None, help="Optional base root for validation.")
    parser.add_argument("--val-gen-root", default=None, help="Optional generative root for validation.")
    parser.add_argument("--val-list-file", default=None, help="Optional validation sequence list.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--crop-size", type=int, default=128, help="Random crop size for training clips.")
    parser.add_argument(
        "--val-crop-size",
        type=int,
        default=None,
        help="Optional validation crop size. Defaults to full-frame validation when omitted.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--flow-hidden-channels", type=int, default=32)
    parser.add_argument("--max-flow", type=float, default=24.0)
    parser.add_argument("--residual-scale", type=float, default=0.01)
    parser.add_argument("--alpha-bias", type=float, default=-5.0)
    parser.add_argument("--blend-power", type=float, default=2.0)
    parser.add_argument(
        "--fusion-mode",
        choices=("full_blend", "detail_residual"),
        default="detail_residual",
        help="Conservative detail_residual is recommended for better benchmark stability.",
    )
    parser.set_defaults(
        use_dual_reference_features=True,
        gated_residual=True,
        use_detail_delta_features=True,
    )
    dual_reference_group = parser.add_mutually_exclusive_group()
    dual_reference_group.add_argument("--use-dual-reference-features", dest="use_dual_reference_features", action="store_true")
    dual_reference_group.add_argument(
        "--disable-dual-reference-features",
        dest="use_dual_reference_features",
        action="store_false",
    )
    gated_residual_group = parser.add_mutually_exclusive_group()
    gated_residual_group.add_argument("--gated-residual", dest="gated_residual", action="store_true")
    gated_residual_group.add_argument("--disable-gated-residual", dest="gated_residual", action="store_false")
    detail_delta_group = parser.add_mutually_exclusive_group()
    detail_delta_group.add_argument(
        "--use-detail-delta-features",
        dest="use_detail_delta_features",
        action="store_true",
    )
    detail_delta_group.add_argument(
        "--disable-detail-delta-features",
        dest="use_detail_delta_features",
        action="store_false",
    )
    parser.add_argument("--lambda-temporal-warp", type=float, default=0.2)
    parser.add_argument("--lambda-alpha-temporal", type=float, default=0.05)
    parser.add_argument("--lambda-flow-smooth", type=float, default=0.01)
    parser.add_argument("--lambda-gradient", type=float, default=0.02)
    parser.add_argument("--lambda-alpha-sparse", type=float, default=0.01)
    parser.add_argument("--lambda-base-preserve", type=float, default=2.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--oracle-margin", type=float, default=0.0)
    parser.add_argument("--mask-focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--selection-metric",
        choices=("loss", "psnr"),
        default="psnr",
        help="Metric used to choose best.pt.",
    )
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_loader(
    gt_root: str,
    base_root: str,
    gen_root: str,
    sequence_list_file: str | None,
    crop_size: int | None,
    augment: bool,
    batch_size: int,
    num_workers: int,
    max_sequences: int | None,
    seed: int,
) -> DataLoader:
    dataset = FusionSequenceDataset(
        gt_root=gt_root,
        base_root=base_root,
        gen_root=gen_root,
        sequence_list_file=sequence_list_file,
        crop_size=crop_size,
        augment=augment,
        max_sequences=max_sequences,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse_per_sample = (pred - target).square().flatten(1).mean(dim=1)
    psnr_per_sample = 10.0 * torch.log10(1.0 / mse_per_sample.clamp_min(1e-12))
    return float(psnr_per_sample.mean().item())


def gradient_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def oracle_mask(base: torch.Tensor, gen: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    base_err = (base - target).square().mean(dim=2, keepdim=True)
    gen_err = (gen - target).square().mean(dim=2, keepdim=True)
    return (gen_err + margin < base_err).float()


def weighted_mask_loss(alpha_logits: torch.Tensor, mask_target: torch.Tensor, gamma: float) -> torch.Tensor:
    positive = mask_target.mean().clamp(min=1e-6, max=1.0 - 1e-6)
    pos_weight = ((1.0 - positive) / positive).detach()
    bce = F.binary_cross_entropy_with_logits(alpha_logits, mask_target, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(alpha_logits)
    pt = torch.where(mask_target > 0.5, probs, 1.0 - probs)
    focal = (1.0 - pt).clamp(min=1e-6).pow(gamma)
    return (bce * focal).mean()


def base_preservation_loss(fused: torch.Tensor, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fused_err = (fused - target).abs().mean(dim=2, keepdim=True)
    base_err = (base - target).abs().mean(dim=2, keepdim=True)
    return F.relu(fused_err - base_err).mean()


def flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    flow_dx = flow[..., :, 1:] - flow[..., :, :-1]
    flow_dy = flow[..., 1:, :] - flow[..., :-1, :]
    return flow_dx.abs().mean() + flow_dy.abs().mean()


def motion_temporal_loss(pred: torch.Tensor, target: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if pred.size(1) <= 1:
        return pred.new_tensor(0.0)
    warped_pred_prev = warp_sequence_with_flow(pred[:, :-1], flow[:, 1:])
    warped_gt_prev = warp_sequence_with_flow(target[:, :-1], flow[:, 1:])
    pred_delta = pred[:, 1:] - warped_pred_prev
    target_delta = target[:, 1:] - warped_gt_prev
    return F.l1_loss(pred_delta, target_delta)


def alpha_temporal_loss(alpha_prob: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if alpha_prob.size(1) <= 1:
        return alpha_prob.new_tensor(0.0)
    warped_prev_alpha = warp_sequence_with_flow(alpha_prob[:, :-1], flow[:, 1:])
    return F.l1_loss(alpha_prob[:, 1:], warped_prev_alpha)


def compute_losses(
    outputs: dict[str, torch.Tensor],
    base: torch.Tensor,
    gen: torch.Tensor,
    gt: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    fused = outputs["fused"]
    alpha = outputs["alpha"]
    alpha_prob = outputs["alpha_prob"]
    alpha_logits = outputs["alpha_logits"]
    flow = outputs["flow"]
    mask_target = oracle_mask(base, gen, gt, margin=args.oracle_margin)

    recon_loss = F.l1_loss(fused, gt)
    temporal_loss = motion_temporal_loss(fused, gt, flow)
    alpha_time_loss = alpha_temporal_loss(alpha_prob, flow)
    grad_loss = gradient_consistency_loss(fused, gt)
    alpha_sparse_loss = alpha.mean()
    preserve_loss = base_preservation_loss(fused, base, gt)
    mask_loss = weighted_mask_loss(alpha_logits, mask_target, gamma=args.mask_focal_gamma)
    smooth_loss = flow_smoothness_loss(flow[:, 1:]) if flow.size(1) > 1 else flow.new_tensor(0.0)

    total_loss = recon_loss
    total_loss = total_loss + args.lambda_temporal_warp * temporal_loss
    total_loss = total_loss + args.lambda_alpha_temporal * alpha_time_loss
    total_loss = total_loss + args.lambda_gradient * grad_loss
    total_loss = total_loss + args.lambda_alpha_sparse * alpha_sparse_loss
    total_loss = total_loss + args.lambda_base_preserve * preserve_loss
    total_loss = total_loss + args.lambda_mask * mask_loss
    total_loss = total_loss + args.lambda_flow_smooth * smooth_loss

    return {
        "loss": total_loss,
        "recon": recon_loss,
        "temporal": temporal_loss,
        "alpha_temporal": alpha_time_loss,
        "gradient": grad_loss,
        "alpha_sparse": alpha_sparse_loss,
        "preserve": preserve_loss,
        "mask": mask_loss,
        "flow_smooth": smooth_loss,
    }


def evaluate(
    model: MotionCompensatedFusionNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    total_items = 0
    sums = {
        "loss": 0.0,
        "psnr": 0.0,
        "base_psnr": 0.0,
        "recon": 0.0,
        "temporal": 0.0,
        "alpha_temporal": 0.0,
        "gradient": 0.0,
        "alpha_sparse": 0.0,
        "preserve": 0.0,
        "mask": 0.0,
        "flow_smooth": 0.0,
    }
    with torch.no_grad():
        for batch in loader:
            base = batch["base"].to(device, non_blocking=True)
            gen = batch["gen"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            outputs = model(base, gen)
            losses = compute_losses(outputs, base, gen, gt, args)

            batch_size = base.size(0)
            total_items += batch_size
            sums["loss"] += losses["loss"].item() * batch_size
            for key in ("recon", "temporal", "alpha_temporal", "gradient", "alpha_sparse", "preserve", "mask", "flow_smooth"):
                sums[key] += losses[key].item() * batch_size
            sums["psnr"] += compute_psnr(outputs["fused"], gt) * batch_size
            sums["base_psnr"] += compute_psnr(base, gt) * batch_size

    return {key: value / max(total_items, 1) for key, value in sums.items()}


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_loader = build_loader(
        gt_root=args.train_gt_root,
        base_root=args.train_base_root,
        gen_root=args.train_gen_root,
        sequence_list_file=args.train_list_file,
        crop_size=args.crop_size,
        augment=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_sequences=args.max_train_sequences,
        seed=args.seed,
    )
    val_loader = build_loader(
        gt_root=args.val_gt_root or args.train_gt_root,
        base_root=args.val_base_root or args.train_base_root,
        gen_root=args.val_gen_root or args.train_gen_root,
        sequence_list_file=args.val_list_file,
        crop_size=args.val_crop_size,
        augment=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_sequences=args.max_val_sequences,
        seed=args.seed,
    )

    model = MotionCompensatedFusionNet(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        residual_scale=args.residual_scale,
        alpha_bias=args.alpha_bias,
        blend_power=args.blend_power,
        fusion_mode=args.fusion_mode,
        use_dual_reference_features=args.use_dual_reference_features,
        gated_residual=args.gated_residual,
        use_detail_delta_features=args.use_detail_delta_features,
        flow_hidden_channels=args.flow_hidden_channels,
        max_flow=args.max_flow,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    best_metric_value = float("inf") if args.selection_metric == "loss" else float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_items = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            base = batch["base"].to(device, non_blocking=True)
            gen = batch["gen"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(base, gen)
            losses = compute_losses(outputs, base, gen, gt, args)
            losses["loss"].backward()
            optimizer.step()

            batch_size = base.size(0)
            total_items += batch_size
            running_loss += losses["loss"].item() * batch_size
            progress.set_postfix(
                loss=f"{losses['loss'].item():.4f}",
                recon=f"{losses['recon'].item():.4f}",
                temp=f"{losses['temporal'].item():.4f}",
                alpha_t=f"{losses['alpha_temporal'].item():.4f}",
                flow=f"{losses['flow_smooth'].item():.4f}",
            )

        train_loss = running_loss / max(total_items, 1)
        val_metrics = evaluate(model=model, loader=val_loader, device=device, args=args)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_psnr": val_metrics["psnr"],
            "val_base_psnr": val_metrics["base_psnr"],
            "val_recon_loss": val_metrics["recon"],
            "val_temporal_loss": val_metrics["temporal"],
            "val_alpha_temporal_loss": val_metrics["alpha_temporal"],
            "val_gradient_loss": val_metrics["gradient"],
            "val_alpha_sparse_loss": val_metrics["alpha_sparse"],
            "val_preserve_loss": val_metrics["preserve"],
            "val_mask_loss": val_metrics["mask"],
            "val_flow_smooth_loss": val_metrics["flow_smooth"],
        }
        history.append(record)

        checkpoint = {
            "epoch": epoch,
            "model_type": "motion_compensated_fusion",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": {
                "hidden_channels": args.hidden_channels,
                "num_blocks": args.num_blocks,
                "residual_scale": args.residual_scale,
                "alpha_bias": args.alpha_bias,
                "blend_power": args.blend_power,
                "fusion_mode": args.fusion_mode,
                "use_dual_reference_features": args.use_dual_reference_features,
                "gated_residual": args.gated_residual,
                "use_detail_delta_features": args.use_detail_delta_features,
                "flow_hidden_channels": args.flow_hidden_channels,
                "max_flow": args.max_flow,
            },
            "train_config": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        current_metric_value = val_metrics["loss"] if args.selection_metric == "loss" else val_metrics["psnr"]
        is_best = (
            current_metric_value < best_metric_value
            if args.selection_metric == "loss"
            else current_metric_value > best_metric_value
        )
        if is_best:
            best_metric_value = current_metric_value
            torch.save(checkpoint, output_dir / "best.pt")

    (output_dir / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
