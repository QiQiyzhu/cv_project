#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vsr_part3 import FusionSequenceDataset, LearnableTemporalFusionNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Part 3 Gated Residual Fusion model.")
    parser.add_argument("--train-gt-root", required=True, help="GT root for training sequences.")
    parser.add_argument("--train-base-root", required=True, help="BasicVSR++ output root for training sequences.")
    parser.add_argument("--train-gen-root", required=True, help="Detail branch output root for training sequences, typically HAT-L.")
    parser.add_argument("--train-list-file", default=None, help="Optional sequence list for training.")
    parser.add_argument("--val-gt-root", default=None, help="Optional GT root for validation. Defaults to train GT root.")
    parser.add_argument("--val-base-root", default=None, help="Optional base root for validation. Defaults to train base root.")
    parser.add_argument("--val-gen-root", default=None, help="Optional gen root for validation. Defaults to train gen root.")
    parser.add_argument("--val-list-file", default=None, help="Optional validation sequence list.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--crop-size", type=int, default=128, help="Random crop size for training clips.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument("--lambda-temporal", type=float, default=0.1)
    parser.add_argument("--lambda-gradient", type=float, default=0.0)
    parser.add_argument("--lambda-alpha", type=float, default=0.02)
    parser.add_argument("--lambda-base-preserve", type=float, default=2.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--oracle-margin", type=float, default=0.0)
    parser.add_argument("--alpha-bias", type=float, default=-3.0)
    parser.add_argument("--blend-power", type=float, default=2.0)
    parser.add_argument(
        "--fusion-mode",
        choices=("full_blend", "detail_residual"),
        default="full_blend",
        help="How the network applies the generative branch to the base branch.",
    )
    parser.add_argument(
        "--use-dual-reference-features",
        action="store_true",
        help="Expose base/gen edge and texture cues separately instead of only coarse summary maps.",
    )
    parser.add_argument(
        "--gated-residual",
        action="store_true",
        help="Use an explicit residual gate instead of tying residual amplitude directly to alpha.",
    )
    parser.add_argument(
        "--use-detail-delta-features",
        action="store_true",
        help="Expose high-frequency delta features between the generative and base branches.",
    )
    parser.add_argument("--mask-focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--selection-metric",
        choices=("loss", "psnr"),
        default="loss",
        help="Metric used to decide which checkpoint is saved as best.pt.",
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


def temporal_difference_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.size(1) <= 1:
        return pred.new_tensor(0.0)
    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return F.l1_loss(pred_delta, target_delta)


def gradient_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def oracle_mask(base: torch.Tensor, gen: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    base_err = (base - target).square().mean(dim=2, keepdim=True)
    gen_err = (gen - target).square().mean(dim=2, keepdim=True)
    return (gen_err + margin < base_err).float()


def base_preservation_loss(fused: torch.Tensor, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fused_err = (fused - target).abs().mean(dim=2, keepdim=True)
    base_err = (base - target).abs().mean(dim=2, keepdim=True)
    return F.relu(fused_err - base_err).mean()


def weighted_mask_loss(
    alpha_logits: torch.Tensor,
    mask_target: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    positive = mask_target.mean().clamp(min=1e-6, max=1.0 - 1e-6)
    pos_weight = ((1.0 - positive) / positive).detach()
    bce = F.binary_cross_entropy_with_logits(alpha_logits, mask_target, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(alpha_logits)
    pt = torch.where(mask_target > 0.5, probs, 1.0 - probs)
    focal = (1.0 - pt).clamp(min=1e-6).pow(gamma)
    return (bce * focal).mean()


def evaluate(
    model: LearnableTemporalFusionNet,
    loader: DataLoader,
    device: torch.device,
    lambda_temporal: float,
    lambda_gradient: float,
    lambda_alpha: float,
    lambda_base_preserve: float,
    lambda_mask: float,
    oracle_margin: float,
    mask_focal_gamma: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_base_psnr = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            base = batch["base"].to(device, non_blocking=True)
            gen = batch["gen"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            outputs = model(base, gen)
            fused = outputs["fused"]
            alpha = outputs["alpha"]
            alpha_logits = outputs["alpha_logits"]
            mask_target = oracle_mask(base, gen, gt, margin=oracle_margin)
            loss = F.l1_loss(fused, gt)
            loss = loss + lambda_temporal * temporal_difference_loss(fused, gt)
            loss = loss + lambda_gradient * gradient_consistency_loss(fused, gt)
            loss = loss + lambda_alpha * alpha.mean()
            loss = loss + lambda_base_preserve * base_preservation_loss(fused, base, gt)
            loss = loss + lambda_mask * weighted_mask_loss(alpha_logits, mask_target, gamma=mask_focal_gamma)

            batch_size = base.size(0)
            total_items += batch_size
            total_loss += loss.item() * batch_size
            total_psnr += compute_psnr(fused, gt) * batch_size
            total_base_psnr += compute_psnr(base, gt) * batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "psnr": total_psnr / max(total_items, 1),
        "base_psnr": total_base_psnr / max(total_items, 1),
    }


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
        crop_size=args.crop_size,
        augment=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_sequences=args.max_val_sequences,
        seed=args.seed,
    )

    model = LearnableTemporalFusionNet(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        residual_scale=args.residual_scale,
        alpha_bias=args.alpha_bias,
        blend_power=args.blend_power,
        fusion_mode=args.fusion_mode,
        use_dual_reference_features=args.use_dual_reference_features,
        gated_residual=args.gated_residual,
        use_detail_delta_features=args.use_detail_delta_features,
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
            fused = outputs["fused"]
            alpha = outputs["alpha"]
            alpha_logits = outputs["alpha_logits"]
            mask_target = oracle_mask(base, gen, gt, margin=args.oracle_margin)

            recon_loss = F.l1_loss(fused, gt)
            temp_loss = temporal_difference_loss(fused, gt)
            grad_loss = gradient_consistency_loss(fused, gt)
            alpha_loss = alpha.mean()
            preserve_loss = base_preservation_loss(fused, base, gt)
            mask_loss = weighted_mask_loss(alpha_logits, mask_target, gamma=args.mask_focal_gamma)
            loss = recon_loss + args.lambda_temporal * temp_loss + args.lambda_alpha * alpha_loss
            loss = loss + args.lambda_gradient * grad_loss
            loss = loss + args.lambda_base_preserve * preserve_loss + args.lambda_mask * mask_loss
            loss.backward()
            optimizer.step()

            batch_size = base.size(0)
            total_items += batch_size
            running_loss += loss.item() * batch_size
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                temp=f"{temp_loss.item():.4f}",
                grad=f"{grad_loss.item():.4f}",
                alpha=f"{alpha_loss.item():.4f}",
                preserve=f"{preserve_loss.item():.4f}",
                mask=f"{mask_loss.item():.4f}",
            )

        train_loss = running_loss / max(total_items, 1)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_temporal=args.lambda_temporal,
            lambda_gradient=args.lambda_gradient,
            lambda_alpha=args.lambda_alpha,
            lambda_base_preserve=args.lambda_base_preserve,
            lambda_mask=args.lambda_mask,
            oracle_margin=args.oracle_margin,
            mask_focal_gamma=args.mask_focal_gamma,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_psnr": val_metrics["psnr"],
            "val_base_psnr": val_metrics["base_psnr"],
        }
        history.append(record)

        checkpoint = {
            "epoch": epoch,
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
