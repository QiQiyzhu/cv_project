#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vsr_part1.datasets import SRCNNPatchDataset
from vsr_part1.srcnn import SRCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SRCNN-style baseline for Part 1.")
    parser.add_argument(
        "--train-hr-dir",
        required=True,
        help="Directory containing HR training frames. Nested sequence directories are supported.",
    )
    parser.add_argument(
        "--val-hr-dir",
        default=None,
        help="Optional directory containing HR validation frames. Nested sequence directories are supported.",
    )
    parser.add_argument("--train-list-file", default=None, help="Optional official split list file for training.")
    parser.add_argument("--val-list-file", default=None, help="Optional official split list file for validation.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--samples-per-image", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reconstruction-lr-factor", type=float, default=0.1)
    parser.add_argument("--color-space", choices=("y", "rgb"), default="y")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_loader(
    hr_dir: str,
    sequence_list_file: str | None,
    scale: int,
    patch_size: int,
    samples_per_image: int,
    color_space: str,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> DataLoader:
    dataset = SRCNNPatchDataset(
        hr_dir=hr_dir,
        sequence_list_file=sequence_list_file,
        scale=scale,
        patch_size=patch_size,
        samples_per_image=samples_per_image,
        color_space=color_space,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: SRCNN, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
    return total_loss / max(total_items, 1)


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
        hr_dir=args.train_hr_dir,
        sequence_list_file=args.train_list_file,
        scale=args.scale,
        patch_size=args.patch_size,
        samples_per_image=args.samples_per_image,
        color_space=args.color_space,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )
    val_loader = None
    val_hr_dir = args.val_hr_dir or (args.train_hr_dir if args.val_list_file else None)
    if val_hr_dir:
        val_loader = build_loader(
            hr_dir=val_hr_dir,
            sequence_list_file=args.val_list_file,
            scale=args.scale,
            patch_size=args.patch_size,
            samples_per_image=max(1, args.samples_per_image // 2),
            color_space=args.color_space,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
        )

    in_channels = 1 if args.color_space == "y" else 3
    mapping_kernel_size = 5
    model = SRCNN(in_channels=in_channels, mapping_kernel_size=mapping_kernel_size).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [
                {"params": model.features.parameters()},
                {"params": model.map.parameters()},
                {"params": model.reconstruction.parameters(), "lr": args.lr * args.reconstruction_lr_factor},
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    history = []
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_items = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_items += batch_size
            progress.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = running_loss / max(total_items, 1)
        val_loss = evaluate(model, val_loader, criterion, device) if val_loader is not None else train_loss
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {**vars(args), "in_channels": in_channels, "mapping_kernel_size": mapping_kernel_size},
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_metric:
            best_metric = val_loss
            torch.save(checkpoint, output_dir / "best.pt")

    (output_dir / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
