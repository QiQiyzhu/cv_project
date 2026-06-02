#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from vsr_part3 import run_fusion_dataset_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 3 paper fusion over a nested dataset root.")
    parser.add_argument("--base-root", required=True, help="Root directory of BasicVSR++ outputs.")
    parser.add_argument("--gen-root", required=True, help="Root directory of the detail branch outputs, typically HAT-L.")
    parser.add_argument("--output-root", required=True, help="Root directory for fused outputs.")
    parser.add_argument("--checkpoint", required=True, help="Fusion checkpoint path, usually best.pt.")
    parser.add_argument("--device", default="cuda", help="Device to use, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--fused-delta-scale", type=float, default=1.0, help="Scale applied to (fused - base) before saving.")
    parser.add_argument("--save-alpha", action="store_true", help="Save alpha maps next to fused outputs.")
    parser.add_argument("--alpha-root", default=None, help="Optional output root for alpha maps.")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--summary-json", default=None, help="Optional JSON path for the inference summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_fusion_dataset_inference(
        base_root=args.base_root,
        gen_root=args.gen_root,
        output_root=args.output_root,
        checkpoint_path=args.checkpoint,
        device=args.device,
        fused_delta_scale=args.fused_delta_scale,
        save_alpha=args.save_alpha,
        alpha_root=args.alpha_root,
        max_sequences=args.max_sequences,
    )
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
