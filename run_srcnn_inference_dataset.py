#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from vsr_part2 import run_vsr_dataset_inference, save_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 video SR inference over a nested dataset root.")
    parser.add_argument("--input-root", required=True, help="Root directory of low-resolution input sequences.")
    parser.add_argument("--output-root", required=True, help="Root directory for super-resolved output sequences.")
    parser.add_argument(
        "--model-name",
        default="basicvsr",
        help="In-repo Part 2 model name, for example basicvsr, basicvsr++, realesrgan, or hat_l.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Unused for the in-repo runtimes. Keep empty unless a future model explicitly needs it.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path or URL. If omitted, the runtime uses the default pretrained weights.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--filename-tmpl",
        default="auto",
        help=(
            "Frame filename template for MMagic, for example '{:08d}.png' or 'im{:01d}.png'. "
            "Use 'auto' to infer it from each sequence."
        ),
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Starting frame index for the filename template. If omitted with auto template, it is inferred.",
    )
    parser.add_argument(
        "--sequence-list-file",
        default=None,
        help="Optional relative sequence list file. When provided, only listed sequences are processed.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional limit for smoke tests or partial runs.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=100,
        help="Maximum recurrent chunk length for BasicVSR-style models.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Optional inference window size override. Keep 0 for the default runtime behavior.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to save the inference summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_vsr_dataset_inference(
        input_root=args.input_root,
        output_root=args.output_root,
        model_name=args.model_name,
        device=args.device,
        model_config=args.config,
        model_ckpt=args.checkpoint,
        filename_tmpl=args.filename_tmpl,
        start_idx=args.start_idx,
        sequence_list_file=args.sequence_list_file,
        max_sequences=args.max_sequences,
        max_seq_len=args.max_seq_len,
        window_size=args.window_size,
    )
    if args.summary_json:
        save_summary(summary, args.summary_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
