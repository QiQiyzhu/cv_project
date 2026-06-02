from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.hub import load_state_dict_from_url

from vsr_part1.io_utils import list_images, load_frame, save_frame
from vsr_part2.hat_arch import HAT


DEFAULT_HAT_CHECKPOINT_URLS = {
    "hat": "https://huggingface.co/Acly/hat/resolve/main/HAT_SRx4_ImageNet-pretrain.pth",
    "hat_s": "https://huggingface.co/Acly/hat/resolve/main/HAT-S_SRx4.pth",
    "hat_l": "https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/HAT-L_SRx4_ImageNet-pretrain.pth",
}

DEFAULT_HAT_FILENAMES = {
    "hat": "HAT_SRx4_ImageNet-pretrain.pth",
    "hat_s": "HAT-S_SRx4.pth",
    "hat_l": "HAT-L_SRx4_ImageNet-pretrain.pth",
}

HAT_CONFIGS = {
    "hat": {
        "upscale": 4,
        "in_chans": 3,
        "img_size": 64,
        "window_size": 16,
        "compress_ratio": 3,
        "squeeze_factor": 30,
        "conv_scale": 0.01,
        "overlap_ratio": 0.5,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
    },
    "hat_s": {
        "upscale": 4,
        "in_chans": 3,
        "img_size": 64,
        "window_size": 16,
        "compress_ratio": 24,
        "squeeze_factor": 24,
        "conv_scale": 0.01,
        "overlap_ratio": 0.5,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6, 6, 6],
        "embed_dim": 144,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
    },
    "hat_l": {
        "upscale": 4,
        "in_chans": 3,
        "img_size": 64,
        "window_size": 16,
        "compress_ratio": 3,
        "squeeze_factor": 30,
        "conv_scale": 0.01,
        "overlap_ratio": 0.5,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
    },
}


def _normalize_state_dict(raw_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized_state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        normalized_key = key
        for prefix in ("module.", "params.", "net_g.", "generator."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict


def _select_state_dict(checkpoint_obj: dict[str, object] | torch.Tensor) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("params_ema", "params", "state_dict"):
            maybe_state = checkpoint_obj.get(key)
            if isinstance(maybe_state, dict):
                return _normalize_state_dict(maybe_state)
        if checkpoint_obj and all(torch.is_tensor(value) for value in checkpoint_obj.values()):
            return _normalize_state_dict(checkpoint_obj)  # type: ignore[arg-type]
    raise ValueError("Unsupported HAT checkpoint format.")


def _default_checkpoint_candidates(variant: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    filename = DEFAULT_HAT_FILENAMES[variant]
    return [
        os.environ.get("HAT_CHECKPOINT"),
        str(repo_root / "checkpoints" / "hat" / filename),
        str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename),
    ]


def _load_checkpoint(checkpoint: str | Path | None, variant: str) -> dict[str, torch.Tensor]:
    if checkpoint is None:
        resolved_local_path = next(
            (path for path in _default_checkpoint_candidates(variant) if path and Path(path).is_file()),
            None,
        )
        if resolved_local_path is not None:
            checkpoint_obj = torch.load(resolved_local_path, map_location="cpu")
        else:
            checkpoint_obj = load_state_dict_from_url(DEFAULT_HAT_CHECKPOINT_URLS[variant], map_location="cpu")
    else:
        checkpoint_path = str(checkpoint)
        if checkpoint_path.startswith(("http://", "https://")):
            checkpoint_obj = load_state_dict_from_url(checkpoint_path, map_location="cpu")
        else:
            checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    return _select_state_dict(checkpoint_obj)


def load_hat_model(
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    variant: str = "hat",
) -> HAT:
    if variant not in HAT_CONFIGS:
        raise ValueError(f"Unsupported HAT variant: {variant}")

    model = HAT(**HAT_CONFIGS[variant])
    state_dict = _load_checkpoint(checkpoint=checkpoint, variant=variant)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_frame(output: torch.Tensor) -> np.ndarray:
    return (
        output.squeeze(0)
        .detach()
        .cpu()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )


class HATInferencer:
    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str | Path | None = None,
        variant: str = "hat",
    ) -> None:
        self.device = device
        self.variant = variant
        self.model = load_hat_model(checkpoint=checkpoint, device=device, variant=variant)

    @torch.inference_mode()
    def infer(
        self,
        video: str | Path,
        result_out_dir: str | Path,
        start_idx: int = 0,
        filename_tmpl: str = "{:08d}.png",
        max_seq_len: int | None = None,
        window_size: int = 0,
    ) -> None:
        if window_size not in (0, None):
            raise ValueError("HAT is frame-wise here; window_size must be 0.")
        _ = max_seq_len

        output_dir = Path(result_out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for offset, image_path in enumerate(list_images(video), start=start_idx):
            input_tensor = frame_to_tensor(load_frame(image_path)).to(self.device)
            output_tensor = self.model(input_tensor)
            save_frame(tensor_to_frame(output_tensor), output_dir / filename_tmpl.format(offset))
