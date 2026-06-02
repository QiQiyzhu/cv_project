from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


class SRCNN(nn.Module):
    def __init__(self, in_channels: int = 1, mapping_kernel_size: int = 5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mapping_kernel_size = mapping_kernel_size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=mapping_kernel_size, padding=mapping_kernel_size // 2),
            nn.ReLU(inplace=True),
        )
        self.reconstruction = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.map(x)
        return self.reconstruction(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight.data,
                    0.0,
                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())),
                )
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


def srcnn_kwargs_from_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, int]:
    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        in_channels = int(config.get("in_channels", 1 if config.get("color_space", "y") == "y" else 3))
        mapping_kernel_size = int(config.get("mapping_kernel_size", 5))
        return {"in_channels": in_channels, "mapping_kernel_size": mapping_kernel_size}

    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, Mapping):
        if "features.0.weight" in state_dict and "map.0.weight" in state_dict:
            return {
                "in_channels": int(state_dict["features.0.weight"].shape[1]),
                "mapping_kernel_size": int(state_dict["map.0.weight"].shape[-1]),
            }
        if "layers.0.weight" in state_dict and "layers.2.weight" in state_dict:
            return {
                "in_channels": int(state_dict["layers.0.weight"].shape[1]),
                "mapping_kernel_size": int(state_dict["layers.2.weight"].shape[-1]),
            }
    return {"in_channels": 1, "mapping_kernel_size": 5}


def normalize_srcnn_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "features.0.weight" in state_dict:
        return dict(state_dict)

    key_map = {
        "layers.0.weight": "features.0.weight",
        "layers.0.bias": "features.0.bias",
        "layers.2.weight": "map.0.weight",
        "layers.2.bias": "map.0.bias",
        "layers.4.weight": "reconstruction.weight",
        "layers.4.bias": "reconstruction.bias",
    }
    return {key_map.get(key, key): value for key, value in state_dict.items()}
