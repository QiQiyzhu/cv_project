from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from vsr_part1.io_utils import list_images, load_frame, save_frame


DEFAULT_BASICVSR_VIMEO90K_BI_CKPT = (
    "https://download.openmmlab.com/mmediting/restorers/basicvsr/"
    "basicvsr_vimeo90k_bi_20210409-d2d8f760.pth"
)


def default_init_weights(module: nn.Module, scale: float = 1.0) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
            layer.weight.data *= scale
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
            layer.weight.data *= scale
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def make_layer(block: type[nn.Module], num_blocks: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*(block(**kwargs) for _ in range(num_blocks)))


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(
            f"The spatial sizes of input ({x.size()[-2:]}) and flow ({flow.size()[1:3]}) are not the same."
        )

    _, _, height, width = x.size()
    device = flow.device
    dtype = x.dtype
    if "indexing" in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, height, device=device, dtype=dtype),
            torch.arange(0, width, device=device, dtype=dtype),
            indexing="ij",
        )
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, height, device=device, dtype=dtype),
            torch.arange(0, width, device=device, dtype=dtype),
        )
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(width - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(height - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3).type_as(x)
    return F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class ConvAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, act: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=True)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self) -> None:
        for layer in (self.conv1, self.conv2):
            default_init_weights(layer, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, upsample_kernel: int) -> None:
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2,
        )
        default_init_weights(self, 1.0)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.main(feat)


class SPyNetBasicModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvAct(8, 32, 7, act=True),
            ConvAct(32, 64, 7, act=True),
            ConvAct(64, 32, 7, act=True),
            ConvAct(32, 16, 7, act=True),
            ConvAct(16, 2, 7, act=False),
        )

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = ref.size()

        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        flow = ref[0].new_zeros(batch, 2, height // 32, width // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](
                torch.cat(
                    [
                        ref[level],
                        flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode="border"),
                        flow_up,
                    ],
                    dim=1,
                )
            )
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        height, width = ref.shape[2:4]
        width_up = width if width % 32 == 0 else 32 * (width // 32 + 1)
        height_up = height if height % 32 == 0 else 32 * (height // 32 + 1)
        ref = F.interpolate(ref, size=(height_up, width_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(height_up, width_up), mode="bilinear", align_corners=False)

        flow = F.interpolate(
            self.compute_flow(ref, supp),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        flow[:, 0, :, :] *= float(width) / float(width_up)
        flow[:, 1, :, :] *= float(height) / float(height_up)
        return flow


class BasicVSRNet(nn.Module):
    def __init__(self, mid_channels: int = 64, num_blocks: int = 30) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_mirror_extended = False

    def check_if_mirror_extended(self, lrs: torch.Tensor) -> None:
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch, time, channels, height, width = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, channels, height, width)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, channels, height, width)
        flows_backward = self.spynet(lrs_1, lrs_2).view(batch, time - 1, 2, height, width)

        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(batch, time - 1, 2, height, width)
        return flows_forward, flows_backward

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        batch, time, _, height, width = lrs.size()
        self.check_if_mirror_extended(lrs)
        flows_forward, flows_backward = self.compute_flow(lrs)

        outputs: list[torch.Tensor] = []
        feat_prop = lrs.new_zeros(batch, self.mid_channels, height, width)
        for index in range(time - 1, -1, -1):
            if index < time - 1:
                flow = flows_backward[:, index, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lrs[:, index, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        feat_prop = torch.zeros_like(feat_prop)
        for index in range(time):
            lr_curr = lrs[:, index, :, :, :]
            if index > 0:
                if flows_forward is not None:
                    flow = flows_forward[:, index - 1, :, :, :]
                else:
                    flow = flows_backward[:, -index, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            out = torch.cat([outputs[index], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            out += self.img_upsample(lr_curr)
            outputs[index] = out

        return torch.stack(outputs, dim=1)


def _load_checkpoint(checkpoint: str | Path | None) -> dict[str, torch.Tensor]:
    if checkpoint is None:
        checkpoint_obj = load_state_dict_from_url(DEFAULT_BASICVSR_VIMEO90K_BI_CKPT, map_location="cpu")
    else:
        checkpoint_path = str(checkpoint)
        if checkpoint_path.startswith(("http://", "https://")):
            checkpoint_obj = load_state_dict_from_url(checkpoint_path, map_location="cpu")
        else:
            checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
    normalized_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module.generator."):
            normalized_state_dict[key[len("module.generator.") :]] = value
        elif key.startswith("generator."):
            normalized_state_dict[key[len("generator.") :]] = value
        elif key.startswith("module."):
            normalized_state_dict[key[len("module.") :]] = value
        else:
            normalized_state_dict[key] = value
    return normalized_state_dict


def load_basicvsr_model(
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    strict: bool = True,
) -> BasicVSRNet:
    model = BasicVSRNet()
    state_dict = _load_checkpoint(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model


def sequence_to_tensor(image_paths: Iterable[Path]) -> torch.Tensor:
    frames = [load_frame(path).astype("float32") / 255.0 for path in image_paths]
    if not frames:
        raise ValueError("At least one frame is required for BasicVSR inference.")
    tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).contiguous()
    return tensor.unsqueeze(0)


def tensor_to_frames(output: torch.Tensor) -> list[torch.Tensor]:
    return [frame for frame in output.squeeze(0).detach().cpu()]


def save_output_frames(frames: Iterable[torch.Tensor], output_dir: Path, filename_tmpl: str, start_idx: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, frame in enumerate(frames, start=start_idx):
        image = (
            frame.clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .byte()
            .numpy()
        )
        save_frame(image, output_dir / filename_tmpl.format(offset))


class BasicVSRInferencer:
    def __init__(self, device: str = "cuda", checkpoint: str | Path | None = None) -> None:
        self.device = device
        self.model = load_basicvsr_model(checkpoint=checkpoint, device=device)

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
            raise ValueError("BasicVSR is a recurrent model here; window_size must be 0.")

        image_paths = list_images(video)
        inputs = sequence_to_tensor(image_paths)
        outputs = self._forward(inputs, max_seq_len=max_seq_len)
        save_output_frames(tensor_to_frames(outputs), Path(result_out_dir), filename_tmpl, start_idx)

    def _forward(self, inputs: torch.Tensor, max_seq_len: int | None = None) -> torch.Tensor:
        if max_seq_len is None or max_seq_len <= 0 or inputs.size(1) <= max_seq_len:
            return self.model(inputs.to(self.device)).cpu()

        chunks = []
        for start in range(0, inputs.size(1), max_seq_len):
            end = min(inputs.size(1), start + max_seq_len)
            chunks.append(self.model(inputs[:, start:end].to(self.device)).cpu())
        return torch.cat(chunks, dim=1)
