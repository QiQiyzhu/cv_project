from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model.weight_init import constant_init

from vsr_part1.io_utils import list_images
from vsr_part2.basicvsr_model import (
    PixelShufflePack,
    ResidualBlocksWithInputConv,
    SPyNet,
    _load_checkpoint,
    flow_warp,
    save_output_frames,
    sequence_to_tensor,
    tensor_to_frames,
)


DEFAULT_BASICVSRPP_VIMEO90K_BI_CKPT = (
    "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
    "basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth"
)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs) -> None:
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self) -> None:
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(
        self,
        x: torch.Tensor,
        extra_feat: torch.Tensor,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor,
    ) -> torch.Tensor:
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        offset_1, offset_2, mask = torch.chunk(out, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat((offset_1, offset_2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )


class BasicVSRPlusPlusNet(nn.Module):
    def __init__(
        self,
        mid_channels: int = 64,
        num_blocks: int = 7,
        max_residue_magnitude: int = 10,
        is_low_res_input: bool = True,
        cpu_cache_length: int = 100,
    ) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.cpu_cache = False
        self.is_mirror_extended = False

        self.spynet = SPyNet()

        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
            )

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for branch_index, module_name in enumerate(["backward_1", "forward_1", "backward_2", "forward_2"]):
            self.deform_align[module_name] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude,
            )
            self.backbone[module_name] = ResidualBlocksWithInputConv(
                (2 + branch_index) * mid_channels,
                mid_channels,
                num_blocks,
            )

        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lqs: torch.Tensor) -> None:
        self.is_mirror_extended = False
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch, time, channels, height, width = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, channels, height, width)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, channels, height, width)

        flows_backward = self.spynet(lqs_1, lqs_2).view(batch, time - 1, 2, height, width)
        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(batch, time - 1, 2, height, width)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = None if flows_forward is None else flows_forward.cpu()
        return flows_forward, flows_backward

    def propagate(
        self,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
    ) -> dict[str, list[torch.Tensor]]:
        batch, time, _, height, width = flows.size()
        frame_idx = list(range(0, time + 1))
        flow_idx = list(range(-1, time))
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(batch, self.mid_channels, height, width)
        for iter_idx, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.to(self.spynet.mean.device)
                feat_prop = feat_prop.to(self.spynet.mean.device)

            if iter_idx > 0:
                flow_n1 = flows[:, flow_idx[iter_idx], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.to(self.spynet.mean.device)

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if iter_idx > 1:
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.to(self.spynet.mean.device)

                    flow_n2 = flows[:, flow_idx[iter_idx - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.to(self.spynet.mean.device)
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            feat = [feat_current] + [feats[key][idx] for key in feats if key not in ["spatial", module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [item.to(self.spynet.mean.device) for item in feat]
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def upsample(self, lqs: torch.Tensor, feats: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        outputs = []
        num_outputs = len(feats["spatial"])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for time_idx in range(0, lqs.size(1)):
            hr = [feats[key].pop(0) for key in feats if key != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[time_idx]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.to(self.spynet.mean.device)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr = hr + self.img_upsample(lqs[:, time_idx, :, :, :])
            else:
                hr = hr + lqs[:, time_idx, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()
            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = lqs.size()

        self.cpu_cache = time > self.cpu_cache_length and lqs.is_cuda
        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, channels, height, width),
                scale_factor=0.25,
                mode="bicubic",
            ).view(batch, time, channels, height // 4, width // 4)

        self.check_if_mirror_extended(lqs)

        feats: dict[str, list[torch.Tensor]] = {}
        if self.cpu_cache:
            feats["spatial"] = []
            for time_idx in range(0, time):
                feat = self.feat_extract(lqs[:, time_idx, :, :, :]).cpu()
                feats["spatial"].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, channels, height, width))
            feat_height, feat_width = feats_.shape[2:]
            feats_ = feats_.view(batch, time, -1, feat_height, feat_width)
            feats["spatial"] = [feats_[:, time_idx, :, :, :] for time_idx in range(0, time)]

        if lqs_downsample.size(3) < 64 or lqs_downsample.size(4) < 64:
            raise ValueError(
                "BasicVSR++ expects low-res inputs with height and width at least 64, "
                f"but got {(lqs_downsample.size(3), lqs_downsample.size(4))}."
            )
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        for iter_idx in [1, 2]:
            for direction in ["backward", "forward"]:
                module_name = f"{direction}_{iter_idx}"
                feats[module_name] = []

                if direction == "backward":
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module_name)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


def load_basicvsrpp_model(
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    strict: bool = True,
) -> BasicVSRPlusPlusNet:
    if checkpoint is None:
        checkpoint = DEFAULT_BASICVSRPP_VIMEO90K_BI_CKPT
    model = BasicVSRPlusPlusNet()
    state_dict = _load_checkpoint(checkpoint)
    state_dict.pop("step_counter", None)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model


class BasicVSRPlusPlusInferencer:
    def __init__(self, device: str = "cuda", checkpoint: str | Path | None = None) -> None:
        self.device = device
        self.model = load_basicvsrpp_model(checkpoint=checkpoint, device=device)

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
            raise ValueError("BasicVSR++ is a recurrent model here; window_size must be 0.")

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
