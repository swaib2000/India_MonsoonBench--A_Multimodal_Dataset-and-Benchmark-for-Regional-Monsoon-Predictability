#!/usr/bin/env python3
"""
Train baseline models on the extracted patch dataset and saved split indices.

Expected inputs inside --dataset-dir:
- X.npy
- y.npy
- train_idx.npy
- val_idx.npy
- test_idx.npy
- config.json

Outputs inside --output-dir:
- best_model.pt
- metrics.json
- history.csv
- test_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for train_patch_baselines.py. "
        "Install torch in your training environment first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline patch models")
    parser.add_argument("--dataset-dir", default="patch_dataset")
    parser.add_argument("--output-dir", default="baseline_runs/default")
    parser.add_argument(
        "--model",
        default="cnn",
        choices=["cnn", "resnet", "unet", "conv3d", "convlstm", "swin3d", "swin3d_yearmonth"],
    )
    parser.add_argument(
        "--drop-modalities",
        default="",
        help=(
            "Comma-separated predictor names to ablate by zeroing their channels at train/val/test time. "
            "Examples: 'humidity', 'rh,wind_speed', 'elevation'."
        ),
    )
    parser.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "focal", "ordinal", "ce_ordinal", "focal_ordinal"],
        help=(
            "Training loss. 'ordinal' uses a CDF-based ordinal loss, and "
            "'ce_ordinal' combines cross-entropy with the ordinal term."
        ),
    )
    parser.add_argument(
        "--ordinal-weight",
        type=float,
        default=0.5,
        help="Weight for the ordinal component when --loss=ce_ordinal",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss variants",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-samples", type=int, default=50000)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run all epochs while still saving/evaluating the best validation checkpoint.",
    )
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpyPatchDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        idx_path: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        channel_mask: np.ndarray | None = None,
    ):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = np.load(idx_path)
        self.mean = mean
        self.std = std
        self.channel_mask = channel_mask

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = np.array(self.X[idx], dtype=np.float32, copy=True)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        if self.channel_mask is not None:
            x *= self.channel_mask[:, None, None]
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), idx


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class SmallResNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlainUNetClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _match_spatial(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Center-crop source so it matches the target spatial size.
        This keeps the U-Net robust for odd-sized patches like 15x15.
        """
        if source.shape[-2:] == target.shape[-2:]:
            return source

        target_h, target_w = target.shape[-2:]
        source_h, source_w = source.shape[-2:]
        start_h = max((source_h - target_h) // 2, 0)
        start_w = max((source_w - target_w) // 2, 0)
        return source[:, :, start_h : start_h + target_h, start_w : start_w + target_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.up3(bottleneck)
        enc3 = self._match_spatial(enc3, dec3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        enc2 = self._match_spatial(enc2, dec2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        enc1 = self._match_spatial(enc1, dec1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        logits = self.head(dec1)
        center_y = logits.shape[-2] // 2
        center_x = logits.shape[-1] // 2
        return logits[:, :, center_y, center_x]


class Conv3DBaseline(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        static_channel_count: int,
        dynamic_channel_count: int,
        time_steps: int,
    ):
        super().__init__()
        if dynamic_channel_count <= 0 or time_steps <= 0:
            raise ValueError("Conv3D baseline requires dynamic channels and time steps from config.json")
        expected = static_channel_count + dynamic_channel_count * time_steps
        if expected != in_channels:
            raise ValueError(
                f"Conv3D expected {expected} channels from config.json, got {in_channels}"
            )
        self.static_channel_count = static_channel_count
        self.dynamic_channel_count = dynamic_channel_count
        self.time_steps = time_steps
        volume_channels = static_channel_count + dynamic_channel_count
        self.features = nn.Sequential(
            nn.Conv3d(volume_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        static = x[:, : self.static_channel_count, :, :]
        dynamic = x[:, self.static_channel_count :, :, :]
        dynamic = dynamic.reshape(batch_size, self.dynamic_channel_count, self.time_steps, height, width)
        if self.static_channel_count > 0:
            static = static.unsqueeze(2).expand(-1, -1, self.time_steps, -1, -1)
            volume = torch.cat([static, dynamic], dim=1)
        else:
            volume = dynamic
        return self.head(self.features(volume))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMBaseline(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        static_channel_count: int,
        dynamic_channel_count: int,
        time_steps: int,
        hidden_channels: int = 64,
    ):
        super().__init__()
        if dynamic_channel_count <= 0 or time_steps <= 0:
            raise ValueError("ConvLSTM baseline requires dynamic channels and time steps from config.json")
        expected = static_channel_count + dynamic_channel_count * time_steps
        if expected != in_channels:
            raise ValueError(f"ConvLSTM expected {expected} channels from config.json, got {in_channels}")
        self.static_channel_count = static_channel_count
        self.dynamic_channel_count = dynamic_channel_count
        self.time_steps = time_steps
        step_channels = static_channel_count + dynamic_channel_count
        self.encoder = nn.Sequential(
            nn.Conv2d(step_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.convlstm = ConvLSTMCell(hidden_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        static = x[:, : self.static_channel_count, :, :]
        dynamic = x[:, self.static_channel_count :, :, :]
        dynamic = dynamic.reshape(batch_size, self.dynamic_channel_count, self.time_steps, height, width)

        h = None
        c = None
        for step_idx in range(self.time_steps):
            step_dynamic = dynamic[:, :, step_idx, :, :]
            if self.static_channel_count > 0:
                step_input = torch.cat([static, step_dynamic], dim=1)
            else:
                step_input = step_dynamic
            encoded = self.encoder(step_input)
            if h is None:
                h = torch.zeros_like(encoded)
                c = torch.zeros_like(encoded)
            h, c = self.convlstm(encoded, h, c)
        return self.head(h)


def window_partition_3d(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    batch_size, depth, height, width, channels = x.shape
    window_d, window_h, window_w = window_size
    x = x.view(
        batch_size,
        depth // window_d,
        window_d,
        height // window_h,
        window_h,
        width // window_w,
        window_w,
        channels,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, window_d * window_h * window_w, channels)


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    batch_size: int,
    depth: int,
    height: int,
    width: int,
) -> torch.Tensor:
    window_d, window_h, window_w = window_size
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        depth // window_d,
        height // window_h,
        width // window_w,
        window_d,
        window_h,
        window_w,
        channels,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch_size, depth, height, width, channels)


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, window_size: tuple[int, int, int], num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(tokens, tokens, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(x)


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int, int],
        shift_size: tuple[int, int, int],
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1),
        )

    def _attention_mask(
        self,
        batch_size: int,
        depth: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if max(self.shift_size) == 0:
            return None

        window_d, window_h, window_w = self.window_size
        shift_d, shift_h, shift_w = self.shift_size
        img_mask = torch.zeros((1, depth, height, width, 1), device=device)
        d_slices = (slice(0, -window_d), slice(-window_d, -shift_d), slice(-shift_d, None))
        h_slices = (slice(0, -window_h), slice(-window_h, -shift_h), slice(-shift_h, None))
        w_slices = (slice(0, -window_w), slice(-window_w, -shift_w), slice(-shift_w, None))
        counter = 0
        for d_slice in d_slices:
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, d_slice, h_slice, w_slice, :] = counter
                    counter += 1
        mask_windows = window_partition_3d(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, depth, height, width, channels = x.shape
        shortcut = x
        x = self.norm1(x)

        if max(self.shift_size) > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
            attn_mask = self._attention_mask(batch_size, depth, height, width, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition_3d(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, batch_size, depth, height, width)

        if max(self.shift_size) > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class Swin3DBaseline(nn.Module):
    """
    Compact 3D Swin-style transformer for small multimodal weather patches.
    It uses shifted local attention over time, height, and width, then pools the
    resulting spatiotemporal tokens for rainfall-regime classification.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        static_channel_count: int,
        dynamic_channel_count: int,
        time_steps: int,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        window_size: tuple[int, int, int] = (2, 5, 5),
    ):
        super().__init__()
        if dynamic_channel_count <= 0 or time_steps <= 0:
            raise ValueError("Swin3D baseline requires dynamic channels and time steps from config.json")
        expected = static_channel_count + dynamic_channel_count * time_steps
        if expected != in_channels:
            raise ValueError(f"Swin3D expected {expected} channels from config.json, got {in_channels}")
        self.static_channel_count = static_channel_count
        self.dynamic_channel_count = dynamic_channel_count
        self.time_steps = time_steps
        self.window_size = window_size
        volume_channels = static_channel_count + dynamic_channel_count

        self.patch_embed = nn.Sequential(
            nn.Conv3d(volume_channels, embed_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )
        blocks = []
        for block_idx in range(depth):
            if block_idx % 2 == 0:
                shift_size = (0, 0, 0)
            else:
                shift_size = tuple(size // 2 for size in window_size)
            blocks.append(
                SwinTransformerBlock3D(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def _to_volume(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        static = x[:, : self.static_channel_count, :, :]
        dynamic = x[:, self.static_channel_count :, :, :]
        dynamic = dynamic.reshape(batch_size, self.dynamic_channel_count, self.time_steps, height, width)
        if self.static_channel_count > 0:
            static = static.unsqueeze(2).expand(-1, -1, self.time_steps, -1, -1)
            return torch.cat([static, dynamic], dim=1)
        return dynamic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        volume = self._to_volume(x)
        x = self.patch_embed(volume)
        _, _, depth, height, width = x.shape
        pad_d = (self.window_size[0] - depth % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - height % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - width % self.window_size[2]) % self.window_size[2]
        if pad_d or pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        for block in self.blocks:
            x = block(x)

        if pad_d or pad_h or pad_w:
            x = x[:, :depth, :height, :width, :]
        x = self.norm(x)
        x = x.mean(dim=(1, 2, 3))
        return self.head(x)


class SameMonthYearAttention(nn.Module):
    """
    Axial attention over years for each fixed calendar month and spatial location.
    For example, all January tokens across input years attend to one another,
    all February tokens attend to one another, and so on.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, year_steps, month_steps, height, width = x.shape
        tokens = x.permute(0, 3, 4, 5, 2, 1).contiguous()
        tokens = tokens.view(batch_size * month_steps * height * width, year_steps, channels)
        normed = self.norm(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        tokens = tokens.view(batch_size, month_steps, height, width, year_steps, channels)
        return tokens.permute(0, 5, 4, 1, 2, 3).contiguous()


class YearMonthSwin3DBaseline(nn.Module):
    """
    Year-month-aware 3D Swin baseline.

    The first stage explicitly lets each calendar month attend to the same
    calendar month in other years. The second stage applies shifted-window 3D
    attention over the flattened year-month volume.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        static_channel_count: int,
        dynamic_channel_count: int,
        time_steps: int,
        year_steps: int,
        month_steps: int,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        window_size: tuple[int, int, int] = (2, 5, 5),
    ):
        super().__init__()
        if dynamic_channel_count <= 0 or time_steps <= 0:
            raise ValueError("YearMonthSwin3D requires dynamic channels and time steps from config.json")
        if year_steps <= 0 or month_steps <= 0:
            raise ValueError(
                "YearMonthSwin3D requires year_steps and month_steps in config.json. "
                "Create data with extract_patches.py --sample-mode year_sequence_forecast."
            )
        if year_steps * month_steps != time_steps:
            raise ValueError(f"Expected year_steps*month_steps == time_steps, got {year_steps}*{month_steps}!={time_steps}")
        expected = static_channel_count + dynamic_channel_count * time_steps
        if expected != in_channels:
            raise ValueError(f"YearMonthSwin3D expected {expected} channels from config.json, got {in_channels}")

        self.static_channel_count = static_channel_count
        self.dynamic_channel_count = dynamic_channel_count
        self.time_steps = time_steps
        self.year_steps = year_steps
        self.month_steps = month_steps
        self.window_size = window_size
        volume_channels = static_channel_count + dynamic_channel_count

        self.patch_embed = nn.Sequential(
            nn.Conv3d(volume_channels, embed_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )
        self.same_month_year_attention = SameMonthYearAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if block_idx % 2 == 0 else tuple(size // 2 for size in window_size),
                )
                for block_idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def _to_volume(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        static = x[:, : self.static_channel_count, :, :]
        dynamic = x[:, self.static_channel_count :, :, :]
        dynamic = dynamic.reshape(batch_size, self.dynamic_channel_count, self.time_steps, height, width)
        if self.static_channel_count > 0:
            static = static.unsqueeze(2).expand(-1, -1, self.time_steps, -1, -1)
            return torch.cat([static, dynamic], dim=1)
        return dynamic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        volume = self._to_volume(x)
        x = self.patch_embed(volume)
        _, _, depth, height, width = x.shape

        x = x.reshape(x.shape[0], x.shape[1], self.year_steps, self.month_steps, height, width)
        x = self.same_month_year_attention(x)
        x = x.reshape(x.shape[0], x.shape[1], depth, height, width)

        pad_d = (self.window_size[0] - depth % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - height % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - width % self.window_size[2]) % self.window_size[2]
        if pad_d or pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        for block in self.blocks:
            x = block(x)

        if pad_d or pad_h or pad_w:
            x = x[:, :depth, :height, :width, :]
        x = self.norm(x)
        x = x.mean(dim=(1, 2, 3))
        return self.head(x)


class OrdinalCDFLoss(nn.Module):
    """
    Ordinal loss based on the distance between predicted and target cumulative distributions.
    This penalizes far-away rainfall-class errors more than adjacent-class errors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        pred_cdf = torch.cumsum(probs, dim=1)
        target_one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
        target_cdf = torch.cumsum(target_one_hot, dim=1)
        return torch.mean((pred_cdf - target_cdf) ** 2)


class CombinedOrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, ce_loss: nn.Module, ordinal_weight: float):
        super().__init__()
        self.ce_loss = ce_loss
        self.ordinal_loss = OrdinalCDFLoss()
        self.ordinal_weight = ordinal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_term = self.ce_loss(logits, targets)
        ordinal_term = self.ordinal_loss(logits, targets)
        return ce_term + self.ordinal_weight * ordinal_term


class FocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_term * target_log_probs
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss * weights
        return loss.mean()


class CombinedOrdinalBaseLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, ordinal_weight: float):
        super().__init__()
        self.base_loss = base_loss
        self.ordinal_loss = OrdinalCDFLoss()
        self.ordinal_weight = ordinal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_term = self.base_loss(logits, targets)
        ordinal_term = self.ordinal_loss(logits, targets)
        return base_term + self.ordinal_weight * ordinal_term


def parse_drop_modalities(text: str) -> list[str]:
    if not text.strip():
        return []
    aliases = {
        "humidity": "rh",
        "relative_humidity": "rh",
        "soil": "soil_moisture",
        "soilmoisture": "soil_moisture",
        "wind": "wind_speed",
        "windspeed": "wind_speed",
        "lst": "lst",
        "ndvi": "ndvi",
        "elevation": "elevation",
        "rh": "rh",
        "soil_moisture": "soil_moisture",
        "wind_speed": "wind_speed",
    }
    modalities = []
    for token in text.split(","):
        key = token.strip().lower()
        if not key:
            continue
        modalities.append(aliases.get(key, key))
    return sorted(set(modalities))


def build_channel_mask(dataset_config: dict, drop_modalities: list[str], in_channels: int) -> np.ndarray:
    mask = np.ones(in_channels, dtype=np.float32)
    if not drop_modalities:
        return mask

    channel_names = dataset_config.get("channel_names", [])
    if len(channel_names) != in_channels:
        raise ValueError(
            f"config.json channel_names length {len(channel_names)} does not match in_channels={in_channels}"
        )

    dropped = 0
    for idx, channel_name in enumerate(channel_names):
        for modality in drop_modalities:
            if channel_name == modality or channel_name.startswith(f"{modality}_"):
                mask[idx] = 0.0
                dropped += 1
                break

    if dropped == 0:
        raise ValueError(f"No channels matched requested modalities: {', '.join(drop_modalities)}")
    return mask


def compute_channel_stats(x_path: str, train_idx_path: str, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path, mmap_mode="r")
    train_idx = np.load(train_idx_path)
    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(train_idx) > max_samples:
        train_idx = rng.choice(train_idx, size=max_samples, replace=False)

    channel_sum = None
    channel_sq_sum = None
    total_pixels = 0
    batch_size = 256

    for start in range(0, len(train_idx), batch_size):
        batch_idx = train_idx[start : start + batch_size]
        batch = np.asarray(X[batch_idx], dtype=np.float64)
        batch_sum = batch.sum(axis=(0, 2, 3))
        batch_sq_sum = np.square(batch).sum(axis=(0, 2, 3))
        pixels = batch.shape[0] * batch.shape[2] * batch.shape[3]
        if channel_sum is None:
            channel_sum = batch_sum
            channel_sq_sum = batch_sq_sum
        else:
            channel_sum += batch_sum
            channel_sq_sum += batch_sq_sum
        total_pixels += pixels

    mean = channel_sum / total_pixels
    var = channel_sq_sum / total_pixels - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def make_class_weights(y_path: str, train_idx_path: str, num_classes: int) -> torch.Tensor:
    y = np.load(y_path, mmap_mode="r")
    idx = np.load(train_idx_path)
    counts = np.bincount(np.array(y[idx], dtype=np.int64), minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        matrix[int(y_true), int(y_pred)] += 1
    return matrix


def per_class_f1_from_confusion(matrix: np.ndarray) -> list[float]:
    scores = []
    for cls_idx in range(matrix.shape[0]):
        tp = float(matrix[cls_idx, cls_idx])
        fp = float(matrix[:, cls_idx].sum() - tp)
        fn = float(matrix[cls_idx, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return scores


def weighted_f1(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    matrix = confusion_matrix(labels, preds, num_classes)
    f1_scores = per_class_f1_from_confusion(matrix)
    supports = matrix.sum(axis=1).astype(np.float64)
    total = supports.sum()
    if total == 0:
        return 0.0
    return float(sum(score * support for score, support in zip(f1_scores, supports)) / total)


def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    preds_all = []
    labels_all = []
    sample_ids_all = []

    with torch.no_grad():
        for x, y, sample_ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
            preds_all.append(pred.cpu().numpy())
            labels_all.append(y.cpu().numpy())
            sample_ids_all.append(sample_ids.numpy())

    preds = np.concatenate(preds_all) if preds_all else np.array([], dtype=np.int64)
    labels = np.concatenate(labels_all) if labels_all else np.array([], dtype=np.int64)
    sample_ids = np.concatenate(sample_ids_all) if sample_ids_all else np.array([], dtype=np.int64)
    matrix = confusion_matrix(labels, preds, num_classes)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "weighted_f1": weighted_f1(labels, preds, num_classes),
        "confusion_matrix": matrix,
        "preds": preds,
        "labels": labels,
        "sample_ids": sample_ids,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def build_model(model_name: str, in_channels: int, num_classes: int, dataset_config: dict) -> nn.Module:
    if model_name == "cnn":
        return SmallCNN(in_channels=in_channels, num_classes=num_classes)
    if model_name == "resnet":
        return SmallResNet(in_channels=in_channels, num_classes=num_classes)
    if model_name == "unet":
        return PlainUNetClassifier(in_channels=in_channels, num_classes=num_classes)
    if model_name == "conv3d":
        return Conv3DBaseline(
            in_channels=in_channels,
            num_classes=num_classes,
            static_channel_count=int(dataset_config.get("static_channel_count", 0)),
            dynamic_channel_count=int(dataset_config.get("dynamic_channel_count", 0)),
            time_steps=int(dataset_config.get("time_steps", 0)),
        )
    if model_name == "convlstm":
        return ConvLSTMBaseline(
            in_channels=in_channels,
            num_classes=num_classes,
            static_channel_count=int(dataset_config.get("static_channel_count", 0)),
            dynamic_channel_count=int(dataset_config.get("dynamic_channel_count", 0)),
            time_steps=int(dataset_config.get("time_steps", 0)),
        )
    if model_name == "swin3d":
        return Swin3DBaseline(
            in_channels=in_channels,
            num_classes=num_classes,
            static_channel_count=int(dataset_config.get("static_channel_count", 0)),
            dynamic_channel_count=int(dataset_config.get("dynamic_channel_count", 0)),
            time_steps=int(dataset_config.get("time_steps", 0)),
        )
    if model_name == "swin3d_yearmonth":
        return YearMonthSwin3DBaseline(
            in_channels=in_channels,
            num_classes=num_classes,
            static_channel_count=int(dataset_config.get("static_channel_count", 0)),
            dynamic_channel_count=int(dataset_config.get("dynamic_channel_count", 0)),
            time_steps=int(dataset_config.get("time_steps", 0)),
            year_steps=int(dataset_config.get("year_steps", 0)),
            month_steps=int(dataset_config.get("month_steps", 0)),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    x_path = os.path.join(args.dataset_dir, "X.npy")
    y_path = os.path.join(args.dataset_dir, "y.npy")
    train_idx_path = os.path.join(args.dataset_dir, "train_idx.npy")
    val_idx_path = os.path.join(args.dataset_dir, "val_idx.npy")
    test_idx_path = os.path.join(args.dataset_dir, "test_idx.npy")
    config_path = os.path.join(args.dataset_dir, "config.json")

    with open(config_path) as f:
        dataset_config = json.load(f)

    drop_modalities = parse_drop_modalities(args.drop_modalities)
    in_channels = int(np.load(x_path, mmap_mode="r").shape[1])
    channel_mask = build_channel_mask(dataset_config, drop_modalities, in_channels)

    mean, std = compute_channel_stats(x_path, train_idx_path, args.stats_samples, args.seed)
    np.save(os.path.join(args.output_dir, "channel_mean.npy"), mean)
    np.save(os.path.join(args.output_dir, "channel_std.npy"), std)
    np.save(os.path.join(args.output_dir, "channel_mask.npy"), channel_mask)

    train_ds = NpyPatchDataset(x_path, y_path, train_idx_path, mean=mean, std=std, channel_mask=channel_mask)
    val_ds = NpyPatchDataset(x_path, y_path, val_idx_path, mean=mean, std=std, channel_mask=channel_mask)
    test_ds = NpyPatchDataset(x_path, y_path, test_idx_path, mean=mean, std=std, channel_mask=channel_mask)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_classes = int(np.load(y_path, mmap_mode="r").max()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, in_channels, num_classes, dataset_config).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = make_class_weights(y_path, train_idx_path, num_classes).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    focal_loss = FocalLoss(class_weights=class_weights, gamma=args.focal_gamma)
    if args.loss == "ce":
        criterion = ce_loss
    elif args.loss == "focal":
        criterion = focal_loss
    elif args.loss == "ordinal":
        criterion = OrdinalCDFLoss()
    elif args.loss == "ce_ordinal":
        criterion = CombinedOrdinalCrossEntropyLoss(ce_loss=ce_loss, ordinal_weight=args.ordinal_weight)
    else:
        criterion = CombinedOrdinalBaseLoss(base_loss=focal_loss, ordinal_weight=args.ordinal_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_metrics["acc"])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_wf1={val_metrics['weighted_f1']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "model_name": args.model,
                    "loss_name": args.loss,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            if not args.disable_early_stopping and patience_counter >= args.early_stop_patience:
                print("Early stopping triggered")
                break

    with open(os.path.join(args.output_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

    with open(os.path.join(args.output_dir, "test_predictions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "y_true", "y_pred"])
        writer.writeheader()
        for sample_id, y_true, y_pred in zip(test_metrics["sample_ids"], test_metrics["labels"], test_metrics["preds"]):
            writer.writerow(
                {"sample_id": int(sample_id), "y_true": int(y_true), "y_pred": int(y_pred)}
            )

    class_f1 = per_class_f1_from_confusion(test_metrics["confusion_matrix"])
    metrics = {
        "model": args.model,
        "drop_modalities": drop_modalities,
        "loss": args.loss,
        "ordinal_weight": args.ordinal_weight,
        "focal_gamma": args.focal_gamma,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_per_class_f1": class_f1,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "in_channels": in_channels,
        "num_classes": num_classes,
        "device": str(device),
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model:", args.model)
    print("Best val acc:", best_val_acc)
    print("Test acc:", test_metrics["acc"])
    print("Test weighted F1:", test_metrics["weighted_f1"])
    print("Saved outputs to", args.output_dir)


if __name__ == "__main__":
    main()
