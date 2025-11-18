# Copyright (c) Markus Hiller and Rongkai Ma -- 2022
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Model code essentially copy-paste from the iBOT repository: https://github.com/bytedance/ibot/models,
which is in turn heavily based on DINO and the timm library:
https://github.com/facebookresearch/dino
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
#
#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from utils import trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import to_2tuple


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Window partition utilities
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# Channel Attention Block (CAB)
class ChannelAttentionBlock(nn.Module):
    """Channel Attention Block inspired by SENet with Conv 3x3"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        
        # Global Average Pooling and Global Max Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP with reduction
        hidden_dim = max(dim // reduction, 1)
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Conv 3x3 for local context
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) or (B, H, W, C)
        Returns:
            out: (B, C, H, W) or (B, H, W, C) - same format as input
        """
        # Handle both (B, C, H, W) and (B, H, W, C) formats
        if x.dim() == 4 and x.shape[-1] == self.dim:
            # (B, H, W, C) format
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            need_permute = True
        else:
            need_permute = False
            
        B, C, H, W = x.shape
        
        # Global Average Pooling and Global Max Pooling
        gap_out = self.gap(x)  # (B, C, 1, 1)
        gmp_out = self.gmp(x)  # (B, C, 1, 1)
        
        # Shared MLP
        gap_out = self.fc2(self.act(self.fc1(gap_out)))
        gmp_out = self.fc2(self.act(self.fc1(gmp_out)))
        
        # Combine GAP and GMP
        channel_attn = self.sigmoid(gap_out + gmp_out)  # (B, C, 1, 1)
        
        # Apply channel attention
        x_attn = x * channel_attn
        
        # Apply Conv 3x3 for local context
        x_conv = self.conv3x3(x_attn)
        
        # Residual connection
        out = x_attn + x_conv
        
        if need_permute:
            out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
            
        return out


# Window-based Multi-Head Self-Attention (W-MSA)
class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size) if isinstance(window_size, int) else window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Attention(nn.Module):
    """Standard self-attention (kept for backward compatibility)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # bs, num_tokens, channels (current dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # shape [N, n_h, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attention value product
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


# Mixed Attention Block (MAB) - W-MSA + CAB
class MixedAttentionBlock(nn.Module):
    """Mixed Attention Block combining Window-based Multi-Head Self-Attention and Channel Attention"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cab_reduction=16):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # Channel Attention Block
        self.cab = ChannelAttentionBlock(dim, reduction=cab_reduction)
        self.cab_norm = norm_layer(dim)
        
        self.H = input_resolution[0]
        self.W = input_resolution[1]
        
    def create_attn_mask(self, H, W):
        """Create attention mask for SW-MSA"""
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def forward(self, x):
        """
        Args:
            x: (B, H*W, C) or (B, H*W+1, C) if includes CLS token
        Returns:
            x: (B, H*W, C) or (B, H*W+1, C) - same shape as input
        """
        B, L, C = x.shape
        H = self.H
        W = self.W
        
        # Handle CLS token if present
        has_cls_token = (L == H * W + 1)
        if has_cls_token:
            cls_token = x[:, 0:1]  # (B, 1, C)
            x = x[:, 1:]  # (B, H*W, C)
        elif L != H * W:
            raise ValueError(f"Input sequence length {L} does not match expected H*W = {H}*{W} = {H*W}")
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.create_attn_mask(H, W).to(x.device)
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA
        attn_windows, _ = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # Channel Attention Block
        x_cab = x.view(B, H, W, C)
        x_cab = self.cab(x_cab)
        x_cab = x_cab.view(B, H * W, C)
        x_cab = self.cab_norm(x_cab)
        x = x + self.drop_path(x_cab)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Add CLS token back if it was present
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=1)
        
        return x


# Overlapping Cross-Attention Block (OCAB)
class OverlappingCrossAttentionBlock(nn.Module):
    """Overlapping Cross-Attention Block with Q from non-overlapping windows and K/V from overlapping windows"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, overlap_ratio=0.5,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        
        # Overlapping window size
        self.overlap_window_size = int(window_size * (1 + overlap_ratio))
        
        self.norm1 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        
        # Q projection for non-overlapping windows
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # K, V projections for overlapping windows
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.H = input_resolution[0]
        self.W = input_resolution[1]
        
    def forward(self, x):
        """
        Args:
            x: (B, H*W, C) or (B, H*W+1, C) if includes CLS token
        Returns:
            x: (B, H*W, C) or (B, H*W+1, C) - same shape as input
        """
        B, L, C = x.shape
        H = self.H
        W = self.W
        
        # Handle CLS token if present
        has_cls_token = (L == H * W + 1)
        if has_cls_token:
            cls_token = x[:, 0:1]  # (B, 1, C)
            x = x[:, 1:]  # (B, H*W, C)
        elif L != H * W:
            raise ValueError(f"Input sequence length {L} does not match expected H*W = {H}*{W} = {H*W}")
        
        shortcut = x
        x = self.norm1(x)
        x_kv = self.norm_kv(x)
        
        x = x.view(B, H, W, C)
        x_kv = x_kv.view(B, H, W, C)
        
        # Pad for overlapping windows
        pad_size = (self.overlap_window_size - self.window_size) // 2
        x_kv_padded = F.pad(x_kv, (0, 0, pad_size, pad_size, pad_size, pad_size))
        
        # Partition into non-overlapping windows for Q
        q_windows = window_partition(x, self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, C)
        
        # Partition into overlapping windows for K, V
        kv_windows = window_partition(x_kv_padded, self.overlap_window_size)
        kv_windows = kv_windows.view(-1, self.overlap_window_size * self.overlap_window_size, C)
        
        # Project Q, K, V
        q = self.q(q_windows).reshape(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv_windows).reshape(-1, self.overlap_window_size * self.overlap_window_size, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Merge windows back
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)
        x = x.view(B, H * W, C)
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Add CLS token back if it was present
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=1)
        
        return x


# Multi-Scale Channel Attention Module (MS-CAM)
class MultiScaleChannelAttention(nn.Module):
    """Multi-Scale Channel Attention Module with global and local branches"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dim = dim
        hidden_dim = max(dim // reduction, 1)
        
        # Global attention branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )
        
        # Local attention branch
        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) or (B, H, W, C)
        Returns:
            out: same format as input
        """
        # Handle both formats
        if x.dim() == 4 and x.shape[-1] == self.dim:
            x = x.permute(0, 3, 1, 2)
            need_permute = True
        else:
            need_permute = False
        
        B, C, H, W = x.shape
        
        # Global branch
        global_attn = self.global_branch(x)  # (B, C, 1, 1)
        
        # Local branch
        local_attn = self.local_branch(x)  # (B, C, H, W)
        
        # Fuse
        fused = torch.cat([global_attn.expand_as(x), local_attn], dim=1)
        channel_weights = self.fusion(fused)  # (B, C, H, W)
        
        # Apply attention
        out = x * channel_weights
        
        if need_permute:
            out = out.permute(0, 2, 3, 1)
        
        return out


# Attention Feature Fusion Module (AFFM)
class AttentionFeatureFusionModule(nn.Module):
    """Attention Feature Fusion Module to fuse multi-scale features"""
    def __init__(self, dim, num_scales=3):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # Attention weights for each scale
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * num_scales, dim * num_scales, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * num_scales, num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(dim * num_scales, dim, kernel_size=1)
        
    def forward(self, features_list):
        """
        Args:
            features_list: list of (B, C, H, W) tensors at different scales
        Returns:
            fused: (B, C, H, W)
        """
        # Resize all features to the same size (use the smallest)
        min_h = min(f.shape[2] for f in features_list)
        min_w = min(f.shape[3] for f in features_list)
        
        resized_features = []
        for f in features_list:
            if f.shape[2] != min_h or f.shape[3] != min_w:
                f = F.interpolate(f, size=(min_h, min_w), mode='bilinear', align_corners=False)
            resized_features.append(f)
        
        # Concatenate
        concat_features = torch.cat(resized_features, dim=1)  # (B, C*num_scales, H, W)
        
        # Compute attention weights
        attn_weights = self.scale_attention(concat_features)  # (B, num_scales, 1, 1)
        
        # Weighted sum
        weighted_features = []
        for i, f in enumerate(resized_features):
            weighted_features.append(f * attn_weights[:, i:i+1, :, :])
        
        weighted_concat = torch.cat(weighted_features, dim=1)
        
        # Fusion
        fused = self.fusion_conv(weighted_concat)
        
        return fused


# Dual Reconstruction Feature Fusion (DRFF)
class DualReconstructionFeatureFusion(nn.Module):
    """Dual Reconstruction Feature Fusion module combining AFFM and MS-CAM"""
    def __init__(self, dim, num_scales=3):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # Multi-scale feature extraction (using different kernel sizes)
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=2*i+1, padding=i, groups=dim) 
            for i in range(1, num_scales + 1)
        ])
        
        # AFFM
        self.affm = AttentionFeatureFusionModule(dim, num_scales=num_scales)
        
        # MS-CAM
        self.mscam = MultiScaleChannelAttention(dim)
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, dim),  # Channel-wise normalization that supports arbitrary spatial size
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            refined: (B, C, H, W) refined feature map
        """
        # Extract multi-scale features
        multi_scale_features = []
        for conv in self.scale_convs:
            multi_scale_features.append(conv(x))
        
        # Fuse with AFFM
        fused = self.affm(multi_scale_features)
        
        # Apply MS-CAM
        attended = self.mscam(fused)
        
        # Refinement
        refined = self.refine(attended)
        
        # Residual connection
        out = x + refined
        
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0,
                 use_mab=False, use_ocab=False, input_resolution=None, window_size=7, shift_size=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # Choose attention mechanism
        if use_mab:
            assert input_resolution is not None, "input_resolution required for MAB"
            self.attn = MixedAttentionBlock(
                dim, input_resolution, num_heads, window_size=window_size, shift_size=shift_size,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer)
        elif use_ocab:
            assert input_resolution is not None, "input_resolution required for OCAB"
            self.attn = OverlappingCrossAttentionBlock(
                dim, input_resolution, num_heads, window_size=window_size,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer)
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_mab_or_ocab = use_mab or use_ocab

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False):
        if self.use_mab_or_ocab:
            # MAB and OCAB handle normalization internally
            x = self.attn(x)
        else:
            y, attn = self.attn(self.norm1(x))
            if return_attention:
                return attn
            if self.gamma_1 is None:
                x = x + self.drop_path(y)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * y)
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x)


class VisionTransformer(nn.Module):
    """ Vision Transformer with Multi-Attention Architecture """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=False,
                 init_values=0, use_mean_pooling=False, masked_im_modeling=False,
                 use_mab=False, use_ocab=False, window_size=7, use_drff=False, drff_num_scales=3):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens
        self.use_drff = use_drff
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.img_size = img_size[0]
        self.patches_resolution = (img_size[0] // patch_size, img_size[0] // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # Create blocks with MAB/OCAB if specified
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2 if use_mab else 0
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, use_mab=use_mab, use_ocab=use_ocab,
                input_resolution=self.patches_resolution, window_size=window_size, shift_size=shift_size)
            self.blocks.append(block)

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        
        # DRFF layer after Transformer Encoder
        if use_drff:
            self.drff = DualReconstructionFeatureFusion(embed_dim, num_scales=drff_num_scales)
        else:
            self.drff = None
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_tokens=None, mask=None, return_features=False):
        # mim
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)

        # Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply DRFF if enabled
        if self.use_drff and self.drff is not None:
            # Extract patch features (excluding cls token)
            B, N, C = x.shape
            H = W = int(math.sqrt(N - 1))  # Exclude cls token
            
            # Separate cls token and patch tokens
            cls_token = x[:, 0:1, :]  # (B, 1, C)
            patch_tokens = x[:, 1:, :]  # (B, N-1, C)
            
            # Reshape to (B, C, H, W) for DRFF
            patch_features = patch_tokens.view(B, H, W, C).permute(0, 3, 1, 2)
            
            # Apply DRFF
            refined_features = self.drff(patch_features)  # (B, C, H, W)
            
            # Reshape back to (B, H*W, C)
            refined_features = refined_features.permute(0, 2, 3, 1).view(B, H * W, C)
            
            # Concatenate with cls token
            x = torch.cat([cls_token, refined_features], dim=1)

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        
        if return_features:
            # Return features for reconstruction
            return x
        
        if return_all_tokens:
            return x
        return x[:, 0]

    def get_num_layers(self):
        return len(self.blocks)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model
