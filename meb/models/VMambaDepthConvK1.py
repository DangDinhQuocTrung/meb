import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from meb.models.k1_modules import SS2DepthConv_K1, SS2DConv_K1, SS2Depth_K1
from meb.models.original_modules import SS2D


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    Reference: http://arxiv.org/abs/2401.10166
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # B, C, H, W ==> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)
        x = x.reshape(B, H*2, W*2, C//4)

        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        use_depth: bool = False,
        use_conv: bool = False,
        mode: str = "",
        d_depth_stride: int = 16,
        d_depth_out: int = 1,
        d_depth_squeeze: int = 1,
        conv_mode: str = "",
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        if use_depth and use_conv:
            self.self_attention = SS2DepthConv_K1(
                d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,
                mode=mode,
                d_depth_stride=d_depth_stride,
                d_depth_out=d_depth_out,
                d_depth_squeeze=d_depth_squeeze,
                conv_mode=conv_mode,
                **kwargs)
        elif use_conv:
            self.self_attention = SS2DConv_K1(
                d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,
                mode=mode,
                d_depth_stride=d_depth_stride,
                d_depth_out=d_depth_out,
                d_depth_squeeze=d_depth_squeeze,
                conv_mode=conv_mode,
                **kwargs)
        elif use_depth:
            self.self_attention = SS2Depth_K1(
                d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,
                mode=mode,
                d_depth_stride=d_depth_stride,
                d_depth_out=d_depth_out,
                d_depth_squeeze=d_depth_squeeze,
                conv_mode=conv_mode,
                **kwargs)
        else:
            self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        use_depth=False,
        use_conv=False,
        mode="",
        d_depth_stride=16,
        d_depth_out=1,
        d_depth_squeeze=1,
        conv_mode="",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_depth=use_depth,
                use_conv=use_conv,
                mode=mode,
                d_depth_stride=d_depth_stride,
                d_depth_out=d_depth_out,
                d_depth_squeeze=d_depth_squeeze,
                conv_mode=conv_mode,
            )
            for i in range(depth)])

        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSMEncoder(nn.Module):
    def __init__(
        self, patch_size=4, in_chans=3,
        depths=[1, 1],
        dims=[32, 64],
        d_state=16,
        drop_rate=0.5, attn_drop_rate=0.5, drop_path_rate=0.5,
        norm_layer=nn.LayerNorm, patch_norm=True,
        use_checkpoint=False,
        num_classes=1,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        use_depth = True
        use_conv = True
        mode = "even"
        conv_mode = "full"
        depth_stride_configs = [2, 2]
        depth_stride_output_sizes = [64, 16]
        depth_squeeze_factors = [2, 2]

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                use_depth=use_depth,
                use_conv=use_conv,
                mode=mode,
                d_depth_stride=depth_stride_configs[i_layer],
                d_depth_out=depth_stride_output_sizes[i_layer],
                d_depth_squeeze=depth_squeeze_factors[i_layer],
                conv_mode=conv_mode,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.fc1 = nn.Linear(8 * 8 * dims[-1], 256)
        self.drop3 = nn.Dropout(drop_rate)
        self.fc = nn.Linear(256, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        out = self.fc(self.drop3(x))
        return out


class ZZZNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        h_dims: List[int] = [32, 64, 256],
        dropout: float = 0.5,
        softmax=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        h1 = h_dims[0]
        h2 = h_dims[1]
        h3 = h_dims[2]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=h1, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm2d(h1)
        self.drop1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(in_channels=h1, out_channels=h2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(h2)
        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = VSSBlock(
            hidden_dim=h2,
            drop_path=dropout,
            attn_drop_rate=dropout,
            use_depth=True,
            use_conv=True,
            mode="even",
            conv_mode="full",
            d_depth_stride=2,
            d_depth_out=81,
            d_depth_squeeze=4,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(dropout)

        self.fc1 = nn.Linear(5184, h3)
        self.drop3 = nn.Dropout(dropout)
        self.fc = nn.Linear(h3, num_classes)
        self.softmax = None
        if softmax:
            self.softmax = nn.Softmax(dim=1)
        return

    def forward(self, x):
        x = self.drop1(self.bn1(self.pool(F.relu(self.conv1(x)))))
        x = self.drop2(self.bn2(F.relu(self.conv2(x))))

        x = x.permute(0, 2, 3, 1)
        x = self.drop3(self.pool3(self.conv3(x)))

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc(self.drop3(x))
        if self.softmax:
            x = self.softmax(x)
        return x
