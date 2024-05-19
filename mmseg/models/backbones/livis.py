"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import numpy as np
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from mmseg.registry import MODELS


class ConvBN(BaseModule):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        if with_bn:
            # self.norm = torch.nn.BatchNorm2d(out_planes)
            self.norm = build_norm_layer(norm_cfg, out_planes)[1]

    def forward(self, x):
        x = self.conv(x)
        return self.norm(x) if hasattr(self, "norm") else x


class Block(BaseModule):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=0.,
                 hidden_len=49,  act_layer=nn.GELU, mlp_ratio=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.local_conv = ConvBN(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,with_bn=True, norm_cfg=norm_cfg)
        self.act=act_layer()
        self.hidden_len = hidden_len
        self.mlp_ratio = mlp_ratio
        self.q1 = ConvBN(dim, dim, with_bn=True, norm_cfg=norm_cfg)
        self.q2 = ConvBN(dim, hidden_len, with_bn=True, norm_cfg=norm_cfg)
        self.k = ConvBN(dim, dim, with_bn=True, norm_cfg=norm_cfg)
        self.v = ConvBN(dim, int(self.mlp_ratio * dim), with_bn=True, norm_cfg=norm_cfg)
        # self.qkv = ConvBN(dim, (dim + hidden_len + dim + int(self.mlp_ratio * dim)), with_bn=True)
        self.down_k=ConvBN(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, with_bn=True, norm_cfg=norm_cfg)
        self.mlp_ratio=mlp_ratio

        self.qk_mlp = ConvBN(dim, int(self.mlp_ratio * dim), with_bn=True, norm_cfg=norm_cfg)
        self.mlp = ConvBN(int(self.mlp_ratio * dim), dim, with_bn=True, norm_cfg=norm_cfg)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1,dim,1,1)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.act(self.local_conv(x))
        B,C,H,W = x.shape
        HW = H*W
        q1,q2,k,v = self.act(self.q1(x)), self.act(self.q2(x)), self.k(x), self.act(self.v(x))
        k = self.down_k(k)
        # _,_,H_k,W_k = k.shape
        q = (q1.reshape(B,C,HW))@(q2.reshape(B,self.hidden_len,HW).transpose(-2,-1)) # [b,c,d]
        q = q / HW
        # # new implementation
        # qk = q.mean(dim=-1).view(B,C,1,1) * k # [b,c,h'*w']
        qk = q.mean(dim=-1, keepdim=True).unsqueeze(dim=-1) * k # [b,c,h'*w'] # slightly faster ~0.05 ms
        qk = self.act(self.qk_mlp(qk))
        qk = F.interpolate(qk, [H, W])
        x = qk * v
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + self.drop_path(x)
        return x


@MODELS.register_module()
class LiViS(BaseModule):
    def __init__(self, in_chans=3,
                 act_layer=nn.ReLU, hidden_len=49, mlp_ratio=6,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1,
                 layer_scale_init_value=1e-6,
                 frozen_stages=-1,
                 out_indices=(0, 1, 2, 3),
                 norm_eval=False,
                 pretrained=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 init_cfg=None,
                 **kwargs
                 ):
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.norm_eval = norm_eval
        self.out_indices = out_indices
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            ConvBN(in_chans, 64, kernel_size=5, stride=2, padding=2, norm_cfg=norm_cfg),
            act_layer(),
            ConvBN(64, dims[0], kernel_size=5, stride=2, padding=2, norm_cfg=norm_cfg),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = ConvBN(dims[i], dims[i+1], kernel_size=5, stride=2, padding=2, norm_cfg=norm_cfg)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i],
                act_layer=act_layer, hidden_len=hidden_len,
                mlp_ratio=mlp_ratio,
                drop_path=dp_rates[cur + j],
                norm_cfg=norm_cfg,
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]





    #     self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
    #     self.head = nn.Linear(dims[-1], num_classes)
    #
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         nn.init.constant_(m.bias, 0)


    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        # self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


# @register_model
# def livis_c(**kwargs):  # 2.454 ms
#     model = LiViS(
#         depths=[2, 2, 10, 2], dims=[32, 64, 256, 480], mlp_ratio=5, act_layer=nn.ReLU, hidden_len=16,
#         **kwargs)
#     return model
#
#
