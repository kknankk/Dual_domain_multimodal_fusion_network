
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
# from .loss import RankingLoss, CosineLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
class CXRModels(nn.Module):

    def __init__(self,):
	
        super(CXRModels, self).__init__()
        # self.args = args

        self.vision_backbone = getattr(torchvision.models, 'resnet34')(pretrained=True) #get attribute
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features #找到输入特征数
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual)) #set attribute
            break
        # self.bce_loss = torch.nn.BCELoss(size_average=True)
        self.classifier = nn.Sequential(nn.Linear(d_visual, 4))
        self.feats_dim = d_visual
       

    def forward(self, x, labels=None, n_crops=0, bs=16):
        # lossvalue_bce = torch.zeros(1)
        # print(f'input cxr size {x.shape}') #[batch_size,channel=3,224,224]
        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)

        # preds = torch.sigmoid(preds)


        return visual_feats,preds


#-----------------------WaveViT----------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

import sys
import os
import argparse


# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath('/home/mimic/MIMIC_subset/MIMIC_subset'))
# print(a)
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()

from utils import DWT_2D, IDWT_2D

def rand_bbox(size, lam, scale=1):
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.kv = nn.Linear(dim, dim * 2)
        self.q = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        return cls_embed

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ClassBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = ClassAttention(dim, num_heads)
        self.mlp = FFN(dim, int(dim * mlp_ratio))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.attn(self.norm1(x))
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.proj = nn.Linear(dim+dim//4, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # print(f'1 shape {x.shape}')#[batch,3136,64]
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = x.float()
        # print(f'2 {x.shape}')#[batch,3136,64]

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
    
        x = x.float()
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)

        x_idwt = self.idwt(x_dwt)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)

        kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, 
        dim, 
        num_heads, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratio=1, 
        block_type = 'wave'
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if block_type == 'std_att':
            self.attn = Attention(dim, num_heads)
        else:
            self.attn = WaveAttention(dim, num_heads, sr_ratio)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class WaveViT(nn.Module):
    def __init__(self, 
        in_chans=3, 
        num_classes=4, 
        stem_hidden_dim = 32,
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14], 
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3], 
        sr_ratios=[4, 2, 1, 1], 
        num_stages=4,
#TODO:本来token_label是true，我改为false，具体影响forward函数
        token_label=True,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                # print(f'embeding i=0') #Y
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            block = nn.ModuleList([Block(
                dim = embed_dims[i], 
                num_heads = num_heads[i], 
                mlp_ratio = mlp_ratios[i], 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer,
                sr_ratio = sr_ratios[i],
                block_type='wave' if i < 2 else 'std_att')
            for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        post_layers = ['ca']
        # print(f'post_layers len(post_layers)')
        self.post_network = nn.ModuleList([
            ClassBlock(
                dim = embed_dims[-1], 
                num_heads = num_heads[-1], 
                mlp_ratio = mlp_ratios[-1],
                norm_layer=norm_layer)
            for _ in range(len(post_layers))
        ])

        # classification head
        self.head = nn.Linear(embed_dims[-1], 4)
        #  if num_classes > 0 else nn.Identity()
        ##################################### token_label #####################################
        self.return_dense = token_label
        self.mix_token = token_label
        self.beta = 1.0
        self.pooling_scale = 8
        if self.return_dense:
            self.aux_head = nn.Linear(
                embed_dims[-1],4)
                # 4) if num_classes > 0 else nn.Identity()
        ##################################### token_label #####################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = x.mean(dim=1, keepdim=True)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(f'post_layers len(post_layers)')
        for block in self.post_network:
            x = block(x)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x)[:, 0]
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x

    def forward(self, x):
        # print(f'model input {x.shape}')
        if not self.return_dense:
            print(f'1')
            x1 = self.forward_features(x)

            x = self.head(x1)
            return x1,x
        else:
            # print(f'use else')
            x, H, W = self.forward_embeddings(x)
            # print(f'after forward_embedding x {x.shape}')
            
            # mix token, see token labeling for details.
            if self.mix_token and self.training:
                # print(f'self.mix_token and self.training')
                
                lam = np.random.beta(self.beta, self.beta)
                patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
                    2] // self.pooling_scale
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
                temp_x = x.clone()
                sbbx1,sbby1,sbbx2,sbby2=self.pooling_scale*bbx1,self.pooling_scale*bby1,\
                                        self.pooling_scale*bbx2,self.pooling_scale*bby2
                temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
                x = temp_x
            else:
                
                bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
            
            # print(f' begin forward_tokens') #Y
            x1 = self.forward_tokens(x, H, W)
            # print(f' after forward_tokens x1 {x1.shape}')
            x_cls = self.head(x1[:, 0])
            # print(f'x_cls {x_cls.shape}')
            #TODO:先令x_aux=1
            # x_aux=1
            x_aux = self.aux_head(
                x1[:, 1:]
            )  # generate classes in all feature tokens, see token labeling

            if not self.training:
                # print(f'run 1')#N
                # return x_cls + 0.5 * x_aux.max(1)[0]
                return x1,x_cls
                # return x1,x_cls
#TODO: 只是mark一下下面的if到return前被注了
            if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
                # print(f'run 2')#Y
                x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])

                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x

                x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])
            # print(f'x_cls, x_aux {x_cls.shape, x_aux.shape}')
            # print(f'return---------------')
            return x1,x_cls
        
    def forward_tokens(self, x, H, W):
        B = x.shape[0]
        x = x.view(B, -1, x.size(-1))

        for i in range(self.num_stages):
            if i != 0:
                # print(f'i {i}')
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                x, H, W = patch_embed(x)
                # print(f'after patch embed {x.shape}')

            block = getattr(self, f"block{i + 1}")
            for blk in block:
                # print(f'blk') #Y
                x = blk(x, H, W)

            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x)
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)    
        return x

    def forward_embeddings(self, x):
        x=x.float()
        patch_embed = getattr(self, f"patch_embed{0 + 1}")
        x, H, W = patch_embed(x)
        x = x.view(x.size(0), H, W, -1)
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

@register_model
def wavevit_s(pretrained=False, **kwargs):
    model = WaveViT(
        stem_hidden_dim = 32,
        embed_dims = [64, 128, 320, 448], 
        num_heads = [2, 4, 10, 14], 
        num_classes=4,
        mlp_ratios = [8, 8, 4, 4],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 4, 6, 3], 
        # depths = [1, 1, 1, 1], 
        sr_ratios = [4, 2, 1, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

# @register_model
# def wavevit_s(pretrained=False, **kwargs):
#     model = WaveViT(
#         stem_hidden_dim = 32,
#         embed_dims = [64, 128], 
#         num_heads = [2, 4], 
#         mlp_ratios = [8, 8],
#         norm_layer = partial(nn.LayerNorm, eps=1e-6), 
#         # depths = [3, 4, 6, 3], 
#         depths = [1, 1], 
#         sr_ratios = [4, 2], 
#         num_stages=2,
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model

import torch
# from torchinfo import summary

# # 创建模型实例
# model = wavevit_s(pretrained=False)
# # model=CXRModels()

# # # 将模型转换为浮点数类型
# # model = model.float()

# # # 假设输入图像的大小是 (batch_size, channels, height, width)
# input_tensor = torch.randn(1, 3, 224, 224).float()  # 确保输入是 Float 类型

# # # 打印模型结构
# summary(model, input_data=input_tensor)


# @register_model
# def wavevit_b(pretrained=False, **kwargs):
#     model = WaveViT(
#         stem_hidden_dim = 64,
#         embed_dims = [64, 128, 320, 512], 
#         num_heads = [2, 4, 10, 16], 
#         mlp_ratios = [8, 8, 4, 4], 
#         norm_layer = partial(nn.LayerNorm, eps=1e-6), 
#         depths = [3, 4, 12, 3], 
#         sr_ratios = [4, 2, 1, 1], 
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model

# @register_model
# def wavevit_l(pretrained=False, **kwargs):
#     model = WaveViT(
#         stem_hidden_dim = 64,
#         embed_dims = [96, 192, 384, 512],
#         num_heads = [3, 6, 12, 16], 
#         mlp_ratios = [8, 8, 4, 4],
#         norm_layer = partial(nn.LayerNorm, eps=1e-6), 
#         depths = [3, 6, 18, 3], 
#         sr_ratios = [4, 2, 1, 1], 
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model



# CXRModels                                     [1, 7]                    --
# ├─ResNet: 1-1                                 [1, 512]                  --
# │    └─Conv2d: 2-1                            [1, 64, 112, 112]         9,408
# │    └─BatchNorm2d: 2-2                       [1, 64, 112, 112]         128
# │    └─ReLU: 2-3                              [1, 64, 112, 112]         --
# │    └─MaxPool2d: 2-4                         [1, 64, 56, 56]           --
# │    └─Sequential: 2-5                        [1, 64, 56, 56]           --
# │    │    └─BasicBlock: 3-1                   [1, 64, 56, 56]           73,984
# │    │    └─BasicBlock: 3-2                   [1, 64, 56, 56]           73,984
# │    │    └─BasicBlock: 3-3                   [1, 64, 56, 56]           73,984
# │    └─Sequential: 2-6                        [1, 128, 28, 28]          --
# │    │    └─BasicBlock: 3-4                   [1, 128, 28, 28]          230,144
# │    │    └─BasicBlock: 3-5                   [1, 128, 28, 28]          295,424
# │    │    └─BasicBlock: 3-6                   [1, 128, 28, 28]          295,424
# │    │    └─BasicBlock: 3-7                   [1, 128, 28, 28]          295,424
# │    └─Sequential: 2-7                        [1, 256, 14, 14]          --
# │    │    └─BasicBlock: 3-8                   [1, 256, 14, 14]          919,040
# │    │    └─BasicBlock: 3-9                   [1, 256, 14, 14]          1,180,672
# │    │    └─BasicBlock: 3-10                  [1, 256, 14, 14]          1,180,672
# │    │    └─BasicBlock: 3-11                  [1, 256, 14, 14]          1,180,672
# │    │    └─BasicBlock: 3-12                  [1, 256, 14, 14]          1,180,672
# │    │    └─BasicBlock: 3-13                  [1, 256, 14, 14]          1,180,672
# │    └─Sequential: 2-8                        [1, 512, 7, 7]            --
# │    │    └─BasicBlock: 3-14                  [1, 512, 7, 7]            3,673,088
# │    │    └─BasicBlock: 3-15                  [1, 512, 7, 7]            4,720,640
# │    │    └─BasicBlock: 3-16                  [1, 512, 7, 7]            4,720,640
# │    └─AdaptiveAvgPool2d: 2-9                 [1, 512, 1, 1]            --
# │    └─Identity: 2-10                         [1, 512]                  --
# ├─Sequential: 1-2                             [1, 7]                    --
# │    └─Linear: 2-11                           [1, 7]                    3,591


#=============================GFNet===================
