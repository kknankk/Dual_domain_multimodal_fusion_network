import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from scipy.signal import stft, istft
import torchaudio
import sys
import os
sys.path.append(os.path.abspath('/home/ke/MIMIC_subset/MIMIC_subset'))

from model.fusion_model import ImagePatchEmbed,FeedForward,AddNorm,Image2TextGate,Text2ImageGate,ImageFrequencySelection,TextFrequencySelection,FtLayer,FtBlock
from model.ViT_b16 import VisionTransformer as vit
from model.ViT_b16 import CONFIGS
import model.configs as configs
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
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()



import torch.nn as nn
import numpy as np

#========senet===============
from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #随机生成一个4维数据先
        # x = torch.randn(7, 256, 7, 7)
        b, c ,_= x.size()
        # b, c = x.size()
        # y = self.avg_pool(x).view(b, c)

        # print(f'x {x.shape}')#[bs,128,4112]
        y=self.avg_pool(x)
        # print(f'after avg y {y.shape}')#[bs,128,1]
        y=y.permute(0,2,1)
        y = self.fc(y).view(b, c, 1)
        # print(f'y {y.shape}')
        # x=x.permute(0,2,1)
        return x * y.expand_as(x)
#self.se = SELayer(planes, reduction)
#===================senet==============

#================resnet1d===================
def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
        # if n_filters_out!=12:
            # print(f'12 != n_filters_out {n_filters_out}')
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        # print(f'Input x is on device: {x.device}, Input y is on device: {y.device}')

        if self.skip_connection is not None:
            # print(f'start skip_connection')
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        # print(f'start blk first conv1 {x.shape}')
        # print(f'conv1 weights are on device: {self.conv1.weight.device}')
        # print(f'conv2 weights are on device: {self.conv2.weight.device}')

        x = self.conv1(x)
        # print(f'after blk first conv1 {x.shape}')
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        # print(f'after blk dropout1 x size {x.shape}')

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class ResNet1d(nn.Module):
    def __init__(self,  n_classes=4, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__()
        # First layers
        self.blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh))

        n_filters_in, n_filters_out = 12, self.blocks_dim[0][0]
        n_samples_in, n_samples_out = 4096, self.blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        # self.blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh))

        # Residual block layers
        # self.res_blocks = []
        self.res_blocks = nn.ModuleList()
        for i, (n_filters, n_samples) in enumerate(self.blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            
            # print(f'{i} th downsample {downsample}' )
            # print(f'i th n_filters_out {n_filters_out}')
            resblk1d = ResBlock1d( n_filters_in,n_filters_out, downsample, kernel_size, dropout_rate)
            # self.add_module('resblock1d_{0}'.format(i), resblk1d)
            # self.res_blocks += [resblk1d]
            self.res_blocks.append(resblk1d)

        n_filters_last, n_samples_last = self.blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, 512)

        self.lin1 = nn.Linear(512, 128)
        self.n_blk = len(self.blocks_dim)
        self.lin128 = nn.Linear(320, 128)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        # print(f' 1 input size {x.shape}')#[bs,12,4096]
        x = self.conv1(x)
        # print(f' 2 input size {x.shape}')
        x = self.bn1(x)
        # print(f' 3 input size {x.shape}')

        # Residual blocks
        y = x
        for i,blk in enumerate(self.res_blocks):
            # print(f' {i}th input shape {x.shape}')
            x, y = blk(x, y)
            # print(f' {i}th output x shape {x.shape}')
            # print(f' {i}th output y shape {y.shape}')

        # Flatten array
        # print(f'before flatten {x.shape}')#[bs,320,16]
        x=x.permute(0,2,1)
        x=self.lin128(x)
        # print(f'x {x.shape}')#[bs,16,128]
        x = x.view(-1, 128, 16)  # 将输出调整回 [bs, 128, 16]
        # print(f'after view x {x.shape}')
        # x1 = x.view(x.size(0), -1)

        # # Fully conected layer
        # x2 = self.lin(x1)
        # # print(f'x2 {x2.shape}')#[bs,512]
        # x=self.lin1(x2)
        # # print(f'x {x.shape}')#[bs,128]
        return x
#================resnet1d===================




#==========resnet34=============
# class CXRModels(nn.Module):

#     def __init__(self,):
	
#         super(CXRModels, self).__init__()
#         # self.args = args

#         self.vision_backbone = getattr(torchvision.models, 'resnet34')(pretrained=True) #get attribute
#         classifiers = [ 'classifier', 'fc']
#         for classifier in classifiers:
#             cls_layer = getattr(self.vision_backbone, classifier, None)
#             if cls_layer is None:
#                 continue
#             d_visual = cls_layer.in_features #找到输入特征数
#             setattr(self.vision_backbone, classifier, nn.Identity(d_visual)) #set attribute
#             break
#         # self.bce_loss = torch.nn.BCELoss(size_average=True)
#         self.classifier = nn.Sequential(nn.Linear(d_visual, 128))
#         self.feats_dim = d_visual
        
       

#     def forward(self, x, labels=None, n_crops=0, bs=16):
#         # lossvalue_bce = torch.zeros(1)
#         # print(f'input cxr size {x.shape}') #[batch_size,channel=3,224,224]
#         visual_feats = self.vision_backbone(x)
#         # print(f'visual_feats {visual_feats.shape}')
#         preds = self.classifier(visual_feats)
#         # print(f'preds {preds.shape}')#[5,128]

#         # preds = torch.sigmoid(preds)


#         return visual_feats,preds



class CXRModels(nn.Module):
    def __init__(self):
        super(CXRModels, self).__init__()

        self.vision_backbone = torchvision.models.resnet34(pretrained=True)
        classifiers = ['classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity())
            break
            
        self.classifier = nn.Sequential(nn.Linear(d_visual, 128))
        self.feats_dim = d_visual

    def forward(self, x):
        # 获取 backbone 的输出
        visual_feats = self.vision_backbone.conv1(x)
        visual_feats = self.vision_backbone.bn1(visual_feats)
        visual_feats = self.vision_backbone.relu(visual_feats)
        visual_feats = self.vision_backbone.maxpool(visual_feats)

        visual_feats = self.vision_backbone.layer1(visual_feats)
        visual_feats = self.vision_backbone.layer2(visual_feats)
        visual_feats = self.vision_backbone.layer3(visual_feats)
        visual_feats = self.vision_backbone.layer4(visual_feats)

        # 在这里可以选择是否添加 AdaptiveAvgPool2d
        # visual_feats = self.vision_backbone.avgpool(visual_feats)

        # 如果你希望使用 avgpool，取消上面的注释，并将下面的返回修改为:
        # visual_feats = visual_feats.view(visual_feats.size(0), -1)  # 展平
        # preds = self.classifier(visual_feats)

        # preds = self.classifier(visual_feats.view(visual_feats.size(0), -1))  # 展平
        return visual_feats

#==========resnet34=============




#===========FRSU===========================
class Fusion(nn.Module):
    def __init__(self, d_model, act_layer=torch.tanh):
        super(Fusion, self).__init__()

        self.text_weight = nn.Parameter(torch.randn(128, 128, dtype=torch.float32))
        self.image_weight = nn.Parameter(torch.randn(128, 128, dtype=torch.float32))
        self.fusion_weight = nn.Parameter(torch.randn(128, 128, dtype=torch.float32))
        self.act_layer = act_layer

    def forward(self, text, image):
        alpha = self.js_div(text, image)
        a=torch.matmul(text, self.text_weight)
        b=torch.matmul(image, self.image_weight)
        # print(f'before fusion text {text.shape}')#[bs,128]
        # print(f'before fusion image {image.shape}')#[bs,128]
        # print(f'before fusion self.text_weight {self.text_weight.shape}')#[128,128]
        # print(f'before fusion self.image_weight {self.image_weight.shape}')#[128,128]

        # print(f'in fusion part ecg {a.shape}; cxr {b.shape}')#[batch,128]; [batch,128]
        fusion = torch.matmul(text, self.text_weight) + torch.matmul(image, self.image_weight)
        f = (1-alpha) * fusion + alpha * text + alpha * image

        return f

    @staticmethod
    def js_div(p, q):
        """
        Function that measures JS divergence between target and output logits:
        """
        M = (p + q) / 2
        kl1 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        gamma = 0.5 * kl1 + 0.5 * kl2
        return gamma

class MLP(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outputs_dim, num_class, act_layer=nn.ReLU, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_dim, outputs_dim)
        self.norm2 = nn.LayerNorm(outputs_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(outputs_dim, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act_layer(x)
        x = self.fc3(x)
        return x


class FSRU(nn.Module):
    def __init__(self,  d_text=12, seq_len=4369, img_size=224, patch_size=16, d_model=128,
                 num_filter=2, num_class=4, num_layer=1, dropout=0., mlp_ratio=4.):
        super(FSRU, self).__init__()

        # Text

        self.text_encoder = nn.Sequential(nn.Conv1d(in_channels=12, out_channels=d_model, kernel_size=17, stride=1, padding=8),  # 卷积层

            # nn.Linear(d_text, d_model),
                                        #   nn.LayerNorm(d_model),
                                          )
        # s = seq_len // 2 + 1
#=====未对ecg做rfft,所以特征维度不会变为1/2
        self.ecg_norm=nn.LayerNorm(d_model)
        s=seq_len


        # Image
        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (img_size // patch_size) * (img_size // patch_size)
        n = img_len // 2 + 1

        self.FourierTransormer = FtBlock(d_model, s, n, num_layer, num_filter, dropout)

        self.fusion = Fusion(d_model)

        self.mlp = MLP(d_model, int(mlp_ratio*d_model), d_model, num_class, dropout=dropout)

        trunc_normal_(self.img_pos_embed, std=.02)
        self.apply(self._init_weights)

        self.resnet1d=ResNet1d()
        config = CONFIGS['ViT-B_16']
        self.vit=vit(config)
        # self.vit.load_from(np.load('/home/mimic/MIMIC_subset/MIMIC_subset/imagenet21k_ViT-B_16.npz'))
        original_weights = np.load('/home/mimic/MIMIC_subset/MIMIC_subset/imagenet21k_ViT-B_16.npz')

        # 创建一个新的字典，剔除与 self.head 相关的权重
        filtered_weights = {key: value for key, value in original_weights.items() if 'head' not in key}

        # 加载过滤后的权重
        self.vit.load_from(filtered_weights)
        self.layernorm = nn.LayerNorm(128)
        self.act_layer = nn.ReLU()
        self.cxrmodel=CXRModels()
        self.se=SELayer(128,16)
        self.cxrlin=nn.Linear(512, 128)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
            # trunc_normal_(m.weight, std=.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, text, image):
        # text = text.long()
        # text = self.text_embed(text)  # (batch, seq, dim)
        # print(f'input ecg {text.shape}')#[Batch,4096,12]
        # print(f'image {image.shape}')
        # print(f' FUSU input ecg {text.dtype}')
        print(f'FUSU INPUT 0 ECG {text.shape}')
        text=text.permute(0,2,1)
        # print(f'input ecg {text.shape}')#[bs,12,4096]
#加入了resnet1d提取temporal ecg特征
        ecg_temporal=self.resnet1d(text)
        # print(f'ecg_temporal {ecg_temporal.shape}')#[bs,128,16]
        text = self.text_encoder(text)
        # print(f'after encoder ecg {text.shape}')
        text=text.permute(0,2,1)
        text=self.ecg_norm(text)

        image = image.to(torch.float32)
#加入了vit提取spati cxr
        # cxr_spatial,_ = self.vit(image)
        cxr_spatial=self.cxrmodel(image)

        image = self.img_patch_embed(image)
        # print(f'after iamge embedding {image.shape}')
        image = image + self.img_pos_embed
        image = self.img_pos_drop(image)

        text, image = self.FourierTransormer(text, image)
        # print(f'after FourierTransormer ecg {text.shape}')#[bs,4096,128]
        # print(f'after FourierTransormer cxr {image.shape}')#[bs,196,128]
        image=image.permute(0,2,1)
        # print(f'1 cxr {image.shape}')#[bs,128,196]
        # text = torch.max(text, dim=1)[0]
        # image = torch.max(image, dim=1)[0]
#对所有4个input做layer norm后相加
        # image=self.act_layer(image)
        # text=self.act_layer(text)
        # ecg_temporal=self.act_layer(ecg_temporal)
        # cxr_spatial=self.act_layer(cxr_spatial)

        # print(f'cxr_spatial {cxr_spatial.shape}')#[bs,512,7,7]
        bs,c,h,w=cxr_spatial.shape
        cxr_spatial=cxr_spatial.view(bs,c,h*w)
        # print(f'cxr_spatial {cxr_spatial.shape}')#[bs,512,49]
        cxr_spatial=cxr_spatial.permute(0,2,1)
        cxr_spatial=self.cxrlin(cxr_spatial)#[bs,49,128]
        
        cxr_spatial=cxr_spatial.permute(0,2,1)
        # print(f'2 cxr_spatial {cxr_spatial.shape}')#[bs,128,49]
        image=self.act_layer(image)
        text=self.act_layer(text)
        ecg_temporal=self.act_layer(ecg_temporal)
        cxr_spatial=self.act_layer(cxr_spatial)

        text=text.permute(0,2,1)
        # print(f'frequency ecg {text.shape}')#[8,128,4096]
        # print(f'temporal ecg {ecg_temporal.shape}')#[bs,128,16]
#将frequency domain和spatial domain拼接，之后送入senet
        text=torch.cat([text,ecg_temporal],dim=2)
        # print(f'after cat text {text.shape}')#[bs,128,4112]
        text=self.se(text)
        # print(f'after se ecg {text.shape}')#[bs,128,4112]
        image=torch.cat([image,cxr_spatial],dim=2)
        # print(f'after concate cxr {image.shape}')#[bs,128,245]
        image=self.se(image)
        # print(f'after se cxr {image.shape}')#[[bs,128,245]]
        


        text = torch.max(text, dim=2)[0]
        image = torch.max(image, dim=2)[0]
        # print(f'after max  ecg {text.shape}')#[bsm128]
        # print(f'after max  cxr {image.shape}')#[bsm128]
        f = self.fusion(text, image)  # (batch, d_model)

        outputs = self.mlp(f)

        return text, image, outputs, f




#     Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#             Conv2d-5           [-1, 64, 56, 56]          36,864
#        BatchNorm2d-6           [-1, 64, 56, 56]             128
#               ReLU-7           [-1, 64, 56, 56]               0
#             Conv2d-8           [-1, 64, 56, 56]          36,864
#        BatchNorm2d-9           [-1, 64, 56, 56]             128
#              ReLU-10           [-1, 64, 56, 56]               0
#        BasicBlock-11           [-1, 64, 56, 56]               0
#            Conv2d-12           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-13           [-1, 64, 56, 56]             128
#              ReLU-14           [-1, 64, 56, 56]               0
#            Conv2d-15           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-16           [-1, 64, 56, 56]             128
#              ReLU-17           [-1, 64, 56, 56]               0
#        BasicBlock-18           [-1, 64, 56, 56]               0
#            Conv2d-19           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-20           [-1, 64, 56, 56]             128
#              ReLU-21           [-1, 64, 56, 56]               0
#            Conv2d-22           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-23           [-1, 64, 56, 56]             128
#              ReLU-24           [-1, 64, 56, 56]               0
#        BasicBlock-25           [-1, 64, 56, 56]               0
#            Conv2d-26          [-1, 128, 28, 28]          73,728
#       BatchNorm2d-27          [-1, 128, 28, 28]             256
#              ReLU-28          [-1, 128, 28, 28]               0
#            Conv2d-29          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-30          [-1, 128, 28, 28]             256
#            Conv2d-31          [-1, 128, 28, 28]           8,192
#       BatchNorm2d-32          [-1, 128, 28, 28]             256
#              ReLU-33          [-1, 128, 28, 28]               0
#        BasicBlock-34          [-1, 128, 28, 28]               0
#            Conv2d-35          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-36          [-1, 128, 28, 28]             256
#              ReLU-37          [-1, 128, 28, 28]               0
#            Conv2d-38          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-39          [-1, 128, 28, 28]             256
#              ReLU-40          [-1, 128, 28, 28]               0
#        BasicBlock-41          [-1, 128, 28, 28]               0
#            Conv2d-42          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-43          [-1, 128, 28, 28]             256
#              ReLU-44          [-1, 128, 28, 28]               0
#            Conv2d-45          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-46          [-1, 128, 28, 28]             256
#              ReLU-47          [-1, 128, 28, 28]               0
#        BasicBlock-48          [-1, 128, 28, 28]               0
#            Conv2d-49          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-50          [-1, 128, 28, 28]             256
#              ReLU-51          [-1, 128, 28, 28]               0
#            Conv2d-52          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-53          [-1, 128, 28, 28]             256
#              ReLU-54          [-1, 128, 28, 28]               0
#        BasicBlock-55          [-1, 128, 28, 28]               0
#            Conv2d-56          [-1, 256, 14, 14]         294,912
#       BatchNorm2d-57          [-1, 256, 14, 14]             512
#              ReLU-58          [-1, 256, 14, 14]               0
#            Conv2d-59          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-60          [-1, 256, 14, 14]             512
#            Conv2d-61          [-1, 256, 14, 14]          32,768
#       BatchNorm2d-62          [-1, 256, 14, 14]             512
#              ReLU-63          [-1, 256, 14, 14]               0
#        BasicBlock-64          [-1, 256, 14, 14]               0
#            Conv2d-65          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-66          [-1, 256, 14, 14]             512
#              ReLU-67          [-1, 256, 14, 14]               0
#            Conv2d-68          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-69          [-1, 256, 14, 14]             512
#              ReLU-70          [-1, 256, 14, 14]               0
#        BasicBlock-71          [-1, 256, 14, 14]               0
#            Conv2d-72          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-73          [-1, 256, 14, 14]             512
#              ReLU-74          [-1, 256, 14, 14]               0
#            Conv2d-75          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-76          [-1, 256, 14, 14]             512
#              ReLU-77          [-1, 256, 14, 14]               0
#        BasicBlock-78          [-1, 256, 14, 14]               0
#            Conv2d-79          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-80          [-1, 256, 14, 14]             512
#              ReLU-81          [-1, 256, 14, 14]               0
#            Conv2d-82          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-83          [-1, 256, 14, 14]             512
#              ReLU-84          [-1, 256, 14, 14]               0
#        BasicBlock-85          [-1, 256, 14, 14]               0
#            Conv2d-86          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-87          [-1, 256, 14, 14]             512
#              ReLU-88          [-1, 256, 14, 14]               0
#            Conv2d-89          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-90          [-1, 256, 14, 14]             512
#              ReLU-91          [-1, 256, 14, 14]               0
#        BasicBlock-92          [-1, 256, 14, 14]               0
#            Conv2d-93          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-94          [-1, 256, 14, 14]             512
#              ReLU-95          [-1, 256, 14, 14]               0
#            Conv2d-96          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-97          [-1, 256, 14, 14]             512
#              ReLU-98          [-1, 256, 14, 14]               0
#        BasicBlock-99          [-1, 256, 14, 14]               0
#           Conv2d-100            [-1, 512, 7, 7]       1,179,648
#      BatchNorm2d-101            [-1, 512, 7, 7]           1,024
#             ReLU-102            [-1, 512, 7, 7]               0
#           Conv2d-103            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-104            [-1, 512, 7, 7]           1,024
#           Conv2d-105            [-1, 512, 7, 7]         131,072
#      BatchNorm2d-106            [-1, 512, 7, 7]           1,024
#             ReLU-107            [-1, 512, 7, 7]               0
#       BasicBlock-108            [-1, 512, 7, 7]               0
#           Conv2d-109            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-110            [-1, 512, 7, 7]           1,024
#             ReLU-111            [-1, 512, 7, 7]               0
#           Conv2d-112            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-113            [-1, 512, 7, 7]           1,024
#             ReLU-114            [-1, 512, 7, 7]               0
#       BasicBlock-115            [-1, 512, 7, 7]               0
#           Conv2d-116            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-117            [-1, 512, 7, 7]           1,024
#             ReLU-118            [-1, 512, 7, 7]               0
#           Conv2d-119            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-120            [-1, 512, 7, 7]           1,024
#             ReLU-121            [-1, 512, 7, 7]               0
#       BasicBlock-122            [-1, 512, 7, 7]               0
# AdaptiveAvgPool2d-123            [-1, 512, 1, 1]               0
#         Identity-124                  [-1, 512]               0
#           ResNet-125                  [-1, 512]               0
#           Linear-126                  [-1, 128]          65,664