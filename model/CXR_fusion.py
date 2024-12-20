import torch
import torch.nn as nn
import torch.nn.functional as F
# from Res2Net_v1b import res2net50_v1b
from torchvision.models import resnet50
from utils import PreNorm, FeedForward, Attention
from einops import rearrange




import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # torch.Size([1, 64, 176, 176])
        x = self.maxpool(x)

        x = self.layer1(x)  # torch.Size([1, 256, 88, 88])
        x = self.layer2(x)  # torch.Size([1, 512, 44, 44])
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b lib.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        cp = torch.load('./res2net50_v1b_26w_4s-3cf99910.pth')
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


# def res2net101_v1b(pretrained=False, **kwargs):
#     """Constructs a Res2Net-50_v1b_26w_4s lib.
#     Args:
#         pretrained (bool): If True, returns a lib pre-trained on ImageNet
#     """
#     model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
#     return model


# def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
#     """Constructs a Res2Net-50_v1b_26w_4s lib.
#     Args:
#         pretrained (bool): If True, returns a lib pre-trained on ImageNet
#     """
#     model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
#         # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
#     return model


# def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
#     """Constructs a Res2Net-50_v1b_26w_4s lib.
#     Args:
#         pretrained (bool): If True, returns a lib pre-trained on ImageNet
#     """
#     model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
#     return model


# def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
#     """Constructs a Res2Net-50_v1b_26w_4s lib.
#     Args:
#         pretrained (bool): If True, returns a lib pre-trained on ImageNet
#     """
#     model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['res2net152_v1b_26w_4s']))
#     return model


# if __name__ == '__main__':
#     images = torch.rand(1, 3, 352, 352).cuda(0)
#     model = res2net50_v1b_26w_4s(pretrained=False)
#     model = model.cuda(0)
#     print(model(images).size())





def norm(x):
    return (1 - torch.exp(-x)) / (1 + torch.exp(-x))


def norm_(x):
    import numpy as np
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class HOR(nn.Module):
    def __init__(self):
        super(HOR, self).__init__()
        self.high = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.low = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.value = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.e_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.latter = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

    def forward(self, x_latter, x):
        b, c, h, w = x_latter.shape
        _, c_, _, _ = x.shape
        x_latter_ = self.high(x_latter).reshape(b, c, h * w).contiguous()
        x_ = self.low(x).reshape(b, c_, h * w).permute(0, 2, 1).contiguous()

        p = torch.bmm(x_, x_latter_).contiguous()
        p = self.softmax(p).contiguous()

        e_ = torch.bmm(p, self.value(x).reshape(b, c, h * w).permute(0, 2, 1)).contiguous()
        e = e_ + x_
        e = e.permute(0, 2, 1).contiguous()
        e = self.e_conv(e.reshape(b, c, h, w)).reshape(b, c, h * w).contiguous()

        # e = e.permute(0, 2, 1)
        x_latter_ = self.latter(x_latter).reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        t = torch.bmm(e, x_latter_).contiguous()
        t = self.softmax(t).contiguous()

        x_ = self.mid(x).view(b, c_, h * w).permute(0, 2, 1).contiguous()
        out = torch.bmm(x_, t).permute(0, 2, 1).reshape(b, c, h, w).contiguous()

        return out


class channel_shuffle(nn.Module):
    def __init__(self,groups=4):
        super(channel_shuffle,self).__init__()
        self.groups=groups
    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
               channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

class two_ConvBnRule(nn.Module):

    def __init__(self, in_chan, out_chan=64):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        if mid:
            feat_mid = feat

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        if mid:
            return feat, feat_mid
        return feat

def Seg():

    dict = {0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
                 9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
                 17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
                 25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
                 33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
                 41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
                 49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
                 57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
    a = torch.zeros(1, 64, 1, 1)

    for i in range(0, 32):
        a[0, dict[i+32], 0, 0] = 1

    return a


class PAM(nn.Module):

    def __init__(self, in_dim):

        super(PAM, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels= 2,
            kernel_size=3,
            padding=1
        )

        self.v_rgb = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)
        self.v_freq = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)

    def forward(self, rgb, freq):

        attmap = self.conv( torch.cat( (rgb,freq),1) )
        attmap = torch.sigmoid(attmap)

        rgb = attmap[:,0:1,:,:] * rgb * self.v_rgb
        freq = attmap[:,1:,:,:] * freq * self.v_freq
        out = rgb + freq

        return out


class Main_Net(nn.Module):

    def __init__(self):
        super(Main_Net, self).__init__()
        self.resnet = res2net50_v1b(pretrained=False)
        self.res_con1 = self.resnet.conv1
        self.res_bn1 = self.resnet.bn1
        self.res_relu = self.resnet.relu
        self.res_mxpool = self.resnet.maxpool
        self.res_layer1 = self.resnet.layer1
        self.res_layer2 = self.resnet.layer2
        self.res_layer3 = self.resnet.layer3
        self.res_layer4 = self.resnet.layer4
        self.seg=Seg()
        self.shuffle=channel_shuffle()

        self.high_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0)
        self.low_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0)

        self.band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0)
        self.spatial = Transformer(dim=192, depth=1, heads=2, dim_head=64, mlp_dim=64*2, dropout=0)

        #model_s#
        self.conv_l2 = two_ConvBnRule(256)
        self.conv_l3 = two_ConvBnRule(512)
        self.conv_l4 = two_ConvBnRule(1024)
        self.conv_l5 = two_ConvBnRule(2048)

        #decoder_convlution#
        "chanal_decoder1 = chanal_feat5 + 64 = 1028 + 64 =1092"
        self.conv_decoder1 = two_ConvBnRule(128)
        self.conv_decoder2 = two_ConvBnRule(128)
        self.conv_decoder3 = two_ConvBnRule(128)

        self.PAM2 = PAM(in_dim=64)
        self.PAM3 = PAM(in_dim=64)
        self.PAM4 = PAM(in_dim=64)
        self.PAM5 = PAM(in_dim=64)

        self.conv_r2 = two_ConvBnRule(64)
        self.conv_r3 = two_ConvBnRule(64)
        self.conv_r4 = two_ConvBnRule(64)
        self.conv_r5 = two_ConvBnRule(64)
        self.hor = HOR()

        #1*1 conv
        self.con1_2 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.con1_3 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.con1_4 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.con1_5 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.vector_y = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
        self.vector_cb = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
        self.vector_cr = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)

        self.freq_out_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_2 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_3 = nn.Conv2d(64, 1, 1, 1, 0)
        self.freq_out_4 = nn.Conv2d(64, 1, 1, 1, 0)

        #output
        self.conv_out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            padding=1,
            kernel_size=3
        )
        self.conv_out_2 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            padding=1,
            kernel_size=3
        )
        self.conv_out_3 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            padding=1,
            kernel_size=3
        )
        self.conv_out_4 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            padding=1,
            kernel_size=3
        )

    def forward(self, x, DCT_x, name=None):
        "We assume that the size of x is [4,4,256,256]"
        "then the size of feat1 is [4,64,64,64]"
        "the size of feat2 is [4,256,64,64]"
        "the size of feat3 is [4,512,32,32]"
        "the size of feat4 is [4,1024,16,16]"
        "the size of feat5 is [4,2048,8,8]"
        size = x.size()[2:]

        feat1 = self.res_con1(x)
        feat1 = self.res_bn1(feat1)
        feat1 = self.res_relu(feat1)
        feat1 = self.res_mxpool(feat1)

        feat2 = self.res_layer1(feat1)
        feat3 = self.res_layer2(feat2)
        feat4 = self.res_layer3(feat3)
        feat5 = self.res_layer4(feat4)

        # Module_s
        feat2 = self.conv_l2(feat2)
        feat3 = self.conv_l3(feat3)
        feat4 = self.conv_l4(feat4)
        feat5, feat5_mid = self.conv_l5(feat5, mid=True)
        feat5 = self.hor(feat5, feat5_mid)
        
        self.seg = self.seg.to(DCT_x.device)
        feat_y = DCT_x[:, 0:64, :, :] * (self.seg + norm(self.vector_y))
        feat_Cb = DCT_x[:, 64:128, :, :] * (self.seg + norm(self.vector_cb))
        feat_Cr = DCT_x[:, 128:192, :, :] * (self.seg + norm(self.vector_cr))

        origin_feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
        origin_feat_DCT = self.shuffle(origin_feat_DCT)

        high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
        low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)

        b, n, h, w = high.shape
        high = torch.nn.functional.interpolate(high, size=(16, 16))
        low = torch.nn.functional.interpolate(low, size=(16, 16))

        high = rearrange(high, 'b n h w -> b n (h w)')
        low = rearrange(low, 'b n h w -> b n (h w)')

        high = self.high_band(high)
        low = self.low_band(low)

        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)
        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr = torch.cat([r_l, r_h], 1)

        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)

        feat_DCT = self.band(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = self.spatial(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = rearrange(feat_DCT, 'b n (h w) -> b n h w', h=16)
        feat_DCT = torch.nn.functional.interpolate(feat_DCT, size=(h, w))

        feat_DCT = origin_feat_DCT + feat_DCT

        #using 1*1conv to change the numbers of the channel of DCT_x
        feat_DCT2 = self.con1_2(feat_DCT)
        feat_DCT3 = self.con1_3(feat_DCT)
        feat_DCT4 = self.con1_4(feat_DCT)
        feat_DCT5 = self.con1_5(feat_DCT)

        feat_DCT2 = torch.nn.functional.interpolate(feat_DCT2,size=feat2.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT3 = torch.nn.functional.interpolate(feat_DCT3,size=feat3.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT4 = torch.nn.functional.interpolate(feat_DCT4,size=feat4.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT5 = torch.nn.functional.interpolate(feat_DCT5,size=feat5.size()[2:],mode='bilinear',align_corners=True)

        #feature fusion
        feat2 = self.PAM2(feat2, feat_DCT2)
        feat3 = self.PAM3(feat3, feat_DCT3)
        feat4 = self.PAM4(feat4, feat_DCT4)
        feat5 = self.PAM5(feat5, feat_DCT5)

        freq_output = self.freq_out_1(feat2)
        freq_output_1 = self.freq_out_2(feat3)
        freq_output_2 = self.freq_out_3(feat4)
        freq_output_3 = self.freq_out_4(feat5)

        feat2 = self.conv_r2(feat2)
        feat3 = self.conv_r3(feat3)
        feat4 = self.conv_r4(feat4)
        feat5 = self.conv_r5(feat5)

        #connect feat5 and feat4#
        size4 = feat4.size()[2:]
        feat5 = torch.nn.functional.interpolate(feat5, size=size4, mode='bilinear', align_corners=True)
        feat4 = torch.cat((feat4, feat5), 1)
        feat4 = self.conv_decoder1(feat4)

        # connect feat4 and feat3#
        size3 = feat3.size()[2:]
        feat4 = torch.nn.functional.interpolate(feat4, size=size3, mode='bilinear', align_corners=True)
        feat3 = torch.cat((feat3, feat4), 1)
        feat3 = self.conv_decoder2(feat3)

        # connect feat3 and feat2#
        size2 = feat2.size()[2:]
        feat3 = torch.nn.functional.interpolate(feat3, size=size2, mode='bilinear', align_corners=True)
        feat2 = torch.cat((feat2, feat3), 1)
        feat2 = self.conv_decoder3(feat2)

        #output#
        sizex = x.size()[2:]
        output = self.conv_out(feat2)
        output_1 = self.conv_out(feat3)
        output_2 = self.conv_out(feat4)
        output_3 = self.conv_out(feat5)
        # output = torch.nn.functional.interpolate(output, size=sizex, mode='bilinear', align_corners=True)
        # output_1 = torch.nn.functional.interpolate(output_1, size=sizex, mode='bilinear', align_corners=True)
        # output_2 = torch.nn.functional.interpolate(output_2, size=sizex, mode='bilinear', align_corners=True)
        # output_3 = torch.nn.functional.interpolate(output_3, size=sizex, mode='bilinear', align_corners=True)
        return output, output_1, output_2, output_3, freq_output, freq_output_1, freq_output_2, freq_output_3

# if __name__ == "__main__":
#     x = torch.randn(4, 3, 256, 256)
#     y = torch.randn(4, 192, 32, 32)
#     # detail = Detail_Branch()
#     # feat = detail(x)
#     # print('detail', feat.size())

#     net = Main_Net()
#     logits = net(x, y)
#     print('\nlogits', logits.size())

# from torchinfo import summary

# model = resnet50()  # 替换为 res2net50_v1b()

# # 创建一个示例输入
# input_tensor = torch.randn(16, 3, 224, 224)

# # 使用 torchinfo 显示模型摘要
# summary(model, input_size=(16, 3, 224, 224))