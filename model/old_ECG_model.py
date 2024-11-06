import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath('/data/ke/MIMIC_subset'))
# print(a)
from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
import numpy as np

class LSTM(nn.Module):

    def __init__(self, input_dim=4096, num_classes=7, hidden_dim=128, batch_first=True, dropout=0.0, layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        for layer in range(layers):
            setattr(self, f'layer{layer}', nn.LSTM(
                input_dim, hidden_dim,
                batch_first=batch_first,
                dropout = dropout)
            )
            input_dim = hidden_dim
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.feats_dim = hidden_dim
        self.dense_layer = nn.Linear(hidden_dim, num_classes)
        self.initialize_weights()
        # self.activation = torch.sigmoid
    def initialize_weights(self):
        for model in self.modules():

            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x, seq_lengths=[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]):
#TODO:check the batch size when gpu is free (now batch size if from 14-16)
        # print(f'LSTM model input size {x.shape}')
        # print(f'seq_lengths {len(seq_lengths)}')
        # x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        x=torch.transpose(x,2,1)
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht.squeeze()
        if self.do is not None:
            feats = self.do(feats)
        scores = self.dense_layer(feats)

        scores = torch.sigmoid(scores)
        # print(scores)
        return scores


class ECGModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=1, num_classes=7,dropout=0.5):
        super(ECGModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # print(f'ECG MODELinpur size {x.shape}')
        x=torch.transpose(x,2,1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# -*- coding: utf-8 -*-
"""
Follows https://www.researchgate.net/publication/334407179_ECG_Arrhythmia_Classification_Using_STFT-Based_Spectrogram_and_Convolutional_Neural_Network 
per reviewer 3's suggestions

"""
import torch
import os
from tqdm import tqdm

import torchaudio

import torch
import torch.nn as nn
from copy import deepcopy
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import wfdb

# from MODELS.GenericModel import GenericModel

# # import Support_Functions
# from MODELS.Support_Functions import Custom_Dataset
# from MODELS.Support_Functions import Save_NN
# from MODELS.Support_Functions import Save_Train_Args
# from MODELS.Support_Functions import Structure_Data_NCHW
# from MODELS.Support_Functions import Get_Norm_Func_Params
# from MODELS.Support_Functions import Normalize
# from MODELS.Support_Functions import Get_Loss_Params
# from MODELS.Support_Functions import Get_Loss

#adjust from https://github.com/cavalab/ecg-survival-benchmark/blob/ba18cddb81c5e211b0803c66e7b42165a4a674a2/MODELS/SpectCNNReg_PyCox.py#L54

class Spect_CNN(nn.Module):
    # just wrapping the LSTM here to return the correct h/c outputs, not the outputs per ECG time point
    # https://discuss.pytorch.org/t/cnn-lstm-for-video-classification/185303/7
    
    def __init__ (self):
        super(Spect_CNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(12, 32, (4,4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(64, 64, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features = 246016, out_features = 7)
            )
        
        self.Spect = torchaudio.transforms.MelSpectrogram(sample_rate = 400, n_mels = 512, n_fft=1024, hop_length=8).to('cuda')
        

    def forward(self, input_ecg):
        # hmm. looks like PyCox adds a squeeze in dataloader outputs.
        # ... but only when training. 
        
        # a = self.Spect( torch.transpose( input_ecg,2,1)) # [batch_size,channel(12),512,513]
        a=self.Spect(input_ecg)
        # print(f'spect_cnn after spect input size {a.shape}')
        ret = self.model(a[:,:,:,:512]) # cut last freq to line up size
        # print(f'after spect_cnn the ret: {ret}')
        # print(f'ret shape {ret.shape}')#[batch size,num_cls]
        scores = F.relu(ret)
        scores=torch.sigmoid(scores)
        return scores # output is N x output_shape



class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self,  n_classes=7, kernel_size=17, dropout_rate=0.8):
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
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(self.blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            
            # print(f'{i} th downsample {downsample}' )
            # print(f'i th n_filters_out {n_filters_out}')
            resblk1d = ResBlock1d( n_filters_in,n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = self.blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(self.blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        # print(f' 1 input size {x.shape}')
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
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x


#-------------------frequency domain model
#adaptive from https://github.com/UARK-AICV/ECG_SSL_12Lead/blob/main/models/seresnet2d.py#L112

from torchvision.models import ResNet
from torch import nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet34(num_classes=7):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model



class spectrogram_model(nn.Module):
    def __init__(self,num_classes=7):
        super(spectrogram_model,self).__init__()
        self.backbone = se_resnet34()
        self.backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3)
        list_of_modules = list(self.backbone.children())
        self.features = nn.Sequential(*list_of_modules[:-1])
        num_ftrs = self.backbone.fc.in_features
    
        self.fc = nn.Sequential(
                nn.Linear(in_features=num_ftrs,out_features=num_ftrs//2),
                nn.Linear(in_features=num_ftrs//2,out_features=num_classes)
            )

    def forward(self, x):
        h = self.features(x)
        h1 = h.squeeze()
        x = self.fc(h1)
        return h1,x

