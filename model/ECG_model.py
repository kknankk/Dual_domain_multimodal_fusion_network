import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import argparse


# 将项目根目录添加到 Python 路径
# sys.path.append(os.path.abspath('/data/mimic/MIMIC_subset'))
# print(a)
sys.path.append(os.path.abspath('/home/ke/MIMIC_subset/MIMIC_subset'))

from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
import numpy as np

# class LSTM(nn.Module):

#     def __init__(self, input_dim=4096, num_classes=7, hidden_dim=128, batch_first=True, dropout=0.0, layers=1):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.layers = layers
#         for layer in range(layers):
#             setattr(self, f'layer{layer}', nn.LSTM(
#                 input_dim, hidden_dim,
#                 batch_first=batch_first,
#                 dropout = dropout)
#             )
#             input_dim = hidden_dim
#         self.do = None
#         if dropout > 0.0:
#             self.do = nn.Dropout(dropout)
#         self.feats_dim = hidden_dim
#         self.dense_layer = nn.Linear(hidden_dim, num_classes)
#         self.initialize_weights()
#         # self.activation = torch.sigmoid
#     def initialize_weights(self):
#         for model in self.modules():

#             if type(model) in [nn.Linear]:
#                 nn.init.xavier_uniform_(model.weight)
#                 nn.init.zeros_(model.bias)
#             elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
#                 nn.init.orthogonal_(model.weight_hh_l0)
#                 nn.init.xavier_uniform_(model.weight_ih_l0)
#                 nn.init.zeros_(model.bias_hh_l0)
#                 nn.init.zeros_(model.bias_ih_l0)

#     def forward(self, x, seq_lengths=[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]):
# #TODO:check the batch size when gpu is free (now batch size if from 14-16)
#         # print(f'LSTM model input size {x.shape}')
#         # print(f'seq_lengths {len(seq_lengths)}')
#         # x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
#         x=torch.transpose(x,2,1)
#         for layer in range(self.layers):
#             x, (ht, _) = getattr(self, f'layer{layer}')(x)
#         feats = ht.squeeze()
#         if self.do is not None:
#             feats = self.do(feats)
#         scores = self.dense_layer(feats)

#         scores = torch.sigmoid(scores)
#         # print(scores)
#         return scores


# class ECGModel(nn.Module):
#     def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=7,dropout=0.5):
#         super(ECGModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False,dropout=dropout)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         # print(f'ECG MODELinput size {x.shape}')
#         # x=torch.transpose(x,2,1)
#         x = x.permute(2, 0, 1)
#         out, _ = self.lstm(x)
#         a=out[:, -1, :]
#         # out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
#         out = self.fc(out[-1, :, :])
#         return a,out


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
        scores1 = F.relu(ret)
        scores=torch.sigmoid(scores1)
        return scores1,scores # output is N x output_shape



#----------------------ADJUST from https://github.com/antonior92/ecg-age-prediction/blob/main/resnet.py#L79
import torch.nn as nn
import numpy as np


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


# self.res_blocks = nn.ModuleList()
# for i, (n_filters, n_samples) in enumerate(self.blocks_dim):
#     n_filters_in, n_filters_out = n_filters_out, n_filters
#     n_samples_in, n_samples_out = n_samples_out, n_samples
#     downsample = _downsample(n_samples_in, n_samples_out)
    
#     # print(f'{i} th downsample {downsample}' )
#     # print(f'i th n_filters_out {n_filters_out}')
#     resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
#     self.res_blocks.append(resblk1d)




        # Linear layer
        n_filters_last, n_samples_last = self.blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, 512)
        self.lin1 = nn.Linear(512, n_classes)
        self.n_blk = len(self.blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        # print(f' 1 input size {x.shape}')
        # x=x.permute(0,2,1)
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
        x1 = x.view(x.size(0), -1)

        # Fully conected layer
        x2 = self.lin(x1)
        x=self.lin1(x2)
        return x2,x


#----------print resnet1d------------
# ResNet1d(
#   (conv1): Conv1d(12, 64, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
#   (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (resblock1d_0): ResBlock1d(
#     (conv1): Conv1d(12, 64, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
#     (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU()
#     (dropout1): Dropout(p=0.8, inplace=False)
#     (conv2): Conv1d(64, 64, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
#     (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (dropout2): Dropout(p=0.8, inplace=False)
#     (skip_connection): Sequential(
#       (0): Conv1d(12, 64, kernel_size=(1,), stride=(1,), bias=False)
#     )
#   )
#   (resblock1d_1): ResBlock1d(
#     (conv1): Conv1d(12, 128, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
#     (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU()
#     (dropout1): Dropout(p=0.8, inplace=False)
#     (conv2): Conv1d(128, 128, kernel_size=(17,), stride=(4,), padding=(7,), bias=False)
#     (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (dropout2): Dropout(p=0.8, inplace=False)
#     (skip_connection): Sequential(
#       (0): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
#       (1): Conv1d(12, 128, kernel_size=(1,), stride=(1,), bias=False)
#     )
#   )
#   (resblock1d_2): ResBlock1d(
#     (conv1): Conv1d(12, 196, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
#     (bn1): BatchNorm1d(196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU()
#     (dropout1): Dropout(p=0.8, inplace=False)
#     (conv2): Conv1d(196, 196, kernel_size=(17,), stride=(4,), padding=(7,), bias=False)
#     (bn2): BatchNorm1d(196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (dropout2): Dropout(p=0.8, inplace=False)
#     (skip_connection): Sequential(
#       (0): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
#       (1): Conv1d(12, 196, kernel_size=(1,), stride=(1,), bias=False)
#     )
#   )
#   (resblock1d_3): ResBlock1d(
#     (conv1): Conv1d(12, 256, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
#     (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU()
#     (dropout1): Dropout(p=0.8, inplace=False)
#     (conv2): Conv1d(256, 256, kernel_size=(17,), stride=(4,), padding=(7,), bias=False)
#     (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (dropout2): Dropout(p=0.8, inplace=False)
#     (skip_connection): Sequential(
#       (0): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
#       (1): Conv1d(12, 256, kernel_size=(1,), stride=(1,), bias=False)
#     )
#   )
#   (resblock1d_4): ResBlock1d(
#     (conv1): Conv1d(12, 320, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
#     (bn1): BatchNorm1d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU()
#     (dropout1): Dropout(p=0.8, inplace=False)
#     (conv2): Conv1d(320, 320, kernel_size=(17,), stride=(4,), padding=(7,), bias=False)
#     (bn2): BatchNorm1d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (dropout2): Dropout(p=0.8, inplace=False)
#     (skip_connection): Sequential(
#       (0): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
#       (1): Conv1d(12, 320, kernel_size=(1,), stride=(1,), bias=False)
#     )
#   )
#   (lin): Linear(in_features=5120, out_features=7, bias=True)
# )

# ------------------------------resnet1d----------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv1d-1             [-1, 64, 4096]          13,056
#        BatchNorm1d-2             [-1, 64, 4096]             128
#             Conv1d-3             [-1, 64, 4096]          69,632
#        BatchNorm1d-4             [-1, 64, 4096]             128
#               ReLU-5             [-1, 64, 4096]               0
#            Dropout-6             [-1, 64, 4096]               0
#             Conv1d-7             [-1, 64, 4096]          69,632
#        BatchNorm1d-8             [-1, 64, 4096]             128
#               ReLU-9             [-1, 64, 4096]               0
#           Dropout-10             [-1, 64, 4096]               0
#        ResBlock1d-11  [[-1, 64, 4096], [-1, 64, 4096]]               0
#         MaxPool1d-12             [-1, 64, 1024]               0
#            Conv1d-13            [-1, 128, 1024]           8,192
#            Conv1d-14            [-1, 128, 4096]         139,264
#       BatchNorm1d-15            [-1, 128, 4096]             256
#              ReLU-16            [-1, 128, 4096]               0
#           Dropout-17            [-1, 128, 4096]               0
#            Conv1d-18            [-1, 128, 1024]         278,528
#       BatchNorm1d-19            [-1, 128, 1024]             256
#              ReLU-20            [-1, 128, 1024]               0
#           Dropout-21            [-1, 128, 1024]               0
#        ResBlock1d-22  [[-1, 128, 1024], [-1, 128, 1024]]               0
#         MaxPool1d-23             [-1, 128, 256]               0
#            Conv1d-24             [-1, 196, 256]          25,088
#            Conv1d-25            [-1, 196, 1024]         426,496
#       BatchNorm1d-26            [-1, 196, 1024]             392
#              ReLU-27            [-1, 196, 1024]               0
#           Dropout-28            [-1, 196, 1024]               0
#            Conv1d-29             [-1, 196, 256]         653,072
#       BatchNorm1d-30             [-1, 196, 256]             392
#              ReLU-31             [-1, 196, 256]               0
#           Dropout-32             [-1, 196, 256]               0
#        ResBlock1d-33  [[-1, 196, 256], [-1, 196, 256]]               0
#         MaxPool1d-34              [-1, 196, 64]               0
#            Conv1d-35              [-1, 256, 64]          50,176
#            Conv1d-36             [-1, 256, 256]         852,992
#       BatchNorm1d-37             [-1, 256, 256]             512
#              ReLU-38             [-1, 256, 256]               0
#           Dropout-39             [-1, 256, 256]               0
#            Conv1d-40              [-1, 256, 64]       1,114,112
#       BatchNorm1d-41              [-1, 256, 64]             512
#              ReLU-42              [-1, 256, 64]               0
#           Dropout-43              [-1, 256, 64]               0
#        ResBlock1d-44  [[-1, 256, 64], [-1, 256, 64]]               0
#         MaxPool1d-45              [-1, 256, 16]               0
#            Conv1d-46              [-1, 320, 16]          81,920
#            Conv1d-47              [-1, 320, 64]       1,392,640
#       BatchNorm1d-48              [-1, 320, 64]             640
#              ReLU-49              [-1, 320, 64]               0
#           Dropout-50              [-1, 320, 64]               0
#            Conv1d-51              [-1, 320, 16]       1,740,800
#       BatchNorm1d-52              [-1, 320, 16]             640
#              ReLU-53              [-1, 320, 16]               0
#           Dropout-54              [-1, 320, 16]               0
#        ResBlock1d-55  [[-1, 320, 16], [-1, 320, 16]]               0
#            Linear-56                  [-1, 512]       2,621,952
#            Linear-57                    [-1, 4]           2,052
# ================================================================
# Total params: 9,543,588
# Trainable params: 9,543,588
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.19
# Forward/backward pass size (MB): 676762.63
# Params size (MB): 36.41
# Estimated Total Size (MB): 676799.23



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
            #TODO: 先删除了relu
            nn.ReLU(inplace=True),
            #TODO: 先删除了relu
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
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


def se_resnet34(num_classes=7,pretrained=True):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model



class spectrogram_model(nn.Module):
    def __init__(self,num_classes=4):
        super(spectrogram_model,self).__init__()
        self.backbone = se_resnet34()
        self.backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3)
        list_of_modules = list(self.backbone.children())
        self.features = nn.Sequential(*list_of_modules[:-1])
        num_ftrs = self.backbone.fc.in_features
    
        self.fc = nn.Sequential(
                nn.Linear(in_features=num_ftrs,out_features=num_ftrs//2),
                nn.Dropout(0.8),
                nn.Linear(in_features=num_ftrs//2,out_features=num_classes)
            )

    def forward(self, x):
        # print(f'model ecg {x.shape}')
        h = self.features(x)
        # print(f'before squeeze {h.shape}')
        # h1 = h.squeeze()
        h1 = h.squeeze(dim=-1).squeeze(dim=-1)

        # print(f'after squeeze {h1.shape}')
        x = self.fc(h1)
        # print(f'after fc {x.shape}')
        if x.dim() == 1:  # 如果 output 是 torch.Size([7])
            x = x.unsqueeze(0)
        return h1,x

import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomResNet18, self).__init__()
        # 加载预训练的 ResNet18 模型
        self.model = models.resnet18(pretrained=True)
        
        # 修改第一层卷积层以接受 12 个通道的输入
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改全连接层的输出维度
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x=self.model(x)
        # features = self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(self.model.conv1(x)))))
        
        # # 全连接层输出
        # output = self.model.fc(features.view(features.size(0), -1))  # 展平特征
        
        # return features, output  # 返回特征和最终输出

        # return x
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features = self.model.layer4(x)

        # 使用平均池化来获得特征
        features = self.model.avgpool(features)  # 现在的形状是 (N, 512, 1, 1)
        features = features.view(features.size(0), -1)  # 展平特征

        # 全连接层输出
        output = self.model.fc(features)
        
        return features, output  # 返回特征和最终输出


# 示例使用
# model = CustomResNet18(num_classes=7)
# print(model)
