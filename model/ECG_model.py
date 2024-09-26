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

