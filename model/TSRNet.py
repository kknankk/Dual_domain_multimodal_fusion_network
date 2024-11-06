
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
import math
import numpy as np
import torch
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
import math
import random

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        x = torch.matmul(attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
    
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder1D(nn.Module):
    def __init__(self, nc):
        super(Encoder1D, self).__init__()
        ndf = 32
        self.main = nn.Sequential(
            nn.Conv1d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 16, 50, 15, 1, 0),
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Decoder1D(nn.Module):
    def __init__(self, nc):
        super(Decoder1D, self).__init__()
        ngf = 32
        self.main=nn.Sequential(
            nn.ConvTranspose1d(50, ngf*16, 15, 1, 0),
            nn.BatchNorm1d(ngf*16),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf*2, 4, 2, 1),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf , 4, 2, 1),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Encoder2D(nn.Module):
    def __init__(self, nc):
        super(Encoder2D, self).__init__()
        ndf = 32
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 3, 1, 0),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 3, 1, 0),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 50, 3, 1, 0),
        )

    def forward(self, input):
        # print(f'encoder 2d input {input.shape}')
        output = self.main(input)
        return output
    




# from utils import *
# from lib.modules import *

#Time and Spectrogram Restoration
class TSRNet(nn.Module):

    def __init__(self, enc_in):
        super(TSRNet, self).__init__()

        self.channel = enc_in

        # Time series module 
        self.time_encoder = Encoder1D(enc_in)
        # print(f'list para {list(self.time_encoder.parameters())}')
        self.time_decoder = Decoder1D(enc_in+1)
        
        # Spectrogram module
        self.spec_encoder = Encoder2D(enc_in)
    
    
        self.conv_spec1 = nn.Conv1d(50*51, 50, 3, 1, 1, bias=False)
        
        # self.mlp = nn.Sequential(
        #     # nn.Linear(202, 136),
        #     nn.Linear(8400, 136),
        #     nn.LayerNorm(136),
        #     nn.ReLU()
        # )
        self.mlp = nn.Sequential(
        nn.Linear(8400, 512),  # 第一个线性层：8400 到 512
        nn.LayerNorm(512),     # 对第一个线性层的输出进行层归一化
        nn.ReLU(),             # 激活函数
        nn.Linear(512, 7)      # 第二个线性层：512 到 7
            )
        
        self.attn1 = MultiHeadedAttention(2, 50)
        self.drop = nn.Dropout(0.1)
        self.layer_norm1 = LayerNorm(50)

    def attention_func(self,x, attn, norm):
        attn_latent = attn(x, x, x)
        attn_latent = norm(x + self.drop(attn_latent))
        return attn_latent
    
    def forward(self, time_ecg, spectrogram_ecg):
        #Time ECG encode
        # if not list(self.time_encoder.parameters()):
        #     raise RuntimeError("Time Encoder has no parameters.")
        
        device=time_ecg.device
        self.time_encoder.to(device)
        if not list(self.time_encoder.parameters()):
            raise RuntimeError("Time Encoder has no parameters.")
        # device1=spectrogram_ecg.device

        # print(f'time_ecg is on device: {time_ecg.device}')
        # print(f'freq_ecg is on device: {spectrogram_ecg.device}')
        # model_device = next(self.time_encoder.parameters()).device
        # print(f'Model is on device: {model_device}')
        # print(f'tsrnet input temp_ecg {time_ecg.shape}')
        a=time_ecg.transpose(-1,1)
        # print(f'a {a.shape}')#[bs,12,4096]
        time_features = self.time_encoder(time_ecg.transpose(-1,1)) #(32, 50, 136)

        #Spectrogram ECG encode
        spectrogram_features = self.spec_encoder(spectrogram_ecg.permute(0,3,1,2)) #(32, 50, 63, 66)
        n, c, h, w = spectrogram_features.shape
        spectrogram_features = self.conv_spec1(spectrogram_features.contiguous().view(n, c*h, w)) #(32, 50, 66)
        
        latent_combine = torch.cat([time_features, spectrogram_features], dim=-1)
        #Cross-attention
        latent_combine = latent_combine.transpose(-1, 1)
        attn_latent = self.attention_func(latent_combine, self.attn1, self.layer_norm1)
        attn_latent = self.attention_func(attn_latent, self.attn1, self.layer_norm1)
        # print(f'attn_latent {attn_latent.shape}')#[[64, 168, 50]]
        # latent_combine = attn_latent.transpose(-1, 1)
        attn_latent=attn_latent.reshape(attn_latent.size(0),-1)
        # print(f'attn_latent1 {attn_latent.shape}')#[62,8400]
        # attn_latent=attn_latent.transpose(0,1)
        # print(f'attn_latent1 {attn_latent.shape}')


        
        latent_combine = self.mlp(attn_latent)
        # print(f'latent_combine {latent_combine.shape}')
        return latent_combine
        # output = self.time_decoder(latent_combine)
        # output = output.transpose(-1, 1)

        # return  (output[:,:,0:self.channel],output[:,:,self.channel:self.channel+1])

    

from torchsummary import summary
# time_ecg_input = torch.randn(32, 12, 4096)       # Batch size: 32, Channels: 12, Time steps: 4096
# spectrogram_ecg_input = torch.randn(32, 63, 66, 12)  # Batch size: 32, Height: 63, Width: 66, Channels: 12

# # 初始化模型
# model = TSRNet(enc_in=12)

# 使用 summary 函数输出模型结构和各层大小
# summary(model, [(4096,12), (63, 66, 12)])  # 输入各个模块的形状


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 定义模型
# model = TSRNet(enc_in=12).to(device)  # 将模型移动到 GPU

# # 输入大小
# time_ecg_input = torch.randn(32, 12, 4096).to(device)  # 将输入数据移动到 GPU
# spectrogram_ecg_input = torch.randn(32, 63, 66, 12).to(device)  # 将输入数据移动到 GPU

# # # 使用 summary 函数输出模型结构和各层大小
# summary(model, time_ecg_input, spectrogram_ecg_input)  # 输入各个模块的形状
# # # summary(model, [(4800,12), (63, 78, 12)])  # 输入各个模块的形状

bs, time_length, dim = 32, 4096, 12
input_tensor = torch.randn(bs, dim, time_length)  # 需要将维度调整为 (batch_size, channels, length)

# 创建 Encoder1D 实例
# encoder = Encoder1D(nc=dim)  # dim 为输入的通道数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder1D(nc=dim).to(device)
# 使用 summary 查看各个阶段的输出形状
# summary(encoder, input_size=(dim, time_length))  # 