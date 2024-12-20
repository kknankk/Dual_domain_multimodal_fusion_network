import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math
import matplotlib.pyplot as plt


#==========================modified from ================
#https://github.com/Shuheng-Li/UniTS-Sensory-Time-Series-Classification/blob/main/source/main.py

class FIC(nn.Module):
    def __init__(self, window_size, stride):
        super(FIC, self).__init__()
        self.window_size = window_size
        self.k = int(window_size / 2)
        self.conv = nn.Conv1d(in_channels = 1, out_channels = 2 * int(window_size / 2), kernel_size = window_size,
            stride = stride, padding = 0, bias = False)
        self.init()

    def forward(self, x):
        # x: B * C  * L   
        B, C = x.size(0), x.size(1)

        x = torch.reshape(x, (B * C, -1)).unsqueeze(1)
        x = self.conv(x)
        x = torch.reshape(x, (B, C, -1, x.size(-1)))
        return x # B * C * fc * L
    

    def init(self):
        '''
            Fourier weights initialization
        '''
        basis = torch.tensor([math.pi * 2 * j / self.window_size for j in range(self.window_size)])
        weight = torch.zeros((self.k * 2, self.window_size))
        for i in range(self.k * 2):
            f = int(i / 2) + 1
            if i % 2 == 0:
                weight[i] = torch.cos(f * basis)
            else:
                weight[i] = torch.sin(-f * basis)
        self.conv.weight = torch.nn.Parameter( weight.unsqueeze(1), requires_grad=True)



class TSEnc(nn.Module):
    def __init__(self, window_size, stride, k):
        super(TSEnc, self).__init__()
        '''
            virtual filter choose 2 * k most important channels
        '''
        self.k = k
        self.window_size = window_size
        self.FIC = FIC(window_size = window_size, stride = stride).cuda()
        self.RPC = nn.Conv1d(1, 2*k, kernel_size = window_size, stride = stride)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        # fic #
        h_f = self.FIC(x)

        # virtual filter #
        h_f_pos, idx_pos = (torch.abs(h_f)).topk(2*self.k, dim = -2, largest = True, sorted = True)
        o_f_pos = torch.cat( (h_f_pos, idx_pos.type(torch.Tensor).to(h_f_pos.device )) , -2)

        # rpc #
        B, C = x.size(0), x.size(1)
        x = torch.reshape(x, (B*C, -1)).unsqueeze(1)
        o_t = self.RPC(x)
        o_t = torch.reshape(o_t, (B, C, -1, o_t.size(-1)))
        
        o = torch.cat((o_f_pos, o_t),  -2)
        return o


class resConv1dBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, layer_num):
        super(resConv1dBlock, self).__init__()
        '''
            ResNet 1d convolution block
        '''
        self.layer_num = layer_num
        self.conv1 = nn.ModuleList([
            nn.Conv1d(in_channels = in_channels, out_channels = 2 * in_channels, kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) / 2) )
            for i in range(layer_num)])

        self.bn1 = nn.ModuleList([
            nn.BatchNorm1d(2 * in_channels)
            for i in range(layer_num)])

        self.conv2 = nn.ModuleList([ 
            nn.Conv1d(in_channels = 2 * in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = int((kernel_size - 1) / 2) )
            for i in range(layer_num)])

        self.bn2 = nn.ModuleList([
            nn.BatchNorm1d(in_channels)
            for i in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            tmp = F.relu(self.bn1[i](self.conv1[i](x)))
            x = F.relu(self.bn2[i](self.conv2[i](tmp)) + x)
        return x




class UniTS(nn.Module):
#=======initial=========================
    # def __init__(self, input_size=4096, sensor_num=12, layer_num=1,
    #     window_list=[7, 16, 32, 48, 64, 80, 96, 112, 128], stride_list=[3, 8, 16, 24, 32, 40, 48, 56, 64], k_list=[3, 8, 16, 24, 24, 32, 32, 40, 40], out_dim=7, hidden_channel = 48):
#=========initial==============================
    # def __init__(self, input_size=4096, sensor_num=12, layer_num=1,
    #     window_list=[49,256,512,768,1024,1280,1536,1792,2048], stride_list=[24,128,256,384,512,640,768,896,1024], k_list=[24,128,256,384,512,640,768,896,1024], out_dim=7, hidden_channel = 48):
    def __init__(self, input_size=4096, sensor_num=12, layer_num=1,
        window_list=[49,512,1024,1536,2048], stride_list=[24,256,512,768,1024], k_list=[24,256,512,768,1024], out_dim=4, hidden_channel = 48):

        super(UniTS, self).__init__()
        assert len(window_list) == len(stride_list)
        assert len(window_list) == len(k_list)
        self.hidden_channel = hidden_channel
        self.window_list = window_list

        self.ts_encoders = nn.ModuleList([
           TSEnc(window_list[i], stride_list[i], k_list[i]) for i in range(len(window_list))
            ])
        self.num_frequency_channel = [6 * k_list[i] for i in range(len(window_list))]
        self.current_size = [1 + int((input_size - window_list[i]) / stride_list[i])  for i in range(len(window_list))]
        # o.size(): B * C * num_frequency_channel * current_size
        self.multi_channel_fusion = nn.ModuleList([nn.ModuleList() for _ in range(len(window_list))])
        self.conv_branches = nn.ModuleList([nn.ModuleList() for _ in range(len(window_list))])
        self.bns =  nn.ModuleList([nn.BatchNorm1d(self.hidden_channel) for _ in range(len(window_list))])

        self.multi_channel_fusion = nn.ModuleList([nn.Conv2d(in_channels = sensor_num, out_channels = self.hidden_channel,
              kernel_size = (self.num_frequency_channel[i], 1), stride = (1, 1) ) for i in range(len(window_list) ) ])
        self.end_linear = nn.ModuleList([])

        for i in range(len(window_list)):
            scale = 1
            while self.current_size[i] >= 3:
                self.conv_branches[i].append(
                    resConv1dBlock(in_channels = self.hidden_channel * scale,
                     kernel_size = 3, stride = 1, layer_num = layer_num)
                )
                if scale < 2:
                    # scale up the hidden dims for ResNet only once
                    self.conv_branches[i].append(
                        nn.Conv1d(in_channels = self.hidden_channel * scale, out_channels = self.hidden_channel *2* scale, kernel_size = 1, stride = 1)
                    )
                    scale *= 2

                self.conv_branches[i].append(nn.AvgPool1d(kernel_size = 2))
                self.current_size[i] = 1 + int((self.current_size[i] - 2) / 2)

            self.end_linear.append(
                nn.Linear(self.hidden_channel * self.current_size[i] * scale, self.hidden_channel)
            )
        self.classifier = nn.Linear(self.hidden_channel* len(self.window_list), out_dim)

    def forward(self, x):
        #x: B * L * C
        multi_scale_x = []
        # print(f'becore units x {x.shape}')#[batch,12,4096]
        x=x.permute(0,2,1)
        B = x.size(0)
        C = x.size(2)

        for i in range(len(self.current_size)):
            tmp = self.ts_encoders[i](x)
            #tmp: B * C * fc * L'
            tmp = F.relu(self.bns[i](self.multi_channel_fusion[i](tmp).squeeze(2)))

            for j in range(len(self.conv_branches[i])):
                tmp = self.conv_branches[i][j](tmp)
            tmp = tmp.view(B,-1)
            # tmp : B * l'
            tmp = F.relu(self.end_linear[i](tmp))
            multi_scale_x.append(tmp)
        # x1=multi_scale_x
        x1 = torch.cat(multi_scale_x, -1)
        x = self.classifier(x1) 
        return x1,x




# def main():
#     stft_m = STFTNet(input_size = 256, sensor_num = 6, layer_num = 1,
#         window_list = [16, 32, 48], stride_list = [8, 16, 24], k_list = [6, 8, 10], out_dim = 4, hidden_channel = 32).cuda()

#     total_params = sum(p.numel() for p in stft_m.parameters())
#     print(f'{total_params:,} total parameters.')


#     x = torch.zeros(3, 256, 6).cuda()
#     output = stft_m(x)
#     print(output.size())


# if __name__ == '__main__':
#     main()


# model = UniTS()
# print(model)

# from torchinfo import summary

# # 假设你的模型名为 UniTS
# model = UniTS()

# # 打印模型结构
# summary(model, input_size=(16, 12,4096))  # batch_size = 16, 12 个通道，4096 个时间步

# # 计算模型的参数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # 将参数量转换为百万单位
# total_params = count_parameters(model)
# total_params_in_million = total_params / 1e6

# print(f"Model FUSU has {total_params_in_million:.2f} million parameters.")